"""Core GRPO training utilities shared between Track B (single-task) and
Track C (multi-task).

Algorithm (vanilla GRPO, on-policy):

  for each prompt x in batch:
      sample G completions y_1..y_G ~ π_θ(·|x)         # group
      compute reward r_i = R(task, y_i, label(x))
      center within group:   A_i = (r_i - mean) / (std + eps)
      compute loss:
          L = - mean_i [ A_i · Σ_t log π_θ(y_i_t | x, y_<t) ]    (policy grad)
              + β · mean_t KL(π_θ || π_ref)                       (KL regulariser)

Design choices:
  * Per-task group normalization (caller passes groups tagged with task).
  * KL uses the low-variance k3 estimator from GRPO: KL ≈ ρ − log ρ − 1
    where ρ = π_ref / π_θ, computed per response token.
  * Reference = frozen base model (LoRA adapters disabled).
  * Response mask built from generate() boundaries — loss is only over
    the sampled tokens, not the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class GroupSample:
    """One sampled response, inside a group of G samples for a single prompt."""
    task: str
    prompt_ids: torch.LongTensor       # [Lp]
    response_ids: torch.LongTensor     # [Lr]
    reward: float
    parsed_ok: bool
    label: object                      # for logging only


def build_response_batch(
    samples: List[GroupSample],
    pad_token_id: int,
    device: torch.device,
    max_total_len: int = 4096,
) -> Dict[str, torch.Tensor]:
    """Pack samples into {input_ids, attention_mask, response_mask}.

    response_mask is 1 on tokens produced by the policy (i.e. y_i tokens
    after the prompt), 0 on prompt tokens and padding. The log-prob loss
    is summed only over these positions.
    """
    sequences = []
    resp_masks = []
    for s in samples:
        seq = torch.cat([s.prompt_ids, s.response_ids]).to(device)
        if seq.shape[0] > max_total_len:
            # shouldn't happen if generate is bounded, but guard anyway
            overflow = seq.shape[0] - max_total_len
            seq = seq[overflow:]
            resp_start = max(0, s.prompt_ids.shape[0] - overflow)
        else:
            resp_start = s.prompt_ids.shape[0]
        rm = torch.zeros(seq.shape[0], dtype=torch.bool, device=device)
        rm[resp_start:] = True
        sequences.append(seq)
        resp_masks.append(rm)

    max_len = max(s.shape[0] for s in sequences)
    B = len(sequences)
    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long, device=device)
    attn = torch.zeros((B, max_len), dtype=torch.long, device=device)
    rmask = torch.zeros((B, max_len), dtype=torch.bool, device=device)
    for i, (seq, rm) in enumerate(zip(sequences, resp_masks)):
        L = seq.shape[0]
        input_ids[i, :L] = seq
        attn[i, :L] = 1
        rmask[i, :L] = rm
    return {"input_ids": input_ids, "attention_mask": attn, "response_mask": rmask}


def per_token_logprob(
    model,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return per-token log π(y_t | y_<t), shape [B, L-1]."""
    out = model(input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False)
    logits = out.logits[:, :-1, :]             # [B, L-1, V]
    targets = batch["input_ids"][:, 1:]        # [B, L-1]
    logp = F.log_softmax(logits.float(), dim=-1)
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
    return tok_logp


def response_logprob_sum(tok_logp: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Sum log π over response tokens only. Returns [B]."""
    # response_mask is aligned with input_ids (length L); the per-token
    # logprob is for positions 1..L-1 (predicting token t from 0..t-1),
    # so shift mask by 1 to the left.
    mask = response_mask[:, 1:].float()
    return (tok_logp * mask).sum(dim=-1)


def kl_k3(logp_theta_tok: torch.Tensor,
          logp_ref_tok: torch.Tensor,
          response_mask: torch.Tensor) -> torch.Tensor:
    """k3 estimator: KL ≈ exp(r) - r - 1 where r = logp_ref - logp_theta.
    Low variance and always ≥ 0. Returns scalar (mean over response tokens).
    """
    mask = response_mask[:, 1:].float()
    r = (logp_ref_tok - logp_theta_tok).clamp(-20, 20)
    kl = torch.exp(r) - r - 1.0
    denom = mask.sum().clamp_min(1.0)
    return (kl * mask).sum() / denom


def grpo_group_advantages(
    samples: List[GroupSample],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-task, per-prompt group normalization.

    samples arrive grouped as G consecutive samples per prompt (prompts
    may span multiple tasks in multi-task mode). Group id is derived
    from task + position.

    This utility expects caller to pass EXACTLY one group (G samples
    from one prompt) and returns advantages [G].
    """
    rewards = torch.tensor([s.reward for s in samples], dtype=torch.float32)
    mu = rewards.mean()
    sigma = rewards.std(unbiased=False)
    return (rewards - mu) / (sigma + eps)


def grpo_loss(
    model,
    ref_model_callable,
    samples: List[GroupSample],
    pad_token_id: int,
    device: torch.device,
    kl_beta: float = 0.04,
    max_total_len: int = 4096,
) -> Dict[str, torch.Tensor]:
    """Compute GRPO loss over a SINGLE prompt's G samples.

    `ref_model_callable(batch)` must return per-token log-probs without
    gradients — typically realized by disabling LoRA adapters on `model`
    and calling per_token_logprob with torch.no_grad(); see
    run_single_step.
    """
    batch = build_response_batch(samples, pad_token_id, device, max_total_len)
    logp_tok = per_token_logprob(model, batch)                      # with grad
    with torch.no_grad():
        ref_logp_tok = ref_model_callable(batch)                    # no grad

    # Sum log-prob over response tokens
    logp_sum = response_logprob_sum(logp_tok, batch["response_mask"])  # [B]
    advantages = grpo_group_advantages(samples).to(device)             # [B]

    # policy gradient: maximize A · log π  →  minimize - mean(A · log π)
    pg_loss = -(advantages.detach() * logp_sum).mean()

    kl = kl_k3(logp_tok, ref_logp_tok, batch["response_mask"])
    loss = pg_loss + kl_beta * kl

    stats = {
        "loss": loss.detach().float(),
        "pg_loss": pg_loss.detach().float(),
        "kl": kl.detach().float(),
        "mean_reward": torch.tensor(
            sum(s.reward for s in samples) / len(samples),
            dtype=torch.float32,
        ),
        "max_reward": torch.tensor(
            max(s.reward for s in samples), dtype=torch.float32
        ),
        "min_reward": torch.tensor(
            min(s.reward for s in samples), dtype=torch.float32
        ),
        "reward_std": torch.tensor(
            float(torch.tensor([s.reward for s in samples]).std(unbiased=False)),
        ),
        "parse_rate": torch.tensor(
            sum(1 for s in samples if s.parsed_ok) / len(samples),
            dtype=torch.float32,
        ),
    }
    return {"loss": loss, "stats": stats}


def sample_group(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    G: int,
    temperature: float,
    max_new_tokens: int,
    device: torch.device,
) -> Tuple[torch.LongTensor, List[torch.LongTensor]]:
    """Sample G completions for one prompt. Returns (prompt_ids, [resp_ids_i])."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    prompt_ids = enc.input_ids[0].to(device)
    attn = enc.attention_mask[0].to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=prompt_ids.unsqueeze(0),
            attention_mask=attn.unsqueeze(0),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=max(temperature, 1e-5),
            num_return_sequences=G,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    # out is [G, L_full], prompt repeated G times
    Lp = prompt_ids.shape[0]
    resp_list = [out[i, Lp:].detach().clone() for i in range(G)]
    return prompt_ids.detach().clone(), resp_list
