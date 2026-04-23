"""Track B — single-task GRPO training with LoRA.

Per step:
  1. Sample a prompt from the task's train split.
  2. Sample G completions under the current (LoRA-adapted) policy.
  3. Compute reward for each completion and per-group advantage.
  4. Compute log π_θ (with LoRA) and log π_ref (LoRA disabled, frozen base).
  5. Back-propagate GRPO loss (policy gradient + k3 KL).
  6. Periodically evaluate on val split and log.

One run = one (model, task, iter_idx, config) combination. Hyper-parameters
and reproducibility info are all captured in run_meta.json.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .data import MIMIC4_TASKS, TextSample, load_split
from .grpo_core import (GroupSample, grpo_loss, per_token_logprob,
                        sample_group)
from .metrics import (REFERENCE_BEST, binary_metrics, drugrec_metrics,
                      multiclass_metrics, multilabel_metrics,
                      primary_metric_name)
from .models import (QWEN_MODEL_PATHS, apply_lora, first_cuda_device,
                     gpu_usage_snapshot, load_model_for_training,
                     load_tokenizer)
from .prompts import (LOS_OPTIONS, PHENOTYPES, build_chat_messages)
from .rewards import compute_reward
from .runmeta import (JsonlWriter, append_leaderboard, finalize_run_meta,
                      make_run_dir, now_log, write_run_meta)


@contextlib.contextmanager
def adapter_disabled(model):
    """Disable PEFT LoRA adapters so the forward pass uses the base model."""
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
    else:
        # fallback: set all LoRA scaling to 0 — but PEFT's disable_adapter
        # is always present on PeftModel, so this branch shouldn't fire.
        yield


def ref_logprob_fn(model):
    def _fn(batch):
        with adapter_disabled(model):
            return per_token_logprob(model, batch)
    return _fn


# ---- Quick val eval (shared with Track A but simpler — greedy only) ----

def _eval_quick(
    model, tokenizer, task, samples: List[TextSample],
    max_visits: int, max_codes_per_visit: int, max_new_tokens: int,
    demos: Optional[List[TextSample]] = None,
) -> Dict[str, float]:
    device = first_cuda_device(model)
    preds, labels, probs, rewards = [], [], [], []
    pred_multihot, gold_multihot = [], []
    model.eval()
    for s in samples:
        messages = build_chat_messages(s, demos, max_visits, max_codes_per_visit)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc.input_ids.to(device)
        attn = enc.attention_mask.to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids, attention_mask=attn,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        raw = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
        r = compute_reward(task, raw, s.label)
        rewards.append(r["reward"])
        parsed = r["parsed"]
        if task in ("mimic4_mortality", "mimic4_readmission"):
            labels.append(int(s.label)); preds.append(parsed)
            probs.append(1.0 if parsed == 1 else 0.0 if parsed == 0 else 0.5)
        elif task == "mimic4_los":
            labels.append(int(s.label)); preds.append(parsed)
        elif task == "mimic4_phenotyping":
            gold_multihot.append(list(s.label)); pred_multihot.append(parsed)
        elif task == "mimic4_drugrec":
            gold_multihot.append(list(s.label)); pred_multihot.append(parsed)
    model.train()
    if task in ("mimic4_mortality", "mimic4_readmission"):
        m = binary_metrics(probs, preds, labels)
    elif task == "mimic4_los":
        m = multiclass_metrics(None, preds, labels, n_classes=len(LOS_OPTIONS))
    elif task == "mimic4_phenotyping":
        m = multilabel_metrics(pred_multihot, gold_multihot)
    elif task == "mimic4_drugrec":
        m = drugrec_metrics(pred_multihot, gold_multihot)
    m["mean_reward"] = float(np.mean(rewards)) if rewards else 0.0
    return m


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-tag", required=True, choices=list(QWEN_MODEL_PATHS))
    p.add_argument("--task", required=True, choices=list(MIMIC4_TASKS))
    p.add_argument("--iter-idx", type=int, default=0)
    p.add_argument("--extra-tag", type=str, default="")
    # GRPO
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--kl-beta", type=float, default=0.04)
    p.add_argument("--entropy-coef", type=float, default=0.005)
    p.add_argument("--max-new-tokens", type=int, default=64)
    # Training
    p.add_argument("--total-steps", type=int, default=200)
    p.add_argument("--prompts-per-step", type=int, default=2)   # micro-batch of prompts
    p.add_argument("--learning-rate", type=float, default=0.0)  # 0 = per-model default
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-n", type=int, default=100)
    # Model layout
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-mem-gib-per-gpu", type=int, default=0)
    p.add_argument("--max-visits", type=int, default=10)
    p.add_argument("--max-codes-per-visit", type=int, default=20)
    # Few-shot demos (iter 3+): K training-split demos prepended to every
    # prompt (both rollout and eval). Helpful for tasks where the base model
    # fails to follow the answer format (LOS in particular).
    p.add_argument("--demo-k", type=int, default=0)
    # Reproducibility
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Per-model default lr. 9B was observed to overshoot vs its 0-shot baseline
    # at 1e-5 in iter 0. See INSIGHTS.md §F4.
    if args.learning_rate <= 0:
        args.learning_rate = {"qwen35-4b": 5e-6, "qwen35-9b": 1e-6}.get(
            args.model_tag, 5e-6
        )

    run_dir = make_run_dir(
        track="B_single",
        model_tag=args.model_tag,
        task=args.task,
        iter_idx=args.iter_idx,
        extra_tag=args.extra_tag,
    )
    model_path = QWEN_MODEL_PATHS[args.model_tag]
    meta = write_run_meta(
        run_dir, track="B_single",
        model_path=model_path, model_tag=args.model_tag, task=args.task,
        seed=args.seed, args=vars(args), iter_idx=args.iter_idx,
    )
    now_log("run_start", run_dir=str(run_dir), args=vars(args))

    train_samples = load_split(args.task, "train", seed=args.seed)
    val_samples = load_split(args.task, "val", seed=args.seed, max_n=args.eval_n)
    now_log("data_ready", train_n=len(train_samples), val_n=len(val_samples))

    tokenizer = load_tokenizer(model_path)
    t0 = time.time()
    model = load_model_for_training(
        model_path,
        max_mem_gib_per_gpu=args.max_mem_gib_per_gpu or None,
    )
    model, targets = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    model.print_trainable_parameters()
    now_log("model_loaded", sec=round(time.time() - t0, 1),
            lora_targets=targets, gpu_usage=gpu_usage_snapshot())

    device = first_cuda_device(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.learning_rate)
    from transformers import get_linear_schedule_with_warmup
    sched = get_linear_schedule_with_warmup(optim, args.warmup_steps, args.total_steps)

    train_log = JsonlWriter(run_dir / "train_log.jsonl")
    ref_fn = ref_logprob_fn(model)

    rng = random.Random(args.seed)
    best_primary = -math.inf
    best_metrics: Dict = {}
    primary = primary_metric_name(args.task)

    # Fixed demo set, sampled once — kept identical across all prompts and
    # eval so the demos effectively become part of the system prompt.
    demos: List[TextSample] = []
    if args.demo_k > 0:
        demos = rng.sample(train_samples, min(args.demo_k, len(train_samples)))
        now_log("demos_sampled", k=len(demos),
                labels=[str(getattr(d, "label", "?"))[:40] for d in demos])

    model.train()
    step_start = time.time()
    for step in range(1, args.total_steps + 1):
        optim.zero_grad(set_to_none=True)
        step_stats: List[Dict[str, float]] = []
        for _ in range(args.prompts_per_step):
            s = rng.choice(train_samples)
            messages = build_chat_messages(s, demos, args.max_visits, args.max_codes_per_visit)
            try:
                prompt_ids, resp_list = sample_group(
                    model, tokenizer, messages, args.group_size,
                    args.temperature, args.max_new_tokens, device,
                )
            except Exception as e:
                now_log("sample_error", step=step, err=str(e))
                continue

            group: List[GroupSample] = []
            for resp in resp_list:
                raw = tokenizer.decode(resp, skip_special_tokens=True)
                r = compute_reward(args.task, raw, s.label)
                group.append(GroupSample(
                    task=args.task, prompt_ids=prompt_ids.cpu(),
                    response_ids=resp.cpu(), reward=r["reward"],
                    parsed_ok=r["parsed_ok"], label=s.label,
                ))

            # degenerate group (all zero std) → skip, no gradient
            rs = np.array([g.reward for g in group])
            if rs.std() < 1e-8:
                now_log("degen_group_skip", step=step, reward=float(rs.mean()))
                continue

            out = grpo_loss(
                model, ref_fn, group, tokenizer.pad_token_id, device,
                kl_beta=args.kl_beta, entropy_coef=args.entropy_coef,
                max_total_len=4096,
            )
            (out["loss"] / args.prompts_per_step).backward()
            step_stats.append({k: float(v) for k, v in out["stats"].items()})

        if step_stats:
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optim.step()
            sched.step()

            row = {
                "step": step,
                "lr": sched.get_last_lr()[0],
                "elapsed_s": round(time.time() - step_start, 1),
            }
            keys = step_stats[0].keys()
            for k in keys:
                row[k] = float(np.mean([ss[k] for ss in step_stats]))
            train_log.write(row)
            if step % 5 == 0 or step == 1:
                now_log("step", **row)

        if step % args.eval_every == 0 or step == args.total_steps:
            now_log("eval_start", step=step)
            t_eval = time.time()
            m = _eval_quick(
                model, tokenizer, args.task, val_samples,
                args.max_visits, args.max_codes_per_visit,
                max_new_tokens=args.max_new_tokens,
                demos=demos,
            )
            m["step"] = step
            m["eval_secs"] = round(time.time() - t_eval, 1)
            train_log.write({"event": "eval", **m})
            now_log("eval", **m)
            if primary in m and m[primary] > best_primary:
                best_primary = m[primary]
                best_metrics = m
                # save best adapter
                model.save_pretrained(run_dir / "lora_adapter_best")
                now_log("new_best", step=step, **{primary: m[primary]})

    train_log.close()

    # final save
    model.save_pretrained(run_dir / "lora_adapter_final")
    tokenizer.save_pretrained(run_dir / "lora_adapter_final")
    tokenizer.save_pretrained(run_dir / "lora_adapter_best") if (run_dir / "lora_adapter_best").exists() else None

    final_metrics = {
        "best_val": best_metrics,
        "best_primary": best_primary if best_primary > -math.inf else None,
        "train_steps": args.total_steps,
        "final_gpu_usage": gpu_usage_snapshot(),
    }
    finalize_run_meta(run_dir, meta, final_metrics)

    delta = ""
    ref = REFERENCE_BEST.get(args.task, {})
    if best_primary > -math.inf and ref.get("gru_baseline") is not None:
        delta = round(best_primary - ref["gru_baseline"], 4)

    append_leaderboard({
        "timestamp": meta["start_time"],
        "run_id": run_dir.name,
        "track": "B_single",
        "model": args.model_tag,
        "task": args.task,
        "shots": "",
        "iter": args.iter_idx,
        "metric": primary,
        "value": best_primary if best_primary > -math.inf else "",
        "delta_vs_ref": delta,
        "git_sha": meta["git_sha"],
        "job_id": meta["job_id"],
        "host": meta["host"],
        "seed": args.seed,
        "notes": f"G={args.group_size} kl={args.kl_beta} steps={args.total_steps} lr={args.learning_rate} tag={args.extra_tag}",
    })
    now_log("run_done", run_dir=str(run_dir), best=best_metrics)


if __name__ == "__main__":
    main()
