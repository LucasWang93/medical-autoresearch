"""Track C — multi-task GRPO (shared LoRA adapter).

Differences vs Track B:
  * Tasks are sampled with a UCB bandit over recent val-gain EMA
    (fallback: round-robin for the first pass through all tasks).
  * A single shared LoRA adapter is trained across all selected tasks.
  * Task-weighted loss multiplier (default mirrors GRU study: phen 3x,
    drugrec 2x, los 1.5x, others 1x).
  * Per-task group normalization is automatic — each prompt belongs to
    one task, so grpo_group_advantages centres rewards within the task.
  * Eval is a dict of {task: primary_metric}. The "combined" score is
    the mean of per-task primary metrics (matches multitask iter log).

By default drug_rec is excluded — matching the GRU multitask study,
because the `drugs` feature key fundamentally changes the prompt. The
`--tasks` CLI lets the caller override.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .data import MIMIC4_TASKS, TextSample, load_split
from .grpo_core import GroupSample, grpo_loss, sample_group
from .grpo_single import _eval_quick, ref_logprob_fn
from .metrics import REFERENCE_BEST, primary_metric_name
from .models import (QWEN_MODEL_PATHS, apply_lora, first_cuda_device,
                     gpu_usage_snapshot, load_model_for_training,
                     load_tokenizer)
from .prompts import build_chat_messages
from .rewards import compute_reward
from .runmeta import (JsonlWriter, append_leaderboard, finalize_run_meta,
                      make_run_dir, now_log, write_run_meta)


DEFAULT_TASKS = ["mimic4_mortality", "mimic4_readmission", "mimic4_los", "mimic4_phenotyping"]
DEFAULT_WEIGHTS = {
    "mimic4_mortality": 1.0,
    "mimic4_readmission": 1.0,
    "mimic4_los": 1.5,
    "mimic4_phenotyping": 3.0,
    "mimic4_drugrec": 2.0,
}


class UCBTaskBandit:
    """Simple UCB1 over tasks, with reward = val-primary EMA delta."""
    def __init__(self, tasks: List[str], c: float = 1.0, ema_alpha: float = 0.5):
        self.tasks = list(tasks)
        self.c = c
        self.ema_alpha = ema_alpha
        self.counts = {t: 0 for t in tasks}
        self.rewards = {t: 0.0 for t in tasks}
        self.last_primary: Dict[str, float] = {}
        self.total = 0

    def select(self) -> str:
        for t in self.tasks:
            if self.counts[t] == 0:
                return t
        best_t, best_s = None, -math.inf
        for t in self.tasks:
            avg = self.rewards[t] / self.counts[t]
            bonus = self.c * math.sqrt(math.log(max(1, self.total)) / self.counts[t])
            s = avg + bonus
            if s > best_s:
                best_s, best_t = s, t
        return best_t

    def update_after_eval(self, task: str, primary: float) -> None:
        prev = self.last_primary.get(task, primary)
        delta = primary - prev
        # EMA-smoothed delta as reward
        r = self.rewards[task] / max(1, self.counts[task])
        new_r = (1 - self.ema_alpha) * r + self.ema_alpha * delta
        self.counts[task] += 1
        self.rewards[task] = new_r * self.counts[task]  # restore counts-aligned sum
        self.last_primary[task] = primary
        self.total += 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-tag", required=True, choices=list(QWEN_MODEL_PATHS))
    p.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS),
                   help="comma-separated MIMIC-IV task names")
    p.add_argument("--task-weights", type=str, default="",
                   help='e.g. "mimic4_phenotyping=3.0,mimic4_los=1.5". Missing tasks get 1.0.')
    p.add_argument("--iter-idx", type=int, default=0)
    p.add_argument("--extra-tag", type=str, default="")
    # GRPO
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--kl-beta", type=float, default=0.04)
    p.add_argument("--max-new-tokens", type=int, default=64)
    # Training
    p.add_argument("--total-steps", type=int, default=400)
    p.add_argument("--prompts-per-step", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=80)
    p.add_argument("--eval-n", type=int, default=100)
    p.add_argument("--selection", choices=["bandit", "round_robin"], default="bandit")
    # Model
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-mem-gib-per-gpu", type=int, default=0)
    p.add_argument("--max-visits", type=int, default=10)
    p.add_argument("--max-codes-per-visit", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_weights(spec: str, tasks: List[str]) -> Dict[str, float]:
    w = {t: DEFAULT_WEIGHTS.get(t, 1.0) for t in tasks}
    if not spec:
        return w
    for item in spec.split(","):
        if "=" in item:
            k, v = item.split("=")
            k = k.strip()
            if k in w:
                w[k] = float(v)
    return w


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        if t not in MIMIC4_TASKS:
            raise ValueError(t)
    weights = parse_weights(args.task_weights, tasks)

    run_dir = make_run_dir(
        track="C_multi",
        model_tag=args.model_tag,
        task="all",
        iter_idx=args.iter_idx,
        extra_tag=args.extra_tag,
    )
    model_path = QWEN_MODEL_PATHS[args.model_tag]
    meta = write_run_meta(
        run_dir, track="C_multi",
        model_path=model_path, model_tag=args.model_tag, task="all",
        seed=args.seed, args={**vars(args), "tasks": tasks, "weights": weights},
        iter_idx=args.iter_idx,
    )
    now_log("run_start", run_dir=str(run_dir), tasks=tasks, weights=weights)

    train_splits = {t: load_split(t, "train", seed=args.seed) for t in tasks}
    val_splits = {t: load_split(t, "val", seed=args.seed, max_n=args.eval_n) for t in tasks}
    now_log("data_ready",
            train={t: len(s) for t, s in train_splits.items()},
            val={t: len(s) for t, s in val_splits.items()})

    tokenizer = load_tokenizer(model_path)
    t0 = time.time()
    model = load_model_for_training(
        model_path, max_mem_gib_per_gpu=args.max_mem_gib_per_gpu or None,
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
    bandit = UCBTaskBandit(tasks) if args.selection == "bandit" else None
    rr_idx = 0
    rng = random.Random(args.seed)

    best_combined = -math.inf
    best_per_task: Dict = {}

    model.train()
    step_start = time.time()
    for step in range(1, args.total_steps + 1):
        optim.zero_grad(set_to_none=True)
        step_stats: List[Dict[str, float]] = []
        step_task_counts: Dict[str, int] = {t: 0 for t in tasks}

        for _ in range(args.prompts_per_step):
            if bandit is not None:
                task = bandit.select()
            else:
                task = tasks[rr_idx % len(tasks)]
                rr_idx += 1
            s = rng.choice(train_splits[task])
            messages = build_chat_messages(s, [], args.max_visits, args.max_codes_per_visit)
            try:
                prompt_ids, resp_list = sample_group(
                    model, tokenizer, messages, args.group_size,
                    args.temperature, args.max_new_tokens, device,
                )
            except Exception as e:
                now_log("sample_error", step=step, task=task, err=str(e))
                continue

            group: List[GroupSample] = []
            for resp in resp_list:
                raw = tokenizer.decode(resp, skip_special_tokens=True)
                r = compute_reward(task, raw, s.label)
                group.append(GroupSample(
                    task=task, prompt_ids=prompt_ids.cpu(),
                    response_ids=resp.cpu(), reward=r["reward"],
                    parsed_ok=r["parsed_ok"], label=s.label,
                ))

            rs = np.array([g.reward for g in group])
            if rs.std() < 1e-8:
                continue

            out = grpo_loss(
                model, ref_fn, group, tokenizer.pad_token_id, device,
                kl_beta=args.kl_beta, max_total_len=4096,
            )
            w = weights.get(task, 1.0)
            ((w * out["loss"]) / args.prompts_per_step).backward()
            st = {k: float(v) for k, v in out["stats"].items()}
            st["task"] = task
            st["weight"] = w
            step_stats.append(st)
            step_task_counts[task] += 1

        if not step_stats:
            continue

        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optim.step()
        sched.step()

        row = {
            "step": step,
            "lr": sched.get_last_lr()[0],
            "elapsed_s": round(time.time() - step_start, 1),
            "task_counts": step_task_counts,
        }
        numeric_keys = [k for k in step_stats[0] if k not in ("task",)]
        for k in numeric_keys:
            row[f"{k}_mean"] = float(np.mean([ss[k] for ss in step_stats]))
        train_log.write(row)
        if step % 5 == 0 or step == 1:
            now_log("step", **row)

        if step % args.eval_every == 0 or step == args.total_steps:
            now_log("eval_start", step=step)
            eval_out: Dict[str, Dict[str, float]] = {}
            for task in tasks:
                t_e = time.time()
                m = _eval_quick(
                    model, tokenizer, task, val_splits[task],
                    args.max_visits, args.max_codes_per_visit,
                    max_new_tokens=args.max_new_tokens,
                )
                m["eval_secs"] = round(time.time() - t_e, 1)
                primary = primary_metric_name(task)
                m["primary_name"] = primary
                m["primary_value"] = m.get(primary, 0.0)
                eval_out[task] = m
                if bandit is not None:
                    bandit.update_after_eval(task, m["primary_value"])
            combined = float(np.mean([m["primary_value"] for m in eval_out.values()]))
            train_log.write({"event": "eval", "step": step, "per_task": eval_out, "combined": combined})
            now_log("eval", step=step, combined=combined,
                    **{t: round(eval_out[t]["primary_value"], 4) for t in tasks})
            if combined > best_combined:
                best_combined = combined
                best_per_task = {t: eval_out[t]["primary_value"] for t in tasks}
                model.save_pretrained(run_dir / "lora_adapter_best")
                now_log("new_best", step=step, combined=combined, per_task=best_per_task)

    train_log.close()

    model.save_pretrained(run_dir / "lora_adapter_final")
    tokenizer.save_pretrained(run_dir / "lora_adapter_final")

    final_metrics = {
        "best_combined": best_combined if best_combined > -math.inf else None,
        "best_per_task": best_per_task,
        "train_steps": args.total_steps,
        "final_gpu_usage": gpu_usage_snapshot(),
    }
    finalize_run_meta(run_dir, meta, final_metrics)

    append_leaderboard({
        "timestamp": meta["start_time"],
        "run_id": run_dir.name,
        "track": "C_multi",
        "model": args.model_tag,
        "task": "combined",
        "shots": "",
        "iter": args.iter_idx,
        "metric": "combined_primary_mean",
        "value": best_combined if best_combined > -math.inf else "",
        "delta_vs_ref": "",
        "git_sha": meta["git_sha"],
        "job_id": meta["job_id"],
        "host": meta["host"],
        "seed": args.seed,
        "notes": f"tasks={','.join(tasks)} G={args.group_size} kl={args.kl_beta} steps={args.total_steps} sel={args.selection} tag={args.extra_tag}",
    })
    for t, v in best_per_task.items():
        ref = REFERENCE_BEST.get(t, {})
        base = ref.get("gru_baseline") if ref else None
        delta = round(v - base, 4) if base is not None else ""
        append_leaderboard({
            "timestamp": meta["start_time"],
            "run_id": run_dir.name,
            "track": "C_multi",
            "model": args.model_tag,
            "task": t,
            "shots": "",
            "iter": args.iter_idx,
            "metric": primary_metric_name(t),
            "value": v,
            "delta_vs_ref": delta,
            "git_sha": meta["git_sha"],
            "job_id": meta["job_id"],
            "host": meta["host"],
            "seed": args.seed,
            "notes": f"(multi) combined={round(best_combined,4)} tag={args.extra_tag}",
        })
    now_log("run_done", run_dir=str(run_dir), combined=best_combined, per_task=best_per_task)


if __name__ == "__main__":
    main()
