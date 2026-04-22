"""Track A — few-shot baseline evaluation (inference only, no training).

For each (model, task, shots) tuple:
  1. Load N_eval samples from the test split (default 500; all for small
     splits).
  2. Build chat messages with `shots` demonstrations drawn from train split.
  3. For binary tasks, evaluate via first-token probability over
     {yes, no}-lead tokens → gives proper AUROC.
  4. For multiclass LOS, use first-token probability over the 4 option
     leads → gives per-class probabilities (→ AUROC_macro too).
  5. For multilabel phenotyping / drugrec, generate a short sequence
     and parse the JSON list.

Everything — per-sample prediction, probability, reward — is streamed
to `predictions.jsonl`, and the aggregate lands in `metrics.json` plus a
row in `leaderboard.tsv`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .data import MIMIC4_TASKS, TextSample, load_split
from .metrics import (REFERENCE_BEST, binary_metrics, drugrec_metrics,
                      multiclass_metrics, multilabel_metrics,
                      primary_metric_name)
from .models import (QWEN_MODEL_PATHS, first_cuda_device, gpu_usage_snapshot,
                     load_model_for_inference, load_tokenizer)
from .prompts import (LOS_OPTIONS, PHENOTYPES, build_chat_messages,
                      parse_answer)
from .rewards import compute_reward
from .runmeta import (JsonlWriter, append_leaderboard, finalize_run_meta,
                      make_run_dir, now_log, write_run_meta)


# ---- First-token probability helpers ----

def _first_token_ids(tokenizer, words: List[str]) -> Dict[str, List[int]]:
    """For each word, get all plausible "leading" token ids (with and
    without leading space — chat templates vary)."""
    out: Dict[str, List[int]] = {}
    for w in words:
        ids = set()
        for variant in (w, " " + w, "\n" + w, w.capitalize(), w.upper()):
            tok = tokenizer(variant, add_special_tokens=False).input_ids
            if tok:
                ids.add(tok[0])
        out[w] = sorted(ids)
    return out


def _next_token_logits(model, tokenizer, messages: List[Dict[str, str]]) -> torch.Tensor:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    dev = first_cuda_device(model)
    input_ids = enc.input_ids.to(dev)
    attn = enc.attention_mask.to(dev)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logits = out.logits[0, -1, :].float().cpu()
    return logits


def _class_probs(logits: torch.Tensor, class_to_ids: Dict[str, List[int]]) -> Dict[str, float]:
    """Softmax over the max-logit id of each class group."""
    probs_all = torch.softmax(logits, dim=-1)
    group = {}
    for cls, ids in class_to_ids.items():
        if not ids:
            group[cls] = 0.0
            continue
        group[cls] = float(probs_all[ids].max().item())
    s = sum(group.values())
    if s <= 0:
        return {k: 1.0 / len(group) for k in group}
    return {k: v / s for k, v in group.items()}


def _generate_text(
    model, tokenizer, messages, max_new_tokens: int, temperature: float = 0.0,
) -> str:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    dev = first_cuda_device(model)
    input_ids = enc.input_ids.to(dev)
    attn = enc.attention_mask.to(dev)
    do_sample = temperature > 0
    gen_kw = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kw["temperature"] = temperature
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids, attention_mask=attn, **gen_kw,
        )
    new = out[0, input_ids.shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True)


# ---- Task evaluators ----

def _prompt_hash(messages) -> str:
    h = hashlib.sha1()
    for m in messages:
        h.update((m["role"] + ":" + m["content"]).encode("utf-8"))
    return h.hexdigest()[:12]


def _prepare_demos(
    train_samples: List[TextSample],
    shots: int,
    label_key_seed: int,
    task: str,
) -> List[TextSample]:
    if shots <= 0:
        return []
    rng = random.Random(label_key_seed)
    pool = list(train_samples)
    rng.shuffle(pool)
    # for binary, try to balance positives/negatives
    if task in ("mimic4_mortality", "mimic4_readmission") and shots >= 2:
        pos = [s for s in pool if int(s.label) == 1][:shots // 2 + 1]
        neg = [s for s in pool if int(s.label) == 0][:shots // 2 + 1]
        demos = (pos + neg)[:shots]
        rng.shuffle(demos)
        return demos
    return pool[:shots]


def eval_binary(
    model, tokenizer, task: str, test_samples: List[TextSample],
    train_samples: List[TextSample], shots: int, max_visits: int,
    max_codes_per_visit: int, seed: int, writer: JsonlWriter,
):
    class_to_ids = _first_token_ids(tokenizer, ["yes", "no"])
    probs, preds, labels = [], [], []
    rewards = []
    t0 = time.time()
    for i, s in enumerate(test_samples):
        demos = _prepare_demos(train_samples, shots, seed + i, task)
        messages = build_chat_messages(s, demos, max_visits, max_codes_per_visit)
        try:
            logits = _next_token_logits(model, tokenizer, messages)
            cp = _class_probs(logits, class_to_ids)
            p_yes = cp["yes"]
            pred = 1 if p_yes >= 0.5 else 0
        except Exception as e:
            now_log("eval_error", sample_idx=s.sample_idx, err=str(e))
            p_yes, pred = 0.5, None
        probs.append(float(p_yes))
        preds.append(pred)
        labels.append(int(s.label))
        # reward: use argmax correctness + format ok
        r = compute_reward(task, "yes" if pred == 1 else ("no" if pred == 0 else ""), s.label)
        rewards.append(r["reward"])
        writer.write({
            "sample_idx": s.sample_idx,
            "task": task,
            "prompt_hash": _prompt_hash(messages),
            "p_yes": p_yes,
            "pred": pred,
            "label": int(s.label),
            "reward": r["reward"],
            "parsed_ok": r["parsed_ok"],
        })
        if (i + 1) % 25 == 0:
            now_log("eval_progress", task=task, shots=shots, done=i + 1,
                    total=len(test_samples), secs=round(time.time() - t0, 1))
    return binary_metrics(probs, preds, labels) | {"mean_reward": float(np.mean(rewards))}


def eval_los(
    model, tokenizer, task: str, test_samples, train_samples,
    shots, max_visits, max_codes_per_visit, seed, writer,
):
    class_to_ids = _first_token_ids(tokenizer, LOS_OPTIONS)
    probs_rows = []
    preds, labels, rewards = [], [], []
    t0 = time.time()
    for i, s in enumerate(test_samples):
        demos = _prepare_demos(train_samples, shots, seed + i, task)
        messages = build_chat_messages(s, demos, max_visits, max_codes_per_visit)
        try:
            logits = _next_token_logits(model, tokenizer, messages)
            cp = _class_probs(logits, class_to_ids)
            pv = [cp[o] for o in LOS_OPTIONS]
            pred = int(np.argmax(pv))
        except Exception as e:
            now_log("eval_error", sample_idx=s.sample_idx, err=str(e))
            pv = [0.25] * len(LOS_OPTIONS); pred = None
        probs_rows.append(pv)
        preds.append(pred)
        labels.append(int(s.label))
        raw = LOS_OPTIONS[pred] if pred is not None else ""
        r = compute_reward(task, raw, s.label)
        rewards.append(r["reward"])
        writer.write({
            "sample_idx": s.sample_idx,
            "task": task,
            "prompt_hash": _prompt_hash(messages),
            "probs": pv,
            "pred": pred,
            "label": int(s.label),
            "reward": r["reward"],
            "parsed_ok": r["parsed_ok"],
        })
        if (i + 1) % 25 == 0:
            now_log("eval_progress", task=task, done=i + 1,
                    total=len(test_samples), secs=round(time.time() - t0, 1))
    return multiclass_metrics(
        np.array(probs_rows, dtype=float), preds, labels, n_classes=len(LOS_OPTIONS),
    ) | {"mean_reward": float(np.mean(rewards))}


def eval_phenotyping(
    model, tokenizer, task, test_samples, train_samples,
    shots, max_visits, max_codes_per_visit, seed, writer,
):
    preds, labels, rewards = [], [], []
    t0 = time.time()
    for i, s in enumerate(test_samples):
        demos = _prepare_demos(train_samples, shots, seed + i, task)
        messages = build_chat_messages(s, demos, max_visits, max_codes_per_visit)
        try:
            raw = _generate_text(model, tokenizer, messages, max_new_tokens=128)
        except Exception as e:
            now_log("eval_error", sample_idx=s.sample_idx, err=str(e))
            raw = ""
        r = compute_reward(task, raw, s.label)
        preds.append(r["parsed"])
        labels.append(list(s.label))
        rewards.append(r["reward"])
        writer.write({
            "sample_idx": s.sample_idx,
            "task": task,
            "prompt_hash": _prompt_hash(messages),
            "raw_output": raw,
            "parsed": r["parsed"],
            "label": list(s.label),
            "reward": r["reward"],
            "parsed_ok": r["parsed_ok"],
        })
        if (i + 1) % 10 == 0:
            now_log("eval_progress", task=task, done=i + 1,
                    total=len(test_samples), secs=round(time.time() - t0, 1))
    return multilabel_metrics(preds, labels) | {"mean_reward": float(np.mean(rewards))}


def eval_drugrec(
    model, tokenizer, task, test_samples, train_samples,
    shots, max_visits, max_codes_per_visit, seed, writer,
):
    preds, labels, rewards = [], [], []
    t0 = time.time()
    for i, s in enumerate(test_samples):
        demos = _prepare_demos(train_samples, shots, seed + i, task)
        messages = build_chat_messages(s, demos, max_visits, max_codes_per_visit)
        try:
            raw = _generate_text(model, tokenizer, messages, max_new_tokens=192)
        except Exception as e:
            now_log("eval_error", sample_idx=s.sample_idx, err=str(e))
            raw = ""
        r = compute_reward(task, raw, s.label)
        preds.append(r["parsed"])
        labels.append(list(s.label))
        rewards.append(r["reward"])
        writer.write({
            "sample_idx": s.sample_idx,
            "task": task,
            "prompt_hash": _prompt_hash(messages),
            "raw_output": raw,
            "parsed": r["parsed"],
            "label": list(s.label),
            "reward": r["reward"],
            "parsed_ok": r["parsed_ok"],
        })
        if (i + 1) % 10 == 0:
            now_log("eval_progress", task=task, done=i + 1,
                    total=len(test_samples), secs=round(time.time() - t0, 1))
    return drugrec_metrics(preds, labels) | {"mean_reward": float(np.mean(rewards))}


# ---- CLI ----

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-tag", required=True, choices=list(QWEN_MODEL_PATHS))
    p.add_argument("--task", required=True, choices=list(MIMIC4_TASKS))
    p.add_argument("--shots", type=int, default=0)
    p.add_argument("--n-eval", type=int, default=500,
                   help="Cap on test samples evaluated; 0 = all.")
    p.add_argument("--max-visits", type=int, default=10)
    p.add_argument("--max-codes-per-visit", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-mem-gib-per-gpu", type=int, default=0,
                   help="0 = no cap (let device_map=balanced decide).")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    run_dir = make_run_dir(
        track="A_baseline",
        model_tag=args.model_tag,
        task=args.task,
        shots=args.shots,
    )
    model_path = QWEN_MODEL_PATHS[args.model_tag]
    meta = write_run_meta(
        run_dir,
        track="A_baseline",
        model_path=model_path,
        model_tag=args.model_tag,
        task=args.task,
        seed=args.seed,
        args=vars(args),
        shots=args.shots,
    )
    now_log("run_start", run_dir=str(run_dir), args=vars(args))

    test_samples = load_split(args.task, "test", seed=args.seed, max_n=args.n_eval)
    train_samples = load_split(args.task, "train", seed=args.seed, max_n=max(200, args.shots * 20))
    now_log("data_ready", test_n=len(test_samples), train_n=len(train_samples))

    tokenizer = load_tokenizer(model_path)
    t_load = time.time()
    model = load_model_for_inference(
        model_path,
        max_mem_gib_per_gpu=args.max_mem_gib_per_gpu or None,
    )
    now_log("model_loaded", sec=round(time.time() - t_load, 1),
            gpu_usage=gpu_usage_snapshot())

    writer = JsonlWriter(run_dir / "predictions.jsonl")

    if args.task in ("mimic4_mortality", "mimic4_readmission"):
        metrics = eval_binary(
            model, tokenizer, args.task, test_samples, train_samples,
            args.shots, args.max_visits, args.max_codes_per_visit,
            args.seed, writer,
        )
    elif args.task == "mimic4_los":
        metrics = eval_los(
            model, tokenizer, args.task, test_samples, train_samples,
            args.shots, args.max_visits, args.max_codes_per_visit,
            args.seed, writer,
        )
    elif args.task == "mimic4_phenotyping":
        metrics = eval_phenotyping(
            model, tokenizer, args.task, test_samples, train_samples,
            args.shots, args.max_visits, args.max_codes_per_visit,
            args.seed, writer,
        )
    elif args.task == "mimic4_drugrec":
        metrics = eval_drugrec(
            model, tokenizer, args.task, test_samples, train_samples,
            args.shots, args.max_visits, args.max_codes_per_visit,
            args.seed, writer,
        )
    else:
        raise ValueError(args.task)
    writer.close()

    primary = primary_metric_name(args.task)
    ref = REFERENCE_BEST.get(args.task, {})
    delta = ""
    if primary in metrics and ref.get("gru_baseline") is not None:
        delta = round(metrics[primary] - ref["gru_baseline"], 4)
    finalize_run_meta(run_dir, meta, metrics)
    append_leaderboard({
        "timestamp": meta["start_time"],
        "run_id": run_dir.name,
        "track": "A_baseline",
        "model": args.model_tag,
        "task": args.task,
        "shots": args.shots,
        "iter": "",
        "metric": primary,
        "value": metrics.get(primary, ""),
        "delta_vs_ref": delta,
        "git_sha": meta["git_sha"],
        "job_id": meta["job_id"],
        "host": meta["host"],
        "seed": args.seed,
        "notes": f"n={metrics['n']} parse={metrics.get('parse_rate', 0):.2f}",
    })
    now_log("run_done", run_dir=str(run_dir), metrics=metrics)


if __name__ == "__main__":
    main()
