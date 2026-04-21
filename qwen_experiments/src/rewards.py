"""Per-sample reward functions for GRPO.

Binary tasks
  mortality / readmission → 1 if correct else 0. (For logit-prob eval
  we separately compute AUROC — but GRPO uses only the 0/1 correctness
  reward since sampled argmax is what we trained against.)
Multiclass LOS
  ordinal partial credit: 1 − |pred_bucket − true_bucket| / 3.
  Unparseable → 0.
Multilabel phenotyping
  sample-level F1 between pred multihot and gold multihot.
  Unparseable → 0.
Multilabel drugrec
  Jaccard(pred_set, gold_set). Unparseable → 0. Length penalty to avoid
  degenerate "predict all drugs" hacking: if |pred| > 2*|gold| we shrink
  by |gold| / |pred|.
Format bonus
  +0.05 added if parsing succeeded (gives non-zero gradient even on wrong).
"""

from __future__ import annotations

import json
from typing import Any, List, Optional, Tuple

from .prompts import LOS_OPTIONS, PHENOTYPES, parse_answer


FORMAT_BONUS = 0.05


def _binary_reward(parsed: Optional[int], label: int) -> Tuple[float, bool]:
    if parsed is None:
        return 0.0, False
    return (1.0 if int(parsed) == int(label) else 0.0), True


def _los_reward(parsed: Optional[int], label: int) -> Tuple[float, bool]:
    if parsed is None:
        return 0.0, False
    diff = abs(int(parsed) - int(label))
    denom = max(1, len(LOS_OPTIONS) - 1)
    return (1.0 - diff / denom), True


def _multihot_f1(pred: List[int], gold: List[int]) -> float:
    tp = sum(1 for p, g in zip(pred, gold) if p == 1 and g == 1)
    pp = sum(pred)
    pg = sum(gold)
    if pp == 0 and pg == 0:
        return 1.0
    if pp == 0 or pg == 0:
        return 0.0
    prec = tp / pp
    rec = tp / pg
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _phenotype_reward(parsed: Optional[List[int]], label: List[int]) -> Tuple[float, bool]:
    if parsed is None:
        return 0.0, False
    return _multihot_f1(parsed, label), True


def _jaccard(pred: List[str], gold: List[str]) -> float:
    if not pred and not gold:
        return 1.0
    ps, gs = set(pred), set(gold)
    if not ps and not gs:
        return 1.0
    if not ps or not gs:
        return 0.0
    return len(ps & gs) / len(ps | gs)


def _drugrec_reward(
    parsed: Optional[List[str]],
    label: List[str],
) -> Tuple[float, bool]:
    if parsed is None:
        return 0.0, False
    pred = [p.lower() for p in parsed]
    gold = [g.lower() for g in label]
    r = _jaccard(pred, gold)
    # length penalty against trivial "predict everything"
    if gold and len(pred) > 2 * len(gold):
        r *= len(gold) / max(1, len(pred))
    return r, True


def compute_reward(task: str, raw_output: str, label: Any) -> dict:
    """Return {"reward": float, "parsed_ok": bool, "parsed": Any}."""
    parsed = parse_answer(task, raw_output)
    if task in ("mimic4_mortality", "mimic4_readmission"):
        r, ok = _binary_reward(parsed, label)
    elif task == "mimic4_los":
        r, ok = _los_reward(parsed, label)
    elif task == "mimic4_phenotyping":
        r, ok = _phenotype_reward(parsed, label)
    elif task == "mimic4_drugrec":
        r, ok = _drugrec_reward(parsed, label)
    else:
        raise ValueError(task)
    if ok:
        r += FORMAT_BONUS
    return {"reward": float(r), "parsed_ok": bool(ok), "parsed": parsed}
