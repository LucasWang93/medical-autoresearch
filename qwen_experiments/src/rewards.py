"""Per-sample reward functions for GRPO.

Iter 1 redesign (2026-04-22) — fighting mode collapse observed in iter 0.
Root cause: a flat 0/1 correctness reward on highly class-imbalanced
tasks produces a strong gradient toward "always predict majority class"
because the majority-predictor's reward (≈class prior) dominates within
groups sampled on majority-label prompts, and the minority-label prompts
become degenerate (all-"no" groups with std=0) before the policy has a
chance to learn from them.

Changes from iter 0:
  * Binary (mortality, readmission): correct prediction on the minority
    class gives `pos_weight` × reward; correct on majority gives 1.0;
    wrong gives 0. `pos_weight` is per-task since mortality (~1% pos)
    is more imbalanced than readmission (~19% pos).
  * LOS: keep ordinal partial-credit but scale by sqrt(inverse-freq)
    class weights. The iter 0 collapse to "always predict 3-7d" maps
    to f1_macro = 0.182 in every run. Class weights break that tie
    because correct prediction on a rarer bucket is now worth more.
  * Phenotyping: unchanged (F1 already responds to pred structure).
  * Drugrec: tighten length penalty to `>1.5×|gold|` and add a floor
    penalty when `|pred| < 0.5×|gold|` to discourage empty/tiny lists.
  * Format bonus reduced from 0.05 → 0.02, since it's ~50% of the
    wrong-answer reward signal in binary tasks and it was pushing
    the policy to any parseable constant output.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional, Tuple

from .prompts import LOS_OPTIONS, PHENOTYPES, parse_answer


FORMAT_BONUS = 0.02

# Per-task positive-class weight. Mortality pos rate ~1% → heavy boost.
# Readmission pos rate ~19% → moderate boost.
POS_WEIGHT = {
    "mimic4_mortality": 5.0,
    "mimic4_readmission": 2.0,
}

# sqrt(inverse-frequency) class weights for LOS 4 buckets.
# Rough population frequencies: [<3d, 3-7d, 7-14d, >14d] ≈ [0.25, 0.40, 0.25, 0.10].
# sqrt(1/f) then rescale so majority weight = 1.0.
LOS_CLASS_WEIGHT = [1.27, 1.00, 1.27, 2.00]


def _binary_reward(
    parsed: Optional[int], label: int, task: str,
) -> Tuple[float, bool]:
    if parsed is None:
        return 0.0, False
    if int(parsed) != int(label):
        return 0.0, True
    if int(label) == 1:
        return POS_WEIGHT.get(task, 2.0), True
    return 1.0, True


def _los_reward(parsed: Optional[int], label: int) -> Tuple[float, bool]:
    if parsed is None:
        return 0.0, False
    diff = abs(int(parsed) - int(label))
    denom = max(1, len(LOS_OPTIONS) - 1)
    base = 1.0 - diff / denom
    w = LOS_CLASS_WEIGHT[int(label)] if 0 <= int(label) < len(LOS_CLASS_WEIGHT) else 1.0
    return base * w, True


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
    if gold:
        # tighter than iter 0 (was 2*|gold|)
        if len(pred) > 1.5 * len(gold):
            r *= len(gold) / max(1, len(pred))
        elif len(pred) < 0.5 * len(gold):
            r *= len(pred) / max(1, len(gold))
    return r, True


def compute_reward(task: str, raw_output: str, label: Any) -> dict:
    """Return {"reward": float, "parsed_ok": bool, "parsed": Any}."""
    parsed = parse_answer(task, raw_output)
    if task in ("mimic4_mortality", "mimic4_readmission"):
        r, ok = _binary_reward(parsed, label, task)
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
