"""Aggregate metrics — match the GRU-baseline evaluation so numbers are
directly comparable against BASELINES.md / SUMMARY.md.

  * mortality / readmission: primary AUROC (requires probabilities).
    For argmax-only evaluation we fall back to accuracy + F1.
  * los: primary F1_macro. AUROC_macro computed if probabilities given.
  * phenotyping: primary AUROC_macro if probs; otherwise F1_macro.
  * drugrec: primary Jaccard_samples. Also F1_samples.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def _try_sklearn():
    import sklearn.metrics as sm  # noqa: F401
    return sm


def binary_metrics(
    probs: Optional[List[float]],
    preds: List[int],
    labels: List[int],
) -> Dict[str, float]:
    sm = _try_sklearn()
    labels_arr = np.array(labels, dtype=int)
    preds_arr = np.array([0 if p is None else int(p) for p in preds], dtype=int)
    parseable = np.array([p is not None for p in preds], dtype=bool)
    out: Dict[str, float] = {
        "n": int(len(labels_arr)),
        "n_parseable": int(parseable.sum()),
        "parse_rate": float(parseable.mean()) if len(parseable) else 0.0,
        "accuracy": float((preds_arr == labels_arr).mean()),
        "f1": float(sm.f1_score(labels_arr, preds_arr, zero_division=0)),
    }
    if probs is not None and len(set(labels_arr.tolist())) >= 2:
        probs_arr = np.array(probs, dtype=float)
        try:
            out["auroc"] = float(sm.roc_auc_score(labels_arr, probs_arr))
            out["auprc"] = float(sm.average_precision_score(labels_arr, probs_arr))
        except Exception:
            pass
    return out


def multiclass_metrics(
    probs: Optional[np.ndarray],   # [N, K]
    preds: List[Optional[int]],
    labels: List[int],
    n_classes: int,
) -> Dict[str, float]:
    sm = _try_sklearn()
    labels_arr = np.array(labels, dtype=int)
    preds_arr = np.array([0 if p is None else int(p) for p in preds], dtype=int)
    parseable = np.array([p is not None for p in preds], dtype=bool)
    out = {
        "n": int(len(labels_arr)),
        "n_parseable": int(parseable.sum()),
        "parse_rate": float(parseable.mean()) if len(parseable) else 0.0,
        "accuracy": float((preds_arr == labels_arr).mean()),
        "f1_macro": float(sm.f1_score(labels_arr, preds_arr, labels=list(range(n_classes)), average="macro", zero_division=0)),
        "f1_weighted": float(sm.f1_score(labels_arr, preds_arr, labels=list(range(n_classes)), average="weighted", zero_division=0)),
    }
    if probs is not None and len(set(labels_arr.tolist())) >= 2:
        try:
            out["auroc_macro"] = float(sm.roc_auc_score(labels_arr, probs, multi_class="ovr", average="macro"))
        except Exception:
            pass
    return out


def multilabel_metrics(
    pred_multihot: List[Optional[List[int]]],
    gold_multihot: List[List[int]],
    probs: Optional[np.ndarray] = None,  # [N, L]
) -> Dict[str, float]:
    sm = _try_sklearn()
    L = len(gold_multihot[0]) if gold_multihot else 0
    Y = np.array(gold_multihot, dtype=int)
    P = np.array([
        [0] * L if p is None else p
        for p in pred_multihot
    ], dtype=int)
    parseable = np.array([p is not None for p in pred_multihot], dtype=bool)

    # sample-level Jaccard + F1
    jacs, f1s = [], []
    for pred, gold in zip(P, Y):
        ps, gs = set(np.where(pred)[0].tolist()), set(np.where(gold)[0].tolist())
        if not ps and not gs:
            jacs.append(1.0); f1s.append(1.0); continue
        if not ps or not gs:
            jacs.append(0.0); f1s.append(0.0); continue
        inter = len(ps & gs)
        jacs.append(inter / len(ps | gs))
        prec = inter / len(ps)
        rec = inter / len(gs)
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    out = {
        "n": int(len(Y)),
        "n_parseable": int(parseable.sum()),
        "parse_rate": float(parseable.mean()) if len(parseable) else 0.0,
        "jaccard_samples": float(np.mean(jacs)) if jacs else 0.0,
        "f1_samples": float(np.mean(f1s)) if f1s else 0.0,
    }
    if probs is not None:
        try:
            out["auroc_macro"] = float(sm.roc_auc_score(Y, probs, average="macro"))
            out["auprc_macro"] = float(sm.average_precision_score(Y, probs, average="macro"))
        except Exception:
            pass
    return out


def drugrec_metrics(
    pred_lists: List[Optional[List[str]]],
    gold_lists: List[List[str]],
) -> Dict[str, float]:
    parseable = np.array([p is not None for p in pred_lists], dtype=bool)
    jacs, f1s = [], []
    for pred, gold in zip(pred_lists, gold_lists):
        ps = set([x.lower() for x in pred]) if pred else set()
        gs = set([x.lower() for x in gold]) if gold else set()
        if not ps and not gs:
            jacs.append(1.0); f1s.append(1.0); continue
        if not ps or not gs:
            jacs.append(0.0); f1s.append(0.0); continue
        inter = len(ps & gs)
        jacs.append(inter / len(ps | gs))
        prec = inter / len(ps)
        rec = inter / len(gs)
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return {
        "n": int(len(gold_lists)),
        "n_parseable": int(parseable.sum()),
        "parse_rate": float(parseable.mean()) if len(parseable) else 0.0,
        "jaccard_samples": float(np.mean(jacs)) if jacs else 0.0,
        "f1_samples": float(np.mean(f1s)) if f1s else 0.0,
    }


def primary_metric_name(task: str) -> str:
    return {
        "mimic4_mortality": "auroc",
        "mimic4_readmission": "auroc",
        "mimic4_los": "f1_macro",
        "mimic4_phenotyping": "f1_samples",
        "mimic4_drugrec": "jaccard_samples",
    }[task]


REFERENCE_BEST = {
    "mimic4_mortality": {"metric": "auroc", "gru_baseline": 0.9611, "gru_best": 0.9726, "multitask_best": 0.9755},
    "mimic4_readmission": {"metric": "auroc", "gru_baseline": 0.6688, "gru_best": 0.6981, "multitask_best": 0.7117},
    "mimic4_los": {"metric": "f1_macro", "gru_baseline": 0.4814, "gru_best": 0.5406, "multitask_best": 0.5548},
    "mimic4_phenotyping": {"metric": "f1_samples", "gru_baseline": None, "gru_best": None, "multitask_best": None},
    "mimic4_drugrec": {"metric": "jaccard_samples", "gru_baseline": 0.1661, "gru_best": 0.2052, "multitask_best": None},
}
