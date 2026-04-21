"""MIMIC-IV → text-sample dataset for Qwen experiments.

Wraps medical-autoresearch/prepare.py::MIMIC4Dataset to:
  1. Load the same patient / split / label logic as the GRU baseline
     (so metrics are directly comparable).
  2. Convert int-encoded visit codes back to string ICD / drug names for
     LLM consumption, using the vocab dicts exposed via our small patch.
  3. Cache processed samples as JSONL on disk so later runs skip the
     (slow) CSV read + join pipeline.

Split protocol: 80/10/10 by patient, matching `_load_mimic4_data`.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]  # medical-autoresearch/
QEXP_ROOT = REPO_ROOT / "qwen_experiments"
CACHE_DIR = QEXP_ROOT / ".cache" / "data"

# ensure medical-autoresearch root on path for `import prepare`
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MIMIC4_TASKS: Tuple[str, ...] = (
    "mimic4_mortality",
    "mimic4_readmission",
    "mimic4_los",
    "mimic4_phenotyping",
    "mimic4_drugrec",
)


@dataclass
class TextSample:
    sample_idx: int
    task: str
    visits: List[Dict[str, List[str]]]   # chronological; each: {"conditions": [...], "procedures": [...], "drugs"?: [...]}
    current_idx: int                      # index of "current" visit in visits
    label: Any                             # task-specific
    meta: Dict[str, Any]                   # e.g. {"n_visits": int}

    def to_json(self) -> Dict[str, Any]:
        return {
            "sample_idx": self.sample_idx,
            "task": self.task,
            "visits": self.visits,
            "current_idx": self.current_idx,
            "label": self.label,
            "meta": self.meta,
        }

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> "TextSample":
        return TextSample(
            sample_idx=obj["sample_idx"],
            task=obj["task"],
            visits=obj["visits"],
            current_idx=obj["current_idx"],
            label=obj["label"],
            meta=obj.get("meta", {}),
        )


def _split_indices(n: int, seed: int = 42) -> Dict[str, List[int]]:
    """Deterministic 80/10/10 split by sample index."""
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_test = n // 10
    n_val = n // 10
    test = sorted(idx[:n_test])
    val = sorted(idx[n_test:n_test + n_val])
    train = sorted(idx[n_test + n_val:])
    return {"train": train, "val": val, "test": test}


def _decode_visits(
    conditions_ids: List[List[int]],
    procedures_ids: List[List[int]],
    inv_diag: Dict[int, str],
    inv_proc: Dict[int, str],
    drugs_hist_ids: Optional[List[List[int]]] = None,
    inv_drug: Optional[Dict[int, str]] = None,
) -> List[Dict[str, List[str]]]:
    visits = []
    n = max(len(conditions_ids), len(procedures_ids))
    for t in range(n):
        cond = conditions_ids[t] if t < len(conditions_ids) else []
        proc = procedures_ids[t] if t < len(procedures_ids) else []
        v = {
            "conditions": [inv_diag[c] for c in cond if c in inv_diag],
            "procedures": [inv_proc[p] for p in proc if p in inv_proc],
        }
        if drugs_hist_ids is not None and inv_drug is not None:
            if t < len(drugs_hist_ids):
                v["drugs"] = [inv_drug[d] for d in drugs_hist_ids[t] if d in inv_drug]
            else:
                v["drugs"] = []
        visits.append(v)
    return visits


def _cache_path(task: str) -> Path:
    return CACHE_DIR / f"{task}.jsonl"


def build_text_dataset(
    task: str,
    data_root: Optional[str] = None,
    max_patients: int = 0,
    seed: int = 42,
    force: bool = False,
) -> List[TextSample]:
    """Build (or load) the full text-sample list for a MIMIC-IV task."""
    path = _cache_path(task)
    if path.exists() and not force:
        samples = []
        with open(path) as f:
            for line in f:
                samples.append(TextSample.from_json(json.loads(line)))
        return samples

    import prepare  # local import after sys.path setup
    spec = prepare.TaskRegistry.get(task)
    ds = prepare.MIMIC4Dataset(
        spec,
        data_root=data_root,
        seed=seed,
        dev=False,
        max_patients=max_patients,
    )
    inv_diag = {v: k for k, v in ds.diag_vocab.items()}
    inv_proc = {v: k for k, v in ds.proc_vocab.items()}
    inv_drug = {v: k for k, v in ds.drug_vocab.items()} if ds.drug_vocab else None

    out: List[TextSample] = []
    for i, raw in enumerate(ds.samples):
        cond = raw.get("conditions", [])
        proc = raw.get("procedures", [])
        drugs_hist = raw.get("drugs_hist", None)
        visits = _decode_visits(cond, proc, inv_diag, inv_proc, drugs_hist, inv_drug)
        if not visits:
            continue

        if task == "mimic4_mortality":
            label: Any = int(raw["mortality"])
        elif task == "mimic4_readmission":
            label = int(raw["readmission"])
        elif task == "mimic4_los":
            label = int(raw["los_bucket"])
        elif task == "mimic4_phenotyping":
            multihot = raw["phenotypes"].tolist()
            label = [int(x) for x in multihot]
        elif task == "mimic4_drugrec":
            multihot = raw["drugs"].tolist()
            # store as string list (drug names with label=1) for readability
            label = [inv_drug[i] for i, v in enumerate(multihot) if v > 0 and inv_drug is not None]
        else:
            raise ValueError(task)

        # For phenotyping the "current" visit's diagnoses were dropped in
        # prepare.py to prevent label leakage; current_idx is the last
        # visit in the history.
        current_idx = len(visits) - 1
        sample = TextSample(
            sample_idx=i,
            task=task,
            visits=visits,
            current_idx=current_idx,
            label=label,
            meta={"n_visits": len(visits)},
        )
        out.append(sample)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in out:
            f.write(json.dumps(s.to_json()) + "\n")
    return out


def split_samples(
    samples: List[TextSample], seed: int = 42,
) -> Dict[str, List[TextSample]]:
    idx_map = _split_indices(len(samples), seed)
    return {k: [samples[i] for i in v] for k, v in idx_map.items()}


def load_split(
    task: str,
    split: str,
    data_root: Optional[str] = None,
    seed: int = 42,
    max_n: int = 0,
) -> List[TextSample]:
    all_samples = build_text_dataset(task, data_root=data_root, seed=seed)
    parts = split_samples(all_samples, seed=seed)
    subset = parts[split]
    if max_n > 0:
        subset = subset[:max_n]
    return subset
