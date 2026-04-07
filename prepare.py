"""Fixed infrastructure for Medical AutoResearch. DO NOT MODIFY.

This file provides:
1. TaskRegistry — multi-task interface with future autonomous selection support
2. Synthetic EHR data generation (works without MIMIC access)
3. PyHealth data loading (when MIMIC is available)
4. Standardized evaluation harness
5. Results output matching autoresearch format

The agent modifies train.py, not this file.
"""

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ============================================================
# Constants
# ============================================================

TIME_BUDGET_SECONDS = 300  # 5-minute training budget (wall clock)
EVAL_TIMEOUT_SECONDS = 120
SEED = 42
DATA_DIR = Path(os.environ.get(
    "MEDICAL_AR_DATA",
    str(Path.home() / ".cache" / "medical-autoresearch"),
))
PYHEALTH_ROOT = Path(__file__).resolve().parent.parent / "PyHealth"

# ============================================================
# Task Specification & Registry
# ============================================================

@dataclass
class TaskSpec:
    """Specification for a clinical prediction task.

    This is the core abstraction that enables multi-task autonomous
    selection. Each task fully describes its data schema, evaluation
    protocol, and reward structure so that an RL agent can be trained
    and evaluated without any task-specific code outside of train.py.
    """
    name: str
    task_type: str                       # "multilabel" | "binary" | "multiclass"
    description: str

    # Data schema
    feature_keys: List[str]              # input feature names
    label_key: str                       # output label name
    feature_dims: Dict[str, int] = field(default_factory=dict)
    # e.g. {"conditions": 500, "procedures": 200, "drugs_hist": 150}
    label_dim: int = 2                   # number of output classes / labels

    # Evaluation
    primary_metric: str = "jaccard_samples"
    metric_direction: str = "max"        # "max" or "min"
    metrics: List[str] = field(default_factory=lambda: ["jaccard_samples"])

    # RL reward design
    reward_components: Dict[str, float] = field(default_factory=dict)
    # e.g. {"jaccard": 1.0, "ddi_penalty": -0.5}

    # Data generation params (for synthetic mode)
    n_archetypes: int = 10               # disease archetypes
    max_visits: int = 15
    max_codes_per_visit: int = 12

    # PyHealth integration (for real data mode)
    pyhealth_dataset: str = ""           # e.g. "mimic3"
    pyhealth_task: str = ""              # e.g. "drug_recommendation"


class TaskRegistry:
    """Registry of clinical tasks with multi-task selection interface.

    MVP: single-task selection via get().
    Future: autonomous multi-task selection via select_tasks() and
    MultiTaskLoader for bandit-based task curriculum.
    """
    _tasks: Dict[str, TaskSpec] = {}

    # ------ Core API ------

    @classmethod
    def register(cls, spec: TaskSpec) -> None:
        cls._tasks[spec.name] = spec

    @classmethod
    def get(cls, name: str) -> TaskSpec:
        if name not in cls._tasks:
            raise KeyError(
                f"Task '{name}' not found. Available: {cls.list_tasks()}"
            )
        return cls._tasks[name]

    @classmethod
    def list_tasks(cls) -> List[str]:
        return list(cls._tasks.keys())

    @classmethod
    def list_specs(cls) -> List[TaskSpec]:
        return list(cls._tasks.values())

    # ------ Multi-task interface (reserved for future) ------

    @classmethod
    def select_tasks(
        cls,
        names: Optional[List[str]] = None,
    ) -> List[TaskSpec]:
        """Select multiple tasks. Pass None for all registered tasks."""
        if names is None:
            names = cls.list_tasks()
        return [cls.get(n) for n in names]

    @classmethod
    def get_multi_task_loader(
        cls,
        task_specs: List[TaskSpec],
        strategy: str = "round_robin",
        batch_size: int = 32,
        **kwargs,
    ) -> "MultiTaskLoader":
        """Create a loader that serves batches from multiple tasks.

        Strategies:
          - round_robin: cycle through tasks deterministically
          - proportional: sample proportional to dataset size
          - bandit: UCB-based selection driven by reward signal

        The agent in train.py can call selector.update(task, reward)
        to feed learning progress back into task selection.
        """
        loaders = {}
        for spec in task_specs:
            train_dl, _, _ = load_task_data(spec.name, batch_size=batch_size)
            loaders[spec.name] = train_dl
        return MultiTaskLoader(loaders, task_specs, strategy=strategy, **kwargs)


class MultiTaskLoader:
    """Iterates over multiple task dataloaders with a selection strategy.

    Reserved interface — the agent can use this in train.py to
    autonomously choose which task to train on each step.
    """

    def __init__(
        self,
        loaders: Dict[str, DataLoader],
        specs: List[TaskSpec],
        strategy: str = "round_robin",
        ucb_c: float = 1.0,
    ):
        self.loaders = loaders
        self.specs = {s.name: s for s in specs}
        self.strategy = strategy
        self.ucb_c = ucb_c

        self._iters: Dict[str, Any] = {}
        self._task_names = list(loaders.keys())
        self._idx = 0

        # Bandit state
        self._counts = {n: 0 for n in self._task_names}
        self._rewards = {n: 0.0 for n in self._task_names}
        self._total = 0

    def _get_iter(self, name: str):
        if name not in self._iters:
            self._iters[name] = iter(self.loaders[name])
        return self._iters[name]

    def _next_batch(self, name: str):
        it = self._get_iter(name)
        try:
            return next(it)
        except StopIteration:
            self._iters[name] = iter(self.loaders[name])
            return next(self._iters[name])

    def _select_task(self) -> str:
        if self.strategy == "round_robin":
            name = self._task_names[self._idx % len(self._task_names)]
            self._idx += 1
            return name
        elif self.strategy == "proportional":
            sizes = [len(self.loaders[n].dataset) for n in self._task_names]
            total = sum(sizes)
            probs = [s / total for s in sizes]
            return np.random.choice(self._task_names, p=probs)
        elif self.strategy == "bandit":
            return self._ucb_select()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _ucb_select(self) -> str:
        # Ensure each task tried at least once
        for n in self._task_names:
            if self._counts[n] == 0:
                return n
        # UCB1
        best_name, best_score = None, -float("inf")
        for n in self._task_names:
            avg = self._rewards[n] / self._counts[n]
            bonus = self.ucb_c * math.sqrt(
                math.log(self._total) / self._counts[n]
            )
            score = avg + bonus
            if score > best_score:
                best_score = score
                best_name = n
        return best_name

    def sample(self) -> Tuple[str, dict, TaskSpec]:
        """Sample a (task_name, batch, task_spec) tuple."""
        name = self._select_task()
        batch = self._next_batch(name)
        return name, batch, self.specs[name]

    def update_reward(self, task_name: str, reward: float) -> None:
        """Feed reward signal back for bandit-based selection."""
        self._counts[task_name] += 1
        self._rewards[task_name] += reward
        self._total += 1

    def get_task_iter(self, task_name: str):
        """Get an infinite iterator for a specific task."""
        while True:
            yield self._next_batch(task_name)


# ============================================================
# Register built-in tasks
# ============================================================

TaskRegistry.register(TaskSpec(
    name="drug_recommendation",
    task_type="multilabel",
    description=(
        "Predict drug combinations for patients given diagnosis and "
        "procedure history. Core RL challenge: sequential drug decisions "
        "with DDI safety constraints."
    ),
    feature_keys=["conditions", "procedures", "drugs_hist"],
    label_key="drugs",
    feature_dims={"conditions": 500, "procedures": 200, "drugs_hist": 150},
    label_dim=150,
    primary_metric="jaccard_samples",
    metric_direction="max",
    metrics=["jaccard_samples", "f1_samples", "pr_auc_samples"],
    reward_components={"jaccard_samples": 1.0, "ddi_rate": -0.5},
    n_archetypes=10,
    max_visits=15,
    max_codes_per_visit=12,
    pyhealth_dataset="mimic3",
    pyhealth_task="drug_recommendation",
))

TaskRegistry.register(TaskSpec(
    name="mortality_prediction",
    task_type="binary",
    description=(
        "Predict in-hospital mortality from sequential EHR data. "
        "RL angle: adaptive attention over visit history."
    ),
    feature_keys=["conditions", "procedures"],
    label_key="mortality",
    feature_dims={"conditions": 500, "procedures": 200},
    label_dim=2,
    primary_metric="auroc",
    metric_direction="max",
    metrics=["auroc", "auprc", "f1"],
    reward_components={"auroc": 1.0},
    n_archetypes=8,
    max_visits=20,
    max_codes_per_visit=15,
    pyhealth_dataset="mimic3",
    pyhealth_task="mortality_prediction",
))

TaskRegistry.register(TaskSpec(
    name="readmission_prediction",
    task_type="binary",
    description=(
        "Predict 30-day hospital readmission. "
        "RL angle: learning which visit patterns signal readmission risk."
    ),
    feature_keys=["conditions", "procedures", "drugs"],
    label_key="readmission",
    feature_dims={"conditions": 500, "procedures": 200, "drugs": 150},
    label_dim=2,
    primary_metric="auroc",
    metric_direction="max",
    metrics=["auroc", "auprc", "f1"],
    reward_components={"auroc": 1.0},
    n_archetypes=8,
    max_visits=20,
    max_codes_per_visit=15,
    pyhealth_dataset="mimic3",
    pyhealth_task="readmission_prediction",
))

TaskRegistry.register(TaskSpec(
    name="length_of_stay",
    task_type="multiclass",
    description=(
        "Predict length-of-stay bucket (short / medium / long). "
        "RL angle: sequential feature selection for triage."
    ),
    feature_keys=["conditions", "procedures"],
    label_key="los",
    feature_dims={"conditions": 500, "procedures": 200},
    label_dim=3,
    primary_metric="f1_macro",
    metric_direction="max",
    metrics=["f1_macro", "accuracy", "auroc_macro"],
    reward_components={"f1_macro": 1.0},
    n_archetypes=6,
    max_visits=20,
    max_codes_per_visit=15,
    pyhealth_dataset="mimic3",
    pyhealth_task="length_of_stay_prediction",
))


# ============================================================
# Synthetic EHR Data (works without MIMIC)
# ============================================================

class SyntheticEHRDataset(Dataset):
    """Synthetic EHR dataset with learnable structure.

    Creates disease *archetypes* — clusters of conditions that co-occur
    and map to specific drug combinations. This gives the RL agent a
    non-trivial pattern to discover while keeping the system fully
    self-contained.

    For drug_recommendation, the synthetic data also includes a DDI
    matrix so that DDI-aware reward shaping is possible.
    """

    def __init__(self, task_spec: TaskSpec, n_patients: int = 2000, seed: int = 42):
        super().__init__()
        self.spec = task_spec
        self.rng = np.random.RandomState(seed)
        self.samples: List[Dict[str, Any]] = []

        # Build archetype structure
        self._build_archetypes()

        # Generate patient trajectories
        for _ in range(n_patients):
            self._generate_patient()

    def _build_archetypes(self):
        """Create disease archetypes with condition→drug mappings."""
        spec = self.spec
        self.archetypes = []

        for i in range(spec.n_archetypes):
            n_conds = self.rng.randint(5, 20)
            cond_pool = self.rng.choice(
                spec.feature_dims.get("conditions", 500),
                size=n_conds, replace=False,
            ).tolist()

            n_procs = self.rng.randint(3, 10)
            proc_pool = self.rng.choice(
                spec.feature_dims.get("procedures", 200),
                size=n_procs, replace=False,
            ).tolist()

            if spec.task_type == "multilabel":
                n_drugs = self.rng.randint(3, 8)
                drug_pool = self.rng.choice(
                    spec.label_dim, size=n_drugs, replace=False,
                ).tolist()
            else:
                drug_pool = []

            # For binary/multiclass tasks, assign a label tendency
            label_tendency = i % spec.label_dim

            self.archetypes.append({
                "conditions": cond_pool,
                "procedures": proc_pool,
                "drugs": drug_pool,
                "label_tendency": label_tendency,
            })

        # DDI matrix (for drug_recommendation)
        if spec.task_type == "multilabel":
            n_drugs = spec.label_dim
            self.ddi_matrix = np.zeros((n_drugs, n_drugs), dtype=np.float32)
            n_interactions = int(n_drugs * (n_drugs - 1) * 0.05)
            for _ in range(n_interactions):
                a, b = self.rng.choice(n_drugs, 2, replace=False)
                self.ddi_matrix[a, b] = 1.0
                self.ddi_matrix[b, a] = 1.0
        else:
            self.ddi_matrix = None

    def _generate_patient(self):
        spec = self.spec
        n_visits = self.rng.randint(2, spec.max_visits + 1)

        # Assign 1-2 archetypes to this patient
        n_arch = self.rng.choice([1, 2], p=[0.7, 0.3])
        archs = [self.archetypes[i] for i in
                 self.rng.choice(len(self.archetypes), n_arch, replace=False)]

        all_conditions = []
        all_procedures = []
        all_drugs = []

        for v in range(n_visits):
            arch = archs[v % len(archs)]

            # Sample a subset of the archetype's codes
            n_c = min(
                self.rng.randint(2, spec.max_codes_per_visit + 1),
                len(arch["conditions"]),
            )
            conds = self.rng.choice(
                arch["conditions"], n_c, replace=False,
            ).tolist()

            n_p = min(
                self.rng.randint(1, spec.max_codes_per_visit + 1),
                len(arch["procedures"]),
            )
            procs = self.rng.choice(
                arch["procedures"], n_p, replace=False,
            ).tolist()

            all_conditions.append(conds)
            all_procedures.append(procs)

            if arch["drugs"]:
                n_d = min(
                    self.rng.randint(2, len(arch["drugs"]) + 1),
                    len(arch["drugs"]),
                )
                drugs = self.rng.choice(
                    arch["drugs"], n_d, replace=False,
                ).tolist()
                # Add noise: 20% chance of a random extra drug
                if self.rng.random() < 0.2:
                    extra = self.rng.randint(0, spec.label_dim)
                    drugs.append(extra)
                all_drugs.append(drugs)

        # Create samples: one per visit (starting from visit 2)
        for t in range(1, n_visits):
            sample: Dict[str, Any] = {}

            # Cumulative history up to and including visit t
            sample["conditions"] = all_conditions[: t + 1]
            sample["procedures"] = all_procedures[: t + 1]

            if "drugs_hist" in spec.feature_keys and all_drugs:
                sample["drugs_hist"] = all_drugs[:t]  # exclude current
            if "drugs" in spec.feature_keys and all_drugs:
                sample["drugs"] = all_drugs[:t]

            # Labels
            if spec.task_type == "multilabel" and all_drugs:
                target_drugs = all_drugs[t] if t < len(all_drugs) else all_drugs[-1]
                multihot = np.zeros(spec.label_dim, dtype=np.float32)
                for d in target_drugs:
                    if d < spec.label_dim:
                        multihot[d] = 1.0
                sample[spec.label_key] = multihot
            elif spec.task_type == "binary":
                tendency = archs[0]["label_tendency"]
                # Add noise
                if self.rng.random() < 0.15:
                    tendency = 1 - tendency
                sample[spec.label_key] = tendency
            elif spec.task_type == "multiclass":
                tendency = archs[0]["label_tendency"]
                if self.rng.random() < 0.2:
                    tendency = self.rng.randint(0, spec.label_dim)
                sample[spec.label_key] = tendency

            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class InMemoryEHRDataset(Dataset):
    """Simple in-memory dataset used after normalizing PyHealth samples."""

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def _pad_nested_sequence(
    batch_seqs: List[List[List[int]]],
    vocab_size: int,
    max_visits: int = 0,
    max_codes: int = 0,
) -> torch.LongTensor:
    """Pad nested sequences to [batch, max_visits, max_codes]."""
    if max_visits == 0:
        max_visits = max(len(seq) for seq in batch_seqs)
    if max_codes == 0:
        max_codes = max(
            max((len(codes) for codes in seq), default=1)
            for seq in batch_seqs
        )
    # Clamp codes to valid range
    result = torch.zeros(len(batch_seqs), max_visits, max_codes, dtype=torch.long)
    for i, seq in enumerate(batch_seqs):
        for j, codes in enumerate(seq):
            if j >= max_visits:
                break
            for k, c in enumerate(codes):
                if k >= max_codes:
                    break
                result[i, j, k] = min(c, vocab_size - 1)
    return result


def _build_visit_mask(
    batch_seqs: List[List[List[int]]],
    max_visits: int,
) -> torch.BoolTensor:
    """Build boolean mask [batch, max_visits]."""
    mask = torch.zeros(len(batch_seqs), max_visits, dtype=torch.bool)
    for i, seq in enumerate(batch_seqs):
        mask[i, : min(len(seq), max_visits)] = True
    return mask


def collate_fn_factory(task_spec: TaskSpec):
    """Create a collate function for the given task."""

    def collate(batch: List[dict]) -> dict:
        result = {}

        # Determine max sizes
        max_visits = max(
            len(sample[task_spec.feature_keys[0]]) for sample in batch
        )
        max_codes = 0
        for key in task_spec.feature_keys:
            for sample in batch:
                if key in sample:
                    for codes in sample[key]:
                        max_codes = max(max_codes, len(codes))
        max_codes = max(max_codes, 1)

        # Pad feature sequences
        for key in task_spec.feature_keys:
            seqs = [sample.get(key, [[]]) for sample in batch]
            vocab = task_spec.feature_dims.get(key, 500)
            result[key] = _pad_nested_sequence(
                seqs, vocab, max_visits, max_codes,
            )

        # Visit mask
        seqs_for_mask = [sample[task_spec.feature_keys[0]] for sample in batch]
        result["mask"] = _build_visit_mask(seqs_for_mask, max_visits)

        # Labels
        if task_spec.task_type == "multilabel":
            labels = [sample[task_spec.label_key] for sample in batch]
            result[task_spec.label_key] = torch.tensor(
                np.stack(labels), dtype=torch.float,
            )
        elif task_spec.task_type in ("binary", "multiclass"):
            labels = [sample[task_spec.label_key] for sample in batch]
            result[task_spec.label_key] = torch.tensor(labels, dtype=torch.long)

        return result

    return collate


def _ensure_list(value: Any) -> List[Any]:
    """Normalize arbitrary sample values to a Python list."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _ensure_nested_sequence(value: Any) -> List[List[Any]]:
    """Normalize flat or nested visit representations to visit-major format."""
    visits = _ensure_list(value)
    if not visits:
        return [[]]
    if isinstance(visits[0], (list, tuple, np.ndarray)) or torch.is_tensor(visits[0]):
        normalized = []
        for visit in visits:
            visit_list = _ensure_list(visit)
            normalized.append(visit_list)
        return normalized or [[]]
    return [visits]


def _iter_codes(nested: List[List[Any]]):
    for visit in nested:
        for code in visit:
            yield code


def _coerce_scalar(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _resolve_label_key(spec: TaskSpec, sample: Dict[str, Any]) -> str:
    """Find the label key used by a task sample, allowing PyHealth aliases."""
    if spec.label_key in sample:
        return spec.label_key

    task_aliases = {
        "length_of_stay": "los",
    }
    alias = task_aliases.get(spec.name)
    if alias and alias in sample:
        return alias

    protected = set(spec.feature_keys) | {"patient_id", "visit_id", "record_id", "mask"}
    candidates = [key for key in sample.keys() if key not in protected]
    if len(candidates) == 1:
        return candidates[0]

    raise KeyError(
        f"Could not resolve label key for task '{spec.name}'. "
        f"Expected '{spec.label_key}', sample keys: {sorted(sample.keys())}"
    )


def _build_token_mapping(values: List[Any]) -> Dict[Any, int]:
    tokens = sorted({value for value in values if value is not None}, key=lambda x: str(x))
    return {token: idx + 1 for idx, token in enumerate(tokens)}


def _normalize_pyhealth_splits(
    spec: TaskSpec,
    train_split: Dataset,
    val_split: Dataset,
    test_split: Dataset,
) -> Tuple[TaskSpec, DataLoader, DataLoader, DataLoader]:
    """Convert PyHealth samples into the same tensorized format as synthetic data."""
    train_samples = [train_split[i] for i in range(len(train_split))]
    val_samples = [val_split[i] for i in range(len(val_split))]
    test_samples = [test_split[i] for i in range(len(test_split))]
    all_samples = train_samples + val_samples + test_samples

    if not all_samples:
        raise ValueError("PyHealth task produced no samples.")

    feature_values: Dict[str, List[Any]] = {key: [] for key in spec.feature_keys}
    label_values: List[Any] = []

    for sample in all_samples:
        for key in spec.feature_keys:
            if key not in sample:
                continue
            feature_values[key].extend(_iter_codes(_ensure_nested_sequence(sample[key])))
        label_key = _resolve_label_key(spec, sample)
        label_raw = sample[label_key]
        if spec.task_type == "multilabel":
            label_values.extend(_ensure_list(label_raw))
        else:
            label_values.append(_coerce_scalar(label_raw))

    feature_maps = {
        key: _build_token_mapping(values) for key, values in feature_values.items()
    }

    label_map: Dict[Any, int] = {}
    if spec.task_type == "multilabel":
        label_tokens = sorted({value for value in label_values if value is not None}, key=lambda x: str(x))
        label_map = {token: idx for idx, token in enumerate(label_tokens)}
        label_dim = max(len(label_map), 1)
    else:
        unique_labels = sorted({value for value in label_values}, key=lambda x: str(x))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        label_dim = max(len(label_map), spec.label_dim)

    resolved_spec = replace(
        spec,
        feature_dims={key: max(len(mapping) + 1, 2) for key, mapping in feature_maps.items()},
        label_dim=label_dim,
    )

    def encode_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for meta_key in ("patient_id", "visit_id", "record_id"):
            if meta_key in sample:
                result[meta_key] = sample[meta_key]

        for key in resolved_spec.feature_keys:
            visits = _ensure_nested_sequence(sample.get(key, []))
            encoded_visits = []
            for visit in visits:
                encoded_visit = [
                    feature_maps[key][code]
                    for code in visit
                    if code in feature_maps[key]
                ]
                encoded_visits.append(encoded_visit)
            result[key] = encoded_visits or [[]]

        label_key = _resolve_label_key(resolved_spec, sample)
        raw_label = sample[label_key]
        if resolved_spec.task_type == "multilabel":
            multihot = np.zeros(resolved_spec.label_dim, dtype=np.float32)
            for code in _ensure_list(raw_label):
                if code in label_map:
                    multihot[label_map[code]] = 1.0
            result[resolved_spec.label_key] = multihot
        else:
            label_value = _coerce_scalar(raw_label)
            if label_value not in label_map:
                raise KeyError(
                    f"Unexpected label value {label_value!r} for task '{resolved_spec.name}'."
                )
            result[resolved_spec.label_key] = label_map[label_value]
        return result

    train_ds = InMemoryEHRDataset([encode_sample(sample) for sample in train_samples])
    val_ds = InMemoryEHRDataset([encode_sample(sample) for sample in val_samples])
    test_ds = InMemoryEHRDataset([encode_sample(sample) for sample in test_samples])

    collate = collate_fn_factory(resolved_spec)
    common = dict(collate_fn=collate, num_workers=0, pin_memory=torch.cuda.is_available())

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, **common)

    return resolved_spec, train_loader, val_loader, test_loader


# ============================================================
# Data Loading API
# ============================================================

def load_task_data(
    task_name: str,
    batch_size: int = 32,
    use_pyhealth: bool = False,
    data_root: Optional[str] = None,
    n_synthetic_patients: int = 2000,
    seed: int = SEED,
    return_spec: bool = False,
) -> Tuple[Any, ...]:
    """Load data for a clinical task.

    Args:
        task_name: Registered task name.
        batch_size: Batch size for all loaders.
        use_pyhealth: If True, attempt to load real data via PyHealth.
            Requires MIMIC data at data_root.
        data_root: Root directory for MIMIC data (PyHealth mode only).
        n_synthetic_patients: Number of synthetic patients to generate.
        seed: Random seed.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    spec = TaskRegistry.get(task_name)

    if task_name.startswith("support2_"):
        # SUPPORT2 tasks always use real public data
        resolved_spec, train_loader, val_loader, test_loader = _load_support2_data(
            spec, batch_size, seed,
        )
    elif task_name.startswith("mimic4_"):
        # MIMIC-IV tasks use real data from local CSV files
        resolved_spec, train_loader, val_loader, test_loader = _load_mimic4_data(
            spec, batch_size, seed,
            data_root=data_root,
            dev=(n_synthetic_patients <= 1000),  # small n_patients → dev mode
        )
    elif use_pyhealth and data_root:
        resolved_spec, train_loader, val_loader, test_loader = _load_pyhealth_data(
            spec, batch_size, data_root, seed=seed,
        )
    else:
        resolved_spec = spec
        train_loader, val_loader, test_loader = _load_synthetic_data(
            spec, batch_size, n_synthetic_patients, seed,
        )

    if return_spec:
        return resolved_spec, train_loader, val_loader, test_loader
    return train_loader, val_loader, test_loader


def _load_synthetic_data(
    spec: TaskSpec, batch_size: int, n_patients: int, seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Generate synthetic data and split 80/10/10."""
    dataset = SyntheticEHRDataset(spec, n_patients=n_patients, seed=seed)

    n = len(dataset)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test], generator=gen,
    )

    collate = collate_fn_factory(spec)
    common = dict(
        collate_fn=collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)

    return train_loader, val_loader, test_loader


def _load_pyhealth_data(
    spec: TaskSpec, batch_size: int, data_root: str, seed: int = SEED,
) -> Tuple[TaskSpec, DataLoader, DataLoader, DataLoader]:
    """Load real data through PyHealth.

    Requires PyHealth and MIMIC data. Falls back to synthetic on failure.
    """
    try:
        sys.path.insert(0, str(PYHEALTH_ROOT))
        from pyhealth.datasets import MIMIC3Dataset
        from pyhealth.datasets.splitter import split_by_patient

        if spec.pyhealth_dataset == "mimic3":
            base_ds = MIMIC3Dataset(root=data_root, tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"])
        else:
            raise ValueError(f"Unsupported dataset: {spec.pyhealth_dataset}")

        # Import and apply task
        task_module = __import__(
            f"pyhealth.tasks.{spec.pyhealth_task}",
            fromlist=[spec.pyhealth_task],
        )
        task_cls_name = "".join(
            w.capitalize() for w in spec.pyhealth_task.split("_")
        ) + "MIMIC3"
        task_fn = getattr(task_module, task_cls_name)()

        sample_ds = base_ds.set_task(task_fn)
        train_ds, val_ds, test_ds = split_by_patient(sample_ds, [0.8, 0.1, 0.1], seed=seed)

        resolved_spec, train_loader, val_loader, test_loader = _normalize_pyhealth_splits(
            spec, train_ds, val_ds, test_ds,
        )
        return (
            resolved_spec,
            DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True,
                       collate_fn=train_loader.collate_fn, num_workers=0,
                       pin_memory=torch.cuda.is_available()),
            DataLoader(val_loader.dataset, batch_size=batch_size, shuffle=False,
                       collate_fn=val_loader.collate_fn, num_workers=0,
                       pin_memory=torch.cuda.is_available()),
            DataLoader(test_loader.dataset, batch_size=batch_size, shuffle=False,
                       collate_fn=test_loader.collate_fn, num_workers=0,
                       pin_memory=torch.cuda.is_available()),
        )

    except Exception as e:
        print(f"[prepare] PyHealth loading failed: {e}")
        print("[prepare] Falling back to synthetic data.")
        return (
            spec,
            *_load_synthetic_data(spec, batch_size, 2000, seed),
        )


# ============================================================
# SUPPORT2 — Real public clinical data (9105 patients)
# ============================================================

# Feature groups for SUPPORT2 discretization.
# Each numeric feature is binned into N_BINS codes; categorical features
# get one code per unique value.  The result fits our [batch, visits, codes]
# tensor format by treating each feature group as one "visit".

_SUPPORT2_NUMERIC_FEATURES = {
    "vitals": ["meanbp", "hrt", "resp", "temp", "pafi"],
    "labs":   ["wblc", "alb", "bili", "crea", "sod", "ph", "glucose", "bun"],
    "scores": ["sps", "aps", "scoma", "num.co"],
    "adl":    ["adlp", "adls", "adlsc"],
}

_SUPPORT2_CATEGORICAL_FEATURES = {
    "demographics": ["sex", "race", "income", "dzgroup", "dzclass", "ca", "dnr"],
}

_SUPPORT2_N_BINS = 10  # quantile bins per numeric feature

# Per-task feature exclusions to prevent information leakage.
# Each key is a task name; values are features to remove from input.
_SUPPORT2_EXCLUDE = {
    # dzclass label is derived from dzgroup, so both leak the answer
    "support2_dzclass": {"dzgroup", "dzclass"},
    # surv2m is computed from sps; prg2m/prg6m/surv6m are also prognosis scores
    "support2_survival": {"sps", "surv2m", "surv6m", "prg2m", "prg6m"},
    # mortality: no leakage, but exclude hospdead (the label) and derived fields
    "support2_mortality": {"hospdead", "slos"},  # slos = survival LOS, only known post-discharge
}


class Support2Dataset(Dataset):
    """SUPPORT2 public dataset adapted for our framework.

    Downloads 9105 seriously-ill patient records from HuggingFace and
    discretizes features into medical-code-like integers so they work
    with our existing CodeEmbedding → GRU → RL pipeline.

    Feature groups become "visits", codes within each group become "codes
    per visit".  This is a slight abstraction but keeps the entire
    model/eval pipeline identical to the synthetic EHR path.
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        seed: int = SEED,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.spec = task_spec
        self.rng = np.random.RandomState(seed)
        self.ddi_matrix = None  # no DDI for SUPPORT2 tasks

        raw = self._download(cache_dir)
        self.vocab_size, self.samples = self._build_samples(raw, task_spec)

    # ---- download ----

    @staticmethod
    def _download(cache_dir: Optional[str] = None) -> List[dict]:
        """Download SUPPORT2 from HuggingFace (cached after first call)."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "pip install datasets  # required for SUPPORT2"
            )
        kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        ds = load_dataset("jarrydmartinx/support2", split="train", **kwargs)
        return [dict(row) for row in ds]

    # ---- discretize ----

    def _build_samples(
        self, raw: List[dict], spec: TaskSpec,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """Discretize raw SUPPORT2 rows into our sample format."""

        # 0. Determine which features to exclude for this task
        exclude = _SUPPORT2_EXCLUDE.get(spec.name, set())

        # Build filtered feature groups
        num_features = {
            group: [f for f in feats if f not in exclude]
            for group, feats in _SUPPORT2_NUMERIC_FEATURES.items()
        }
        cat_features = {
            group: [f for f in feats if f not in exclude]
            for group, feats in _SUPPORT2_CATEGORICAL_FEATURES.items()
        }

        # 1. Compute quantile bin edges for each numeric feature
        bin_edges: Dict[str, np.ndarray] = {}
        for group_features in num_features.values():
            for feat in group_features:
                vals = [r[feat] for r in raw if r.get(feat) is not None]
                if vals:
                    bin_edges[feat] = np.nanquantile(
                        vals,
                        np.linspace(0, 1, _SUPPORT2_N_BINS + 1)[1:-1],
                    )

        # 2. Build categorical vocabularies
        cat_vocabs: Dict[str, Dict[Any, int]] = {}
        for group_features in cat_features.values():
            for feat in group_features:
                unique = sorted({str(r.get(feat, "missing")) for r in raw})
                cat_vocabs[feat] = {v: i + 1 for i, v in enumerate(unique)}

        # 3. Compute vocab size: sum of all bins + all cat values + 1 (padding)
        offset = 1  # 0 = padding
        feat_offset: Dict[str, int] = {}
        for group_features in num_features.values():
            for feat in group_features:
                feat_offset[feat] = offset
                offset += _SUPPORT2_N_BINS
        for group_features in cat_features.values():
            for feat in group_features:
                feat_offset[feat] = offset
                offset += len(cat_vocabs.get(feat, {}))
        vocab_size = offset

        # 4. Convert each patient row into a sample
        samples = []
        for row in raw:
            sample: Dict[str, Any] = {}

            # Build feature groups as "visits"
            all_visits: List[List[int]] = []

            # Numeric feature groups → one visit each
            for group_name, group_features in num_features.items():
                codes = []
                for feat in group_features:
                    val = row.get(feat)
                    if val is not None and feat in bin_edges:
                        bin_idx = int(np.searchsorted(bin_edges[feat], val))
                        codes.append(feat_offset[feat] + bin_idx)
                    # skip missing
                if not codes:
                    codes = [0]  # padding
                all_visits.append(codes)

            # Categorical feature group → one visit
            for group_name, group_features in cat_features.items():
                codes = []
                for feat in group_features:
                    val = str(row.get(feat, "missing"))
                    vocab = cat_vocabs.get(feat, {})
                    if val in vocab:
                        codes.append(feat_offset[feat] + vocab[val])
                if not codes:
                    codes = [0]
                all_visits.append(codes)

            # Map to our standard feature keys:
            # "conditions" gets vitals+labs+scores (clinical observations)
            # "procedures" gets demographics+adl (patient context)
            clinical_visits = all_visits[:3]  # vitals, labs, scores
            context_visits = all_visits[3:]   # adl, demographics

            sample["conditions"] = clinical_visits if clinical_visits else [[0]]
            sample["procedures"] = context_visits if context_visits else [[0]]

            # Labels
            if spec.label_key == "hospdead":
                sample["hospdead"] = int(row.get("hospdead", 0))
            elif spec.label_key == "dzclass":
                dzclass_map = {
                    "ARF/MOSF": 0,
                    "COPD/CHF/Cirrhosis": 1,
                    "Cancer": 2,
                    "Coma": 3,
                }
                sample["dzclass"] = dzclass_map.get(row.get("dzclass", ""), 0)
            elif spec.label_key == "survival_2m":
                surv = row.get("surv2m")
                # Binary: 1 if survived (surv2m >= 0.5), 0 if not
                sample["survival_2m"] = 1 if (surv is not None and surv >= 0.5) else 0

            samples.append(sample)

        return vocab_size, samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _load_support2_data(
    spec: TaskSpec, batch_size: int, seed: int,
) -> Tuple[TaskSpec, DataLoader, DataLoader, DataLoader]:
    """Load SUPPORT2 real data, split 80/10/10."""
    dataset = Support2Dataset(spec, seed=seed)

    # Update spec with actual vocab size from data
    resolved_spec = replace(
        spec,
        feature_dims={
            "conditions": dataset.vocab_size,
            "procedures": dataset.vocab_size,
        },
    )

    n = len(dataset)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test], generator=gen,
    )

    collate = collate_fn_factory(resolved_spec)
    common = dict(
        collate_fn=collate, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)

    return resolved_spec, train_loader, val_loader, test_loader


# ---- Register SUPPORT2 tasks ----

TaskRegistry.register(TaskSpec(
    name="support2_mortality",
    task_type="binary",
    description=(
        "Predict in-hospital mortality for seriously ill patients. "
        "SUPPORT2 public dataset (9105 patients, 5 US medical centers). "
        "RL angle: adaptive feature selection under missing data."
    ),
    feature_keys=["conditions", "procedures"],
    label_key="hospdead",
    feature_dims={"conditions": 200, "procedures": 200},  # updated at load time
    label_dim=2,
    primary_metric="auroc",
    metric_direction="max",
    metrics=["auroc", "auprc", "f1"],
    reward_components={"auroc": 1.0},
    pyhealth_dataset="support2",
))

TaskRegistry.register(TaskSpec(
    name="support2_dzclass",
    task_type="multiclass",
    description=(
        "Classify disease category (ARF/MOSF, COPD/CHF/Cirrhosis, Cancer, Coma). "
        "SUPPORT2 public dataset. RL angle: learning discriminative clinical patterns."
    ),
    feature_keys=["conditions", "procedures"],
    label_key="dzclass",
    feature_dims={"conditions": 200, "procedures": 200},
    label_dim=4,
    primary_metric="f1_macro",
    metric_direction="max",
    metrics=["f1_macro", "accuracy", "auroc_macro"],
    reward_components={"f1_macro": 1.0},
    pyhealth_dataset="support2",
))

TaskRegistry.register(TaskSpec(
    name="support2_survival",
    task_type="binary",
    description=(
        "Predict 2-month survival (surv2m >= 0.5) for seriously ill patients. "
        "SUPPORT2 public dataset. RL angle: prognosis under uncertainty."
    ),
    feature_keys=["conditions", "procedures"],
    label_key="survival_2m",
    feature_dims={"conditions": 200, "procedures": 200},
    label_dim=2,
    primary_metric="auroc",
    metric_direction="max",
    metrics=["auroc", "auprc", "f1"],
    reward_components={"auroc": 1.0},
    pyhealth_dataset="support2",
))


# ============================================================
# MIMIC-IV — Real longitudinal EHR data (~364K patients)
# ============================================================

# Default MIMIC-IV data root (overridable via --data-root)
_MIMIC4_DEFAULT_ROOT = "/home/sw2572/project_pi_yz875/sw2572/data/physionet.org/files/mimiciv/3.1"

# Vocabulary caps — keep top-N codes to manage embedding dimensions
_MIMIC4_MAX_DIAG_CODES = 500    # top 500 ICD diagnosis codes
_MIMIC4_MAX_PROC_CODES = 200    # top 200 ICD procedure codes
_MIMIC4_MAX_DRUG_CODES = 300    # top 300 drug names

# LOS buckets (days): <3, 3-7, 7-14, >14
_MIMIC4_LOS_BINS = [3.0, 7.0, 14.0]


class MIMIC4Dataset(Dataset):
    """MIMIC-IV dataset loader — reads CSV.gz directly without PyHealth.

    Builds patient → visit (admission) → codes structure from:
      - admissions.csv.gz: visit metadata + mortality/LOS labels
      - diagnoses_icd.csv.gz: ICD diagnosis codes per visit
      - procedures_icd.csv.gz: ICD procedure codes per visit
      - prescriptions.csv.gz: drug names per visit (for drug_rec task)

    Each admission = one visit.  Patients with >= 2 admissions get
    sequential visit history suitable for our GRU pipeline.
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        data_root: Optional[str] = None,
        seed: int = SEED,
        dev: bool = False,
        max_patients: int = 0,
    ):
        super().__init__()
        import pandas as pd

        self.spec = task_spec
        self.rng = np.random.RandomState(seed)
        self.ddi_matrix = None
        root = data_root or _MIMIC4_DEFAULT_ROOT

        # ---- 1. Load admissions (core table) ----
        # In dev mode, limit initial rows to save memory on login nodes
        read_kw: Dict[str, Any] = {}
        if dev or max_patients > 0:
            read_kw["nrows"] = max(max_patients * 10, 10000) if max_patients > 0 else 10000

        adm = pd.read_csv(
            os.path.join(root, "hosp", "admissions.csv.gz"),
            usecols=["subject_id", "hadm_id", "admittime", "dischtime",
                      "hospital_expire_flag"],
            parse_dates=["admittime", "dischtime"],
            **read_kw,
        )
        adm = adm.dropna(subset=["subject_id", "hadm_id"])
        adm["subject_id"] = adm["subject_id"].astype(int)
        adm["hadm_id"] = adm["hadm_id"].astype(int)
        adm = adm.sort_values(["subject_id", "admittime"])

        # Dev mode: keep only first N patients for quick testing
        if dev or max_patients > 0:
            n_keep = max_patients if max_patients > 0 else 1000
            keep_ids = adm["subject_id"].unique()[:n_keep]
            adm = adm[adm["subject_id"].isin(keep_ids)]

        hadm_ids = set(adm["hadm_id"].values)

        # ---- 2. Load diagnoses ----
        diag_kw: Dict[str, Any] = {}
        if dev or max_patients > 0:
            diag_kw["nrows"] = 200000  # ~enough for dev subset
        diag = pd.read_csv(
            os.path.join(root, "hosp", "diagnoses_icd.csv.gz"),
            usecols=["hadm_id", "icd_code"],
            **diag_kw,
        )
        diag = diag[diag["hadm_id"].isin(hadm_ids)]
        diag["icd_code"] = diag["icd_code"].astype(str)

        # Build top-N diagnosis vocabulary
        diag_counts = diag["icd_code"].value_counts()
        top_diag = set(diag_counts.head(_MIMIC4_MAX_DIAG_CODES).index)
        diag = diag[diag["icd_code"].isin(top_diag)]
        diag_vocab = {code: idx + 1 for idx, code in enumerate(sorted(top_diag))}

        # ---- 3. Load procedures ----
        proc_kw: Dict[str, Any] = {}
        if dev or max_patients > 0:
            proc_kw["nrows"] = 50000
        proc = pd.read_csv(
            os.path.join(root, "hosp", "procedures_icd.csv.gz"),
            usecols=["hadm_id", "icd_code"],
            **proc_kw,
        )
        proc = proc[proc["hadm_id"].isin(hadm_ids)]
        proc["icd_code"] = proc["icd_code"].astype(str)

        proc_counts = proc["icd_code"].value_counts()
        top_proc = set(proc_counts.head(_MIMIC4_MAX_PROC_CODES).index)
        proc = proc[proc["icd_code"].isin(top_proc)]
        proc_vocab = {code: idx + 1 for idx, code in enumerate(sorted(top_proc))}

        # ---- 4. Load prescriptions (only for drug_rec task) ----
        drug_vocab: Dict[str, int] = {}
        drug_per_hadm: Dict[int, List[int]] = {}
        need_drugs = task_spec.name == "mimic4_drugrec"

        if need_drugs:
            # Read only needed columns; prescriptions is large (20M rows)
            rx_kw: Dict[str, Any] = {}
            if dev or max_patients > 0:
                rx_kw["nrows"] = 500000
            rx = pd.read_csv(
                os.path.join(root, "hosp", "prescriptions.csv.gz"),
                usecols=["hadm_id", "drug"],
                **rx_kw,
            )
            rx = rx[rx["hadm_id"].isin(hadm_ids)]
            rx["drug"] = rx["drug"].astype(str)

            drug_counts = rx["drug"].value_counts()
            top_drugs = set(drug_counts.head(_MIMIC4_MAX_DRUG_CODES).index)
            rx = rx[rx["drug"].isin(top_drugs)]
            drug_vocab = {d: idx for idx, d in enumerate(sorted(top_drugs))}

            for hadm_id, group in rx.groupby("hadm_id"):
                codes = list({drug_vocab[d] for d in group["drug"] if d in drug_vocab})
                if codes:
                    drug_per_hadm[int(hadm_id)] = codes

        # ---- 5. Group codes per visit ----
        diag_per_hadm: Dict[int, List[int]] = {}
        for hadm_id, group in diag.groupby("hadm_id"):
            codes = list({diag_vocab[c] for c in group["icd_code"] if c in diag_vocab})
            if codes:
                diag_per_hadm[int(hadm_id)] = codes

        proc_per_hadm: Dict[int, List[int]] = {}
        for hadm_id, group in proc.groupby("hadm_id"):
            codes = list({proc_vocab[c] for c in group["icd_code"] if c in proc_vocab})
            if codes:
                proc_per_hadm[int(hadm_id)] = codes

        # Free dataframes
        del diag, proc
        if need_drugs:
            del rx

        # ---- 6. Build patient visit sequences ----
        self.diag_vocab_size = len(diag_vocab) + 1  # +1 for padding
        self.proc_vocab_size = len(proc_vocab) + 1
        self.drug_vocab_size = len(drug_vocab) if drug_vocab else 0

        self.samples: List[Dict[str, Any]] = []
        patients = adm.groupby("subject_id")

        for subject_id, visits in patients:
            visit_list = visits.sort_values("admittime").to_dict("records")
            if len(visit_list) < 2:
                continue  # need >= 2 visits for sequential prediction

            # Build per-visit code lists
            visit_diags = []
            visit_procs = []
            visit_drugs = []

            for v in visit_list:
                hid = int(v["hadm_id"])
                visit_diags.append(diag_per_hadm.get(hid, [0]))
                visit_procs.append(proc_per_hadm.get(hid, [0]))
                if need_drugs:
                    visit_drugs.append(drug_per_hadm.get(hid, []))

            # Create samples: one per visit (from visit index 1 onward)
            for t in range(1, len(visit_list)):
                sample: Dict[str, Any] = {}
                v = visit_list[t]

                # Feature history up to and including visit t
                sample["conditions"] = visit_diags[: t + 1]
                sample["procedures"] = visit_procs[: t + 1]

                if need_drugs:
                    sample["drugs_hist"] = visit_drugs[:t]

                # Labels
                if task_spec.label_key == "mortality":
                    sample["mortality"] = int(v.get("hospital_expire_flag", 0))
                elif task_spec.label_key == "readmission":
                    # 30-day readmission: was there another admission within 30 days?
                    if t + 1 < len(visit_list):
                        dischtime = v.get("dischtime")
                        next_admittime = visit_list[t + 1].get("admittime")
                        if pd.notna(dischtime) and pd.notna(next_admittime):
                            gap = (next_admittime - dischtime).days
                            sample["readmission"] = 1 if gap <= 30 else 0
                        else:
                            sample["readmission"] = 0
                    else:
                        sample["readmission"] = 0
                elif task_spec.label_key == "los_bucket":
                    # LOS in days
                    dischtime = v.get("dischtime")
                    admittime = v.get("admittime")
                    if pd.notna(dischtime) and pd.notna(admittime):
                        los_days = (dischtime - admittime).total_seconds() / 86400
                    else:
                        los_days = 0.0
                    # Bucket: 0=<3d, 1=3-7d, 2=7-14d, 3=>14d
                    bucket = int(np.searchsorted(_MIMIC4_LOS_BINS, los_days))
                    sample["los_bucket"] = bucket
                elif task_spec.label_key == "drugs":
                    # Multilabel: drugs prescribed in current visit
                    hid = int(v["hadm_id"])
                    target = drug_per_hadm.get(hid, [])
                    multihot = np.zeros(len(drug_vocab), dtype=np.float32)
                    for d in target:
                        multihot[d] = 1.0
                    sample["drugs"] = multihot

                self.samples.append(sample)

        print(f"[MIMIC-IV] Loaded {len(self.samples)} samples from "
              f"{len(patients)} patients ({task_spec.name})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _load_mimic4_data(
    spec: TaskSpec, batch_size: int, seed: int,
    data_root: Optional[str] = None,
    dev: bool = False,
    max_patients: int = 0,
) -> Tuple[TaskSpec, DataLoader, DataLoader, DataLoader]:
    """Load MIMIC-IV real data, split 80/10/10 by patient."""
    dataset = MIMIC4Dataset(
        spec, data_root=data_root, seed=seed,
        dev=dev, max_patients=max_patients,
    )

    # Update spec with actual vocab sizes
    feature_dims = {
        "conditions": dataset.diag_vocab_size,
        "procedures": dataset.proc_vocab_size,
    }
    label_dim = spec.label_dim
    if spec.task_type == "multilabel" and dataset.drug_vocab_size > 0:
        label_dim = dataset.drug_vocab_size

    resolved_spec = replace(spec, feature_dims=feature_dims, label_dim=label_dim)

    n = len(dataset)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test], generator=gen,
    )

    collate = collate_fn_factory(resolved_spec)
    common = dict(
        collate_fn=collate, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)

    return resolved_spec, train_loader, val_loader, test_loader


# ---- Register MIMIC-IV tasks ----

TaskRegistry.register(TaskSpec(
    name="mimic4_mortality",
    task_type="binary",
    description=(
        "Predict in-hospital mortality from MIMIC-IV longitudinal EHR. "
        "Uses ICD diagnoses + procedures across multiple admissions. "
        "~364K patients, real ICU data."
    ),
    feature_keys=["conditions", "procedures"],
    label_key="mortality",
    feature_dims={"conditions": _MIMIC4_MAX_DIAG_CODES + 1, "procedures": _MIMIC4_MAX_PROC_CODES + 1},
    label_dim=2,
    primary_metric="auroc",
    metric_direction="max",
    metrics=["auroc", "auprc", "f1"],
    reward_components={"auroc": 1.0},
    pyhealth_dataset="mimic4",
    pyhealth_task="mortality_prediction",
))

TaskRegistry.register(TaskSpec(
    name="mimic4_readmission",
    task_type="binary",
    description=(
        "Predict 30-day hospital readmission from MIMIC-IV longitudinal EHR. "
        "Uses ICD diagnoses + procedures across multiple admissions."
    ),
    feature_keys=["conditions", "procedures"],
    label_key="readmission",
    feature_dims={"conditions": _MIMIC4_MAX_DIAG_CODES + 1, "procedures": _MIMIC4_MAX_PROC_CODES + 1},
    label_dim=2,
    primary_metric="auroc",
    metric_direction="max",
    metrics=["auroc", "auprc", "f1"],
    reward_components={"auroc": 1.0},
    pyhealth_dataset="mimic4",
    pyhealth_task="readmission_prediction",
))

TaskRegistry.register(TaskSpec(
    name="mimic4_los",
    task_type="multiclass",
    description=(
        "Predict length-of-stay bucket (<3d, 3-7d, 7-14d, >14d) from MIMIC-IV. "
        "Uses ICD diagnoses + procedures across multiple admissions."
    ),
    feature_keys=["conditions", "procedures"],
    label_key="los_bucket",
    feature_dims={"conditions": _MIMIC4_MAX_DIAG_CODES + 1, "procedures": _MIMIC4_MAX_PROC_CODES + 1},
    label_dim=4,
    primary_metric="f1_macro",
    metric_direction="max",
    metrics=["f1_macro", "accuracy", "auroc_macro"],
    reward_components={"f1_macro": 1.0},
    pyhealth_dataset="mimic4",
    pyhealth_task="length_of_stay_prediction",
))

TaskRegistry.register(TaskSpec(
    name="mimic4_drugrec",
    task_type="multilabel",
    description=(
        "Predict drug prescriptions from MIMIC-IV longitudinal EHR. "
        "Uses ICD diagnoses + procedures as features, predicts top-300 drugs."
    ),
    feature_keys=["conditions", "procedures", "drugs_hist"],
    label_key="drugs",
    feature_dims={"conditions": _MIMIC4_MAX_DIAG_CODES + 1, "procedures": _MIMIC4_MAX_PROC_CODES + 1,
                  "drugs_hist": _MIMIC4_MAX_DRUG_CODES},
    label_dim=_MIMIC4_MAX_DRUG_CODES,
    primary_metric="jaccard_samples",
    metric_direction="max",
    metrics=["jaccard_samples", "f1_samples", "pr_auc_samples"],
    reward_components={"jaccard_samples": 1.0, "ddi_rate": -0.5},
    pyhealth_dataset="mimic4",
    pyhealth_task="drug_recommendation",
))


def get_ddi_matrix(task_name: str, seed: int = SEED) -> Optional[np.ndarray]:
    """Get DDI adjacency matrix for drug recommendation tasks."""
    spec = TaskRegistry.get(task_name)
    if spec.task_type != "multilabel":
        return None
    ds = SyntheticEHRDataset(spec, n_patients=10, seed=seed)
    return ds.ddi_matrix


# ============================================================
# Evaluation Harness
# ============================================================

def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    task_spec: TaskSpec,
    device: str = "cuda",
) -> Dict[str, float]:
    """Standardized evaluation. DO NOT MODIFY.

    Runs the model on the given dataloader and computes all metrics
    specified in the task spec.

    Returns:
        Dictionary of metric_name -> score
    """
    model.eval()
    all_y_true = []
    all_y_prob = []
    all_y_pred = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            output = model(**batch)

            if "y_prob" in output:
                y_prob = output["y_prob"].cpu()
            elif "logit" in output:
                logit = output["logit"].cpu()
                if task_spec.task_type == "multilabel":
                    y_prob = torch.sigmoid(logit)
                elif task_spec.task_type == "binary":
                    y_prob = torch.softmax(logit, dim=-1)[:, 1]
                else:
                    y_prob = torch.softmax(logit, dim=-1)
            else:
                continue

            if "y_true" in output:
                y_true = output["y_true"].cpu()
            else:
                y_true = batch[task_spec.label_key].cpu()

            all_y_prob.append(y_prob)
            all_y_true.append(y_true)

    if not all_y_prob:
        return {m: 0.0 for m in task_spec.metrics}

    y_prob = torch.cat(all_y_prob, dim=0).numpy()
    y_true = torch.cat(all_y_true, dim=0).numpy()

    if task_spec.task_type == "multilabel":
        y_pred = (y_prob >= 0.5).astype(float)
    elif task_spec.task_type == "binary":
        y_pred = (y_prob >= 0.5).astype(float)
    elif task_spec.task_type == "multiclass":
        y_pred = y_prob.argmax(axis=-1)

    metrics = {}
    for metric_name in task_spec.metrics:
        metrics[metric_name] = _compute_metric(
            metric_name, y_true, y_prob, y_pred, task_spec,
        )

    return metrics


def _compute_metric(
    name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    spec: TaskSpec,
) -> float:
    """Compute a single metric. Handles edge cases gracefully."""
    from sklearn.metrics import (
        accuracy_score,
        auc,
        average_precision_score,
        f1_score,
        jaccard_score,
        precision_recall_curve,
        roc_auc_score,
    )

    try:
        if name == "jaccard_samples":
            return float(jaccard_score(y_true, y_pred, average="samples", zero_division=0))
        elif name == "f1_samples":
            return float(f1_score(y_true, y_pred, average="samples", zero_division=0))
        elif name == "f1":
            return float(f1_score(y_true, y_pred, average="binary", zero_division=0))
        elif name == "f1_macro":
            return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        elif name == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        elif name == "auroc":
            if len(np.unique(y_true)) < 2:
                return 0.5
            return float(roc_auc_score(y_true, y_prob))
        elif name == "auroc_macro":
            if y_prob.ndim == 1:
                return 0.5
            try:
                return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
            except ValueError:
                return 0.5
        elif name == "auprc":
            if len(np.unique(y_true)) < 2:
                return 0.5
            return float(average_precision_score(y_true, y_prob))
        elif name == "pr_auc_samples":
            # Per-label AUPRC, averaged
            n_labels = y_true.shape[1] if y_true.ndim > 1 else 1
            aucs = []
            for j in range(n_labels):
                yt = y_true[:, j] if y_true.ndim > 1 else y_true
                yp = y_prob[:, j] if y_prob.ndim > 1 else y_prob
                if len(np.unique(yt)) < 2:
                    continue
                aucs.append(float(average_precision_score(yt, yp)))
            return float(np.mean(aucs)) if aucs else 0.0
        else:
            return 0.0
    except Exception:
        return 0.0


def compute_reward(metrics: Dict[str, float], task_spec: TaskSpec) -> float:
    """Compute scalar reward from metrics using task reward components."""
    aliases = {
        "jaccard": "jaccard_samples",
        "ddi_penalty": "ddi_rate",
    }
    reward = 0.0
    for component, weight in task_spec.reward_components.items():
        metric_name = component if component in metrics else aliases.get(component, component)
        if metric_name in metrics:
            reward += weight * metrics[metric_name]
    return reward


def compute_ddi_rate(
    y_pred: np.ndarray,
    ddi_matrix: Optional[np.ndarray],
) -> float:
    """Compute DDI rate for predicted drug combinations."""
    if ddi_matrix is None or y_pred.ndim < 2:
        return 0.0
    total_pairs = 0
    ddi_pairs = 0
    for row in y_pred:
        drugs = np.where(row > 0.5)[0]
        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                total_pairs += 1
                if ddi_matrix[drugs[i], drugs[j]] > 0:
                    ddi_pairs += 1
    return ddi_pairs / max(total_pairs, 1)


# ============================================================
# Results Output (matches autoresearch format)
# ============================================================

def print_results(
    metrics: Dict[str, float],
    task_spec: TaskSpec,
    training_seconds: float,
    total_seconds: float,
    peak_vram_mb: float = 0.0,
    num_params: int = 0,
    **extra,
) -> None:
    """Print standardized result summary.

    Format is grep-friendly to match autoresearch conventions:
        grep "^primary_metric\\|^reward:" run.log
    """
    primary = metrics.get(task_spec.primary_metric, 0.0)
    reward = compute_reward(metrics, task_spec)

    print("\n---")
    print(f"primary_metric ({task_spec.primary_metric}): {primary:.6f}")
    print(f"reward:           {reward:.6f}")
    for name in sorted(metrics.keys()):
        if name != task_spec.primary_metric:
            print(f"{name}:{''.ljust(max(1, 18 - len(name)))} {metrics[name]:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"task:             {task_spec.name}")
    print(f"num_params:       {num_params}")
    for k, v in extra.items():
        print(f"{k}: {v}")
    print("---")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_peak_vram_mb() -> float:
    if torch.cuda.is_available():
        try:
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        except Exception:
            return 0.0
    return 0.0


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
