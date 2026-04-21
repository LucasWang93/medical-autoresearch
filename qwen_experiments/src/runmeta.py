"""Run metadata + append-only leaderboard.

Every run (Track A/B/C) writes a standard `run_meta.json` capturing:
  - git commit sha of medical-autoresearch/ at run time
  - slurm job id, hostname, cuda visible devices
  - full argv, seed, model path + tag, data root
  - wall-clock start/end

Every run appends a single line to qwen_experiments/leaderboard.tsv after
its metrics are written, so the full history is one grep away.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


QEXP_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = QEXP_ROOT.parent
LEADERBOARD = QEXP_ROOT / "leaderboard.tsv"
LEADERBOARD_HEADER = (
    "timestamp\trun_id\ttrack\tmodel\ttask\tshots\titer\tmetric\tvalue\t"
    "delta_vs_ref\tgit_sha\tjob_id\thost\tseed\tnotes\n"
)


def git_sha(repo: Path = REPO_ROOT) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "nogit"


def ensure_leaderboard() -> None:
    if not LEADERBOARD.exists():
        LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
        LEADERBOARD.write_text(LEADERBOARD_HEADER)


def append_leaderboard(row: Dict[str, Any]) -> None:
    ensure_leaderboard()
    fields = [
        "timestamp", "run_id", "track", "model", "task", "shots", "iter",
        "metric", "value", "delta_vs_ref", "git_sha", "job_id", "host",
        "seed", "notes",
    ]
    line = "\t".join(str(row.get(f, "")) for f in fields) + "\n"
    with open(LEADERBOARD, "a") as f:
        f.write(line)


def make_run_dir(
    track: str,
    model_tag: str,
    task: str,
    *,
    shots: Optional[int] = None,
    iter_idx: Optional[int] = None,
    extra_tag: str = "",
) -> Path:
    """Create a time-stamped, human-readable run directory.

    Naming:
      A_baseline/{model}__{task}__{shots}shot__{ts}__{sha}/
      B_single/{model}__{task}__iter{N}__{ts}__{sha}/
      C_multi/{model}__all__iter{N}__{ts}__{sha}/
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sha = git_sha()
    parts = [model_tag, task]
    if shots is not None:
        parts.append(f"{shots}shot")
    if iter_idx is not None:
        parts.append(f"iter{iter_idx:02d}")
    if extra_tag:
        parts.append(extra_tag)
    parts.extend([ts, sha])
    name = "__".join(parts)
    d = QEXP_ROOT / "runs" / track / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_run_meta(
    run_dir: Path,
    *,
    track: str,
    model_path: str,
    model_tag: str,
    task: str,
    seed: int,
    args: Dict[str, Any],
    shots: Optional[int] = None,
    iter_idx: Optional[int] = None,
    notes: str = "",
) -> Dict[str, Any]:
    meta = {
        "track": track,
        "run_id": run_dir.name,
        "model_path": model_path,
        "model_tag": model_tag,
        "task": task,
        "shots": shots,
        "iter": iter_idx,
        "seed": seed,
        "git_sha": git_sha(),
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "argv": sys.argv,
        "args": args,
        "notes": notes,
        "start_time": datetime.now().isoformat(timespec="seconds"),
    }
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def finalize_run_meta(run_dir: Path, meta: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    meta["end_time"] = datetime.now().isoformat(timespec="seconds")
    try:
        t0 = datetime.fromisoformat(meta["start_time"])
        t1 = datetime.fromisoformat(meta["end_time"])
        meta["duration_s"] = int((t1 - t0).total_seconds())
    except Exception:
        meta["duration_s"] = None
    meta["metrics"] = metrics
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


class JsonlWriter:
    """Streaming JSONL writer for per-sample predictions / step traces."""
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "w", buffering=1)  # line-buffered

    def write(self, obj: Dict[str, Any]) -> None:
        self._f.write(json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n")

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _json_default(obj):
    import numpy as np
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def now_log(tag: str, **kw) -> None:
    payload = {"event": tag, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **kw}
    print(json.dumps(payload, ensure_ascii=False, default=_json_default), flush=True)
