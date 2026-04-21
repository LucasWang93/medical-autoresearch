# qwen_experiments — Qwen3.5 LLM re-run of the MIMIC-IV study

Parallel pipeline to the GRU+RL baseline in the parent directory, using
Qwen3.5-4B and Qwen3.5-9B as the policy.

## Three tracks (run in order)

| Track | What | Script |
|-------|------|--------|
| **A** | Few-shot baseline (0/5-shot, inference only) | `src/baseline_fewshot.py` |
| **B** | Single-task GRPO LoRA + autonomous iter | `src/grpo_single.py` + `src/single_iter_loop.py` |
| **C** | Multi-task GRPO (shared LoRA + bandit + per-task group norm) | `src/grpo_multi.py` + `src/multi_iter_loop.py` |

## Data

5 MIMIC-IV tasks from `../prepare.py`:

- `mimic4_mortality` — AUROC (bin)
- `mimic4_readmission` — AUROC (bin)
- `mimic4_los` — F1_macro (4-bucket multiclass)
- `mimic4_phenotyping` — AUROC_macro (25-label multilabel)
- `mimic4_drugrec` — Jaccard (top-300 multilabel)

Split: 80/10/10, deterministic by sample-index + seed (see `src/data.py`).
Raw patient → text conversion is cached at `.cache/data/<task>.jsonl`
after the first run so later runs avoid the ~minute-level CSV read.

## Reference numbers (from SUMMARY.md, GRU baseline)

| task | metric | GRU baseline | GRU best | multitask best |
|------|--------|--------------|----------|----------------|
| mortality | AUROC | 0.9611 | 0.9726 | 0.9755 |
| readmission | AUROC | 0.6688 | 0.6981 | 0.7117 |
| los | F1_macro | 0.4814 | 0.5406 | 0.5548 |
| phenotyping | AUROC_macro | 0.8165 | 0.8333 | 0.8264 |
| drugrec | Jaccard | 0.1661 | 0.2052 | — (excluded) |

## Output directory layout

```
qwen_experiments/
├── runs/
│   ├── A_baseline/{model}__{task}__{shots}shot__{ts}__{sha}/
│   ├── B_single/{model}__{task}__iter{NN}__{ts}__{sha}/
│   └── C_multi/{model}__all__iter{NN}__{ts}__{sha}/
├── leaderboard.tsv          # append-only: every run adds a row
├── ITERATION_LOG.md         # tracks Track B/C decisions (keep/revert)
└── .cache/data/             # text-sample JSONL cache
```

Each run directory contains:
- `run_meta.json` — git sha, slurm job id, hostname, args, start/end, duration, metrics
- `metrics.json` — aggregate primary metric + all secondaries
- `predictions.jsonl` — per-sample `{sample_idx, prompt_hash, raw_output, parsed, label, reward, prob?}`
- `stdout.log` / `stderr.log` (via sbatch `--output=`)
- For B/C: `lora_adapter/`, `train_log.jsonl` (per-step loss / reward stats), `iter_config.json`

## Models

| tag | path | size |
|-----|------|------|
| `qwen35-4b` | `/nfs/roberts/project/pi_yz875/sw2572/models/Qwen3.5-4B` | ~9 GB bf16 |
| `qwen35-9b` | `/nfs/roberts/project/pi_yz875/sw2572/models/Qwen3.5-9B` | ~19 GB bf16 |

## Environment

Existing venv + module-loaded torch: `source ../../../../gpus/env_qwen35.sh`.

## Commands (see scripts/ for the launchers)

```bash
# Track A — both models × 5 tasks × {0, 5} shots
sbatch qwen_experiments/scripts/run_baseline.sbatch qwen35-9b mimic4_mortality 0
sbatch qwen_experiments/scripts/run_baseline.sbatch qwen35-9b mimic4_mortality 5
# ... (driver script loops all 5 tasks)

# Track B — single-task autonomous GRPO iteration
sbatch qwen_experiments/scripts/run_single_grpo.sbatch qwen35-9b mimic4_los 0

# Track C — multi-task GRPO
sbatch qwen_experiments/scripts/run_multi_grpo.sbatch qwen35-9b 0
```

Leaderboard at any time: `tail -n 50 qwen_experiments/leaderboard.tsv | column -t -s$'\t'`
