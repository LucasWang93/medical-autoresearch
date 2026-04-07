# Medical AutoResearch

Autonomous RL research system for clinical prediction tasks. Combines Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) experiment loop with [PyHealth](https://github.com/sunlabuiuc/PyHealth) medical infrastructure.

The system iterates autonomously: modify model → submit GPU job → evaluate → keep or revert → repeat.

## Project Structure

```
├── prepare.py          # FIXED: task registry, synthetic EHR data, PyHealth loader, evaluation harness
├── train.py            # AGENT MODIFIES: RL model, reward shaping, training loop
├── program.md          # Autonomous experiment loop instructions for Claude Code
├── scripts/run.sh      # Slurm GPU job launcher
├── tests/              # 112 unit tests
│   ├── test_prepare.py
│   └── test_train.py
├── results/            # Experiment logs (run_<JOBID>.log)
├── results.tsv         # Experiment results tracking
├── CLAUDE.md           # Project context for Claude Code
└── pyproject.toml
```

## Available Tasks

| Task | Type | Primary Metric | Description |
|------|------|---------------|-------------|
| `drug_recommendation` | multilabel | jaccard_samples | Drug combination prediction with DDI safety |
| `mortality_prediction` | binary | auroc | In-hospital mortality from visit history |
| `readmission_prediction` | binary | auroc | 30-day readmission prediction |
| `length_of_stay` | multiclass | f1_macro | LOS bucket (short/medium/long) |

## Quick Start

### Local (CPU, synthetic data)

```bash
# Single task, short run
python train.py --time-budget 60 --n-patients 500

# Switch task
python train.py --task mortality_prediction --time-budget 60

# Minimal smoke test
python train.py --time-budget 0 --n-patients 10 --batch-size 2
```

### Slurm Cluster (GPU)

```bash
# Quick test (gpu_devel, 10 min wall time, 5 min training)
sbatch scripts/run.sh

# Override task
sbatch scripts/run.sh --task mortality_prediction

# Longer training on standard GPU
sbatch --partition=gpu --time=00:30:00 scripts/run.sh --time-budget 600

# Scavenge partition (faster queue, preemptible)
sbatch --partition=scavenge_gpu --time=00:10:00 scripts/run.sh --time-budget 120

# Run all 4 task baselines
for task in drug_recommendation mortality_prediction readmission_prediction length_of_stay; do
  sbatch --partition=scavenge_gpu --time=00:10:00 scripts/run.sh --task $task --time-budget 120
done
```

### Check Results

```bash
# Job status
squeue -u $USER

# Extract metrics from log
grep "^primary_metric\|^reward:" results/run_<JOBID>.log

# Full summary
tail -20 results/run_<JOBID>.log

# If empty output → job crashed, check error:
tail -50 results/run_<JOBID>.log
```

### Real MIMIC Data (via PyHealth)

```bash
python train.py --use-pyhealth --data-root /path/to/mimiciii
```

Falls back to synthetic data if PyHealth loading fails.

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | drug_recommendation | Task name |
| `--time-budget` | 300 (5 min) | Training wall-clock seconds |
| `--batch-size` | 32 | Batch size |
| `--n-patients` | 2000 | Synthetic patients to generate |
| `--use-pyhealth` | false | Use real MIMIC data |
| `--data-root` | None | MIMIC data directory |

## Tests

```bash
# Run all 112 tests (~100s)
python -m unittest discover -s tests

# Run specific test file
python -m unittest tests.test_prepare -v
python -m unittest tests.test_train -v
```

Coverage:
- Task registry, all 4 task specs
- Reward computation, metric calculation, DDI rate
- Synthetic data generation (all task types)
- Collation and padding
- PyHealth data normalization
- Multi-task loader (round_robin, bandit, proportional)
- Model components (CodeEmbedding, PolicyAgent, ClinicalRLModel)
- Forward pass for all task types + edge cases
- RL loss computation
- Reward shaping
- End-to-end evaluation
- CLI argument parsing

## Autonomous Experiment Loop

The core workflow (defined in `program.md`):

1. Read current state (`git log`, `results.tsv`, recent logs)
2. Form hypothesis, modify `train.py`
3. `git commit`
4. `sbatch scripts/run.sh` → wait → read results
5. If improved → keep commit. If not → `git reset`
6. Log to `results.tsv`
7. Repeat forever

To start: point Claude Code at this repo and say "按照 program.md 开始自主实验循环".

### Results Tracking (results.tsv)

Tab-separated, one row per experiment:

```
commit	job_id	primary_metric	reward	memory_gb	status	task	description
baseline	7439080	0.654256	0.000000	0.0	keep	drug_recommendation	baseline REINFORCE, 60s budget
```

## Cluster Partitions

| Partition | Use Case | GPUs |
|-----------|----------|------|
| `gpu_devel` | Quick tests (10 min) | Mixed |
| `gpu` | Standard runs (30 min) | Mixed |
| `scavenge_gpu` | Preemptible, faster queue | Mixed |
| `gpu_h200` | Large models | H200 |
| `gpu_b200` | Large models | B200 |

Account: `pi_yz875`

## Dependencies

- System module: `PyTorch/2.7.1-foss-2024a-CUDA-12.6.0`
- pip: `scikit-learn` (auto-installed by run.sh)
- Optional: `pyhealth` (for real MIMIC data)

## Current Baseline

| Task | Metric | Score | Job |
|------|--------|-------|-----|
| drug_recommendation | jaccard_samples | 0.654 | 7439080 |
| mortality_prediction | auroc | — | pending |
| readmission_prediction | auroc | — | pending |
| length_of_stay | f1_macro | — | pending |
