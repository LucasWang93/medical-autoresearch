# Medical AutoResearch

Autonomous RL research system for clinical prediction tasks. Combines Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) experiment loop with [PyHealth](https://github.com/sunlabuiuc/PyHealth) medical infrastructure.

The system iterates autonomously: modify model → submit GPU job → evaluate → keep or revert → repeat.

## Project Structure

```
├── prepare.py          # FIXED: task registry, data loaders (synthetic/SUPPORT2/MIMIC-IV), evaluation
├── train.py            # AGENT MODIFIES: RL model, reward shaping, training loop
├── program.md          # Autonomous experiment loop instructions for Claude Code
├── BASELINES.md        # Baseline results, model architecture, analysis
├── scripts/run.sh      # Slurm GPU job launcher
├── tests/              # Unit tests (prepare + train)
│   ├── test_prepare.py
│   └── test_train.py
├── results/            # Experiment logs (run_<JOBID>.log)
├── results.tsv         # Experiment results tracking
├── CLAUDE.md           # Project context for Claude Code
└── pyproject.toml
```

## Available Tasks (11 total)

### MIMIC-IV Real Data (223K patients)
| Task | Type | Metric | Baseline |
|------|------|--------|----------|
| `mimic4_mortality` | binary | AUROC | 0.961 |
| `mimic4_readmission` | binary | AUROC | 0.669 |
| `mimic4_los` | 4-class | F1_macro | 0.481 |
| `mimic4_drugrec` | multilabel | Jaccard | 0.166 |

### SUPPORT2 Real Data (9K patients)
| Task | Type | Metric | Baseline |
|------|------|--------|----------|
| `support2_mortality` | binary | AUROC | 0.916 |
| `support2_dzclass` | 4-class | F1_macro | 0.779 |
| `support2_survival` | binary | AUROC | 0.936 |

### Synthetic Data (development)
| Task | Type | Metric | Baseline |
|------|------|--------|----------|
| `drug_recommendation` | multilabel | Jaccard | 0.654 |
| `mortality_prediction` | binary | AUROC | — |
| `readmission_prediction` | binary | AUROC | — |
| `length_of_stay` | multiclass | F1_macro | — |

See [BASELINES.md](BASELINES.md) for detailed model architecture, training setup, and analysis.

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

### MIMIC-IV Tasks (real data, auto-detected)

```bash
# MIMIC-IV tasks auto-detect local data at the configured path
sbatch --partition=gpu --time=00:20:00 scripts/run.sh --task mimic4_mortality --time-budget 120

# Run all MIMIC-IV baselines
for task in mimic4_mortality mimic4_readmission mimic4_los mimic4_drugrec; do
  sbatch --partition=gpu --time=00:20:00 scripts/run.sh --task $task --time-budget 120
done
```

### SUPPORT2 Tasks (auto-downloads from HuggingFace)

```bash
sbatch scripts/run.sh --task support2_mortality --time-budget 120
```

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
# Run all tests (may OOM on login node — run file-by-file if needed)
python -m unittest discover -s tests

# Run by file (recommended on login nodes)
python -m unittest tests.test_prepare -v
python -m unittest tests.test_train -v

# Run only MIMIC-IV tests
python -m unittest tests.test_prepare.TestMIMIC4Tasks tests.test_prepare.TestMIMIC4DataLoading -v
```

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
- pip: `scikit-learn datasets pandas` (auto-installed by run.sh)
- MIMIC-IV 3.1 data (for `mimic4_*` tasks, local access required)
- SUPPORT2 data (for `support2_*` tasks, auto-downloaded from HuggingFace)
