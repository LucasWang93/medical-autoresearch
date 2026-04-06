# Medical AutoResearch — RL for Clinical Prediction

An autonomous research system that discovers RL-based methods for clinical prediction tasks. Built on the autoresearch pattern (Karpathy) with PyHealth medical infrastructure.

## Setup

To set up a new experiment run:

1. **Agree on a run tag**: propose a tag (e.g. `apr6`). The branch `medical-ar/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b medical-ar/<tag>` from current main.
3. **Read the files**: The repo is small, read everything:
   - `README.md` — project context
   - `prepare.py` — fixed infrastructure (task registry, data, eval). **Do not modify**.
   - `train.py` — the file you modify. RL model, reward, training loop.
4. **Install dependencies**: `uv sync`
5. **Initialize `results.tsv`**: header row only. The baseline will be the first run.
6. **Confirm and go**.

## What You Modify

**Only `train.py`.** Everything in this file is fair game:

### Model Architecture
- `CodeEmbedding`: how medical codes are embedded (sum? attention? transformer?)
- `ClinicalRLModel`: the full model pipeline
- RNN type (GRU/LSTM/Transformer)
- History fusion strategy
- Output head design

### RL Algorithm
- `PolicyAgent`: the action selection policy
- `_compute_rl_loss()`: reward computation, discount, advantage estimation
- Can switch from REINFORCE to PPO, A2C, DQN, SAC, etc.
- Exploration strategy (entropy bonus, epsilon-greedy, etc.)

### Reward Shaping
- `shape_reward()`: custom reward components
- DDI penalty, diversity bonus, clinical relevance, etc.
- Delayed reward vs immediate reward design

### Hyperparameters
- All CAPS constants at the top of `train.py`

### Task Selection
- `TASK_NAME` — switch between registered tasks:
  - `drug_recommendation` (multilabel, default)
  - `mortality_prediction` (binary)
  - `readmission_prediction` (binary)
  - `length_of_stay` (multiclass)
- For multi-task: use `TaskRegistry.select_tasks()` and `TaskRegistry.get_multi_task_loader()`

## What You Cannot Modify

- `prepare.py` — the evaluation harness, data loading, and task registry are fixed.
- Do not install new packages beyond what's in `pyproject.toml`.
- Do not change the output format (the `print_results` function).

## The Goal

**Maximize the primary metric** for the chosen task(s).

For drug_recommendation: maximize `jaccard_samples` (predicted drug set overlap).
For mortality/readmission: maximize `auroc`.
For length_of_stay: maximize `f1_macro`.

Secondary goals:
- Minimize DDI rate (drug recommendation)
- Keep model simple when possible
- Explore multi-task synergies

## Slurm & GPU Resources

This system runs on a Slurm-managed cluster. **Never run training on the login node.**

### Partitions
| Partition | Use Case | Time Limit |
|-----------|----------|------------|
| `gpu_devel` | Quick tests, smoke runs | 10 min |
| `gpu` | Standard experiments | 30 min |
| `gpu_rtx6000` | RTX 6000 GPU nodes | 30 min |
| `gpu_h200` | H200 for large models | 30 min |
| `gpu_b200` | B200 for large models | 30 min |

### Running an experiment
```bash
# Quick test (gpu_devel, 10 min, default 5-min training budget)
sbatch scripts/run.sh

# Override task
sbatch scripts/run.sh --task mortality_prediction

# Longer training budget on better GPU
sbatch --partition=gpu --time=00:30:00 scripts/run.sh --time-budget 600

# Check status
squeue -u $USER

# Read results
grep "^primary_metric\|^reward:" results/run_<JOBID>.log
```

### CLI args for train.py
- `--task NAME` — override task (drug_recommendation, mortality_prediction, etc.)
- `--time-budget SEC` — override training time budget
- `--batch-size N` — override batch size
- `--n-patients N` — number of synthetic patients (default 2000)
- `--use-pyhealth` — use real MIMIC data via PyHealth
- `--data-root PATH` — MIMIC data root (with --use-pyhealth)

## Training Budget

Each run trains for **5 minutes** (wall clock) by default, configurable via `--time-budget`. The metric is computed on a held-out test set after training.

## Output Format

The training script prints a summary:

```
---
primary_metric (jaccard_samples): 0.452300
reward:           0.452300
f1_samples:       0.561200
pr_auc_samples:   0.623400
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     4560.2
task:             drug_recommendation
num_params:       1234567
---
```

Extract key metric: `grep "^primary_metric\|^reward:" results/run_<JOBID>.log`

## Logging Results

Log to `results.tsv` (tab-separated):

```
commit	job_id	primary_metric	reward	memory_gb	status	task	description
a1b2c3d	123456	0.452300	0.452300	4.5	keep	drug_recommendation	baseline REINFORCE
b2c3d4e	123457	0.481200	0.471200	4.6	keep	drug_recommendation	PPO with entropy decay
c3d4e5f	123458	0.000000	0.000000	0.0	crash	drug_recommendation	transformer encoder OOM
```

## The Experiment Loop

LOOP FOREVER:

1. Look at the current state (git, results.tsv, recent job logs)
2. Form a hypothesis and modify `train.py`
3. `git commit`
4. Submit: `sbatch scripts/run.sh` (or with overrides)
5. Wait for completion: `squeue -u $USER` then read log
6. Read results: `grep "^primary_metric\|^reward:\|^peak_vram_mb:" results/run_<JOBID>.log`
7. If grep is empty → crash. `tail -n 50 results/run_<JOBID>.log` for stack trace.
8. Log to `results.tsv`
9. If improved → keep commit
10. If not improved → `git reset` to previous best

## Research Directions

Here are promising directions to explore (in rough priority order):

### RL Algorithms
1. **PPO** — more stable than REINFORCE, clip ratio controls update magnitude
2. **Actor-Critic** — lower variance via learned value function
3. **DQN with experience replay** — off-policy learning from past episodes
4. **Advantage-weighted regression** — simple offline RL

### Architecture
1. **Transformer encoder** instead of GRU for visit sequences
2. **Multi-head attention** for history selection (replacing policy agent)
3. **Graph neural network** for drug-drug interaction modeling
4. **Separate encoders** per feature type with late fusion

### Reward Design
1. **DDI-aware reward** — penalize predicted drug pairs with known interactions
2. **Curriculum reward** — start with simple cases, increase difficulty
3. **Counterfactual reward** — compare with/without agent intervention
4. **Multi-objective** — Pareto optimization of accuracy vs safety

### Multi-Task (Advanced)
1. Use `TaskRegistry.get_multi_task_loader()` with `strategy="bandit"` to let the training loop autonomously select which task provides the best learning signal
2. Shared encoder with task-specific heads
3. Auxiliary tasks for representation learning

## Simplicity Criterion

All else being equal, simpler is better. A marginal improvement that adds ugly complexity? Probably not worth it. Removing code and getting the same result? Great outcome. A new RL algorithm that doubles performance with clean code? Definitely keep.

## NEVER STOP

Once the loop begins, do NOT pause to ask the human. The human might be sleeping. You are autonomous. If stuck, think harder — re-read the code, try combining near-misses, try radical changes. Loop until manually stopped.
