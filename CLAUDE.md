# Medical AutoResearch

Autonomous RL research system for clinical prediction, combining
autoresearch (Karpathy) experiment loop + PyHealth medical infrastructure.

## Project Location
`/nfs/roberts/project/pi_yz875/sw2572/codes/auto-research-health/medical-autoresearch/`

## Architecture
- `prepare.py` — FIXED: TaskRegistry with multi-task interface, synthetic EHR data, evaluation harness
- `train.py` — AGENT MODIFIES: REINFORCE-based clinical RL model, reward shaping, training loop
- `program.md` — Agent instructions (autoresearch-style autonomous loop + Slurm workflow)
- `scripts/run.sh` — Slurm job launcher
- `results/` — Experiment logs (`run_<JOBID>.log`)

## Current Status (2026-04-06)

### MVP framework implemented, first GPU test in progress

1. **Data pipeline**: Working. Synthetic EHR data generates realistic patient trajectories
   with disease archetypes, DDI matrix, and learnable patterns. Tested on login node.
2. **Model**: REINFORCE-based ClinicalRLModel with PolicyAgent for history selection.
3. **Slurm integration**: Fixed after two iterations:
   - Fix 1: hardcoded BASEDIR (Slurm changes $0 on compute nodes)
   - Fix 2: use system `module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0` instead of
     pip-installed torch 2.11 (CUDA 13.0 driver mismatch on cluster)
   - Fix 3: `total_mem` → `total_memory` attribute name for PyTorch 2.7
4. **Test job**: Job 7439080 submitted to scavenge_gpu, was RUNNING on a1118u03n01 (NVIDIA L40S).

### What to do next

1. **Check if test job finished**:
   ```bash
   squeue -u $USER
   cat results/run_7439080.log
   grep "^primary_metric\|^reward:" results/run_7439080.log
   ```

2. **If job succeeded** — the full pipeline works. Next steps:
   - `git add -A && git commit` to save fixes
   - Run baseline on all 4 tasks to establish baselines
   - Start the autonomous experiment loop (per program.md)

3. **If job failed** — read the log for errors:
   ```bash
   tail -50 results/run_7439080.log
   ```
   Common issues:
   - OOM → reduce BATCH_SIZE or EMBEDDING_DIM in train.py
   - Module not found → check `module load` in scripts/run.sh
   - CUDA error → ensure PyTorch module matches GPU driver

4. **Run all 4 task baselines**:
   ```bash
   for task in drug_recommendation mortality_prediction readmission_prediction length_of_stay; do
     sbatch --partition=scavenge_gpu --time=00:10:00 scripts/run.sh --task $task --time-budget 120
   done
   ```

5. **Start autonomous loop**: point Claude Code at `program.md` and let it iterate on `train.py`

## Slurm Configuration
- **Account**: pi_yz875
- **Quick test**: `sbatch scripts/run.sh --time-budget 60 --n-patients 500`
- **Standard**: `sbatch --partition=gpu --time=00:30:00 scripts/run.sh`
- **Large model**: `sbatch --partition=gpu_h200 --time=00:30:00 scripts/run.sh`
- **Scavenge** (faster queue): `sbatch --partition=scavenge_gpu --time=00:10:00 scripts/run.sh`
- Logs: `results/run_<JOBID>.log`
- Check: `squeue -u $USER`
- Results: `grep "^primary_metric\|^reward:" results/run_<JOBID>.log`

## Available Tasks
| Task | Type | Primary Metric | RL Angle |
|------|------|---------------|----------|
| drug_recommendation | multilabel | jaccard_samples | Sequential drug decisions + DDI safety |
| mortality_prediction | binary | auroc | Adaptive history attention |
| readmission_prediction | binary | auroc | Visit pattern recognition |
| length_of_stay | multiclass | f1_macro | Sequential feature selection |

## CLI Args (train.py)
- `--task NAME` — override task
- `--time-budget SEC` — training time budget (default 300)
- `--batch-size N` — batch size (default 32)
- `--n-patients N` — synthetic patients (default 2000)
- `--use-pyhealth --data-root PATH` — use real MIMIC data

## Multi-Task Interface (reserved for future)
`TaskRegistry.get_multi_task_loader(specs, strategy="bandit")` returns a
`MultiTaskLoader` with UCB-based autonomous task selection. The agent can
call `loader.update_reward(task_name, reward)` to feed learning progress
back. Strategies: round_robin, proportional, bandit.

## Dependencies
- System module: `PyTorch/2.7.1-foss-2024a-CUDA-12.6.0`
- User pip: `scikit-learn` (auto-installed by run.sh)
- PyHealth at `../PyHealth/` (for future real-data mode)

## Related Repos
- `../autoresearch/` — Karpathy's original autoresearch (reference for loop pattern)
- `../PyHealth/` — UIUC Sun Lab medical AI library (data/models/tasks)
- `../../medqa-research/` — MedMemory project (prior RL-adjacent medical QA work)
