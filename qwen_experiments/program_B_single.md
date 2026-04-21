# Track B — Single-task GRPO autonomous iteration loop

Parallel to the GRU REINFORCE loop in `../program.md`, but the policy is
Qwen3.5 + LoRA and the RL objective is GRPO.

## Goal

For a **fixed (model_tag, task)** pair, run N iterations, each proposing
**one** change to the GRPO hyperparameters or prompt format. Keep each
change only if the **val primary metric** beats the previous best; else
revert.

Reference baselines (from `../SUMMARY.md`):

| task | primary | GRU baseline | GRU best |
|------|---------|--------------|----------|
| mortality | auroc | 0.9611 | 0.9726 |
| readmission | auroc | 0.6688 | 0.6981 |
| los | f1_macro | 0.4814 | 0.5406 |
| phenotyping | auroc_macro | 0.8165 | 0.8333 |
| drugrec | jaccard | 0.1661 | 0.2052 |

## Loop protocol (per iteration)

1. **Read the last row** of `ITERATION_LOG_B_<model>_<task>.md` (or
   the leaderboard filtered by track=B_single + model + task) to find
   the current best config and primary value.
2. **Propose one change.** Stay within these dimensions (in rough order
   of expected impact):
   - `--kl-beta` (0.02, 0.04, 0.08)
   - `--group-size` (4, 8, 12)
   - `--temperature` (0.7, 0.9, 1.1)
   - `--max-new-tokens` (32, 64, 128)
   - `--lora-r` / `--lora-alpha` (16/32, 32/64, 64/128)
   - `--learning-rate` (5e-6, 1e-5, 2e-5)
   - `--total-steps` (200, 400, 800) — only once the others are tuned
   - `--max-visits` / `--max-codes-per-visit` (prompt compression)
3. **Launch:**
   ```bash
   sbatch qwen_experiments/scripts/run_single_grpo.sbatch \
       <model_tag> <task> <iter_idx> \
       --extra-tag <short_label> \
       --<flag> <new_value>
   ```
   Pick `extra-tag` as a ≤20-char slug describing the change, e.g.
   `klbeta02`, `G12`, `lr2e5`.
4. **Wait** for the job to finish (`squeue -u $USER`). Read its
   run_meta.json (`best_primary`, `best_val`) and the train_log.jsonl
   (last `event: eval` entries).
5. **Decision:**
   - If `best_primary > previous_best`: **KEEP**. The best-adapter is
     persisted in `runs/B_single/<run_id>/lora_adapter_best/`.
   - Else: **REVERT**. No adapter deletion needed; just note the
     negative result.
6. **Append** to `ITERATION_LOG_B_<model>_<task>.md`:

```
| iter | commit_sha | job_id | config_delta | primary | Δ vs prev best | decision |
|------|------------|--------|--------------|---------|----------------|----------|
```

## Stop conditions

- No improvement for 3 iterations in a row AND >5 iterations already
  completed → declare task-saturated for this model at this architecture.
- Single iteration wall-clock > 4 h on H200 → shrink `--total-steps` or
  `--group-size` before the next iter.

## What NOT to iterate on

- Reward function: already finalized in `src/rewards.py`. Only touch it
  if parse_rate < 0.5 persists.
- Eval protocol: metrics/splits are fixed (comparable to GRU baseline).
