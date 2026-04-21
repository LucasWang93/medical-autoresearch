# Track C — Multi-task GRPO joint training

Parallel to the multi-task-loop plan in `../.multitask_loop_plan.md`,
but the policy is Qwen3.5 + a single shared LoRA adapter, and the
objective is GRPO.

Default tasks: `mortality, readmission, los, phenotyping`
(drugrec optional — pass `--tasks mimic4_mortality,...,mimic4_drugrec`).

## Goal

Start from the best Track-A baseline (or from a fresh base model) and
iterate, each time proposing **one** joint-training change. Keep if the
`combined = mean(per_task_primary)` on val improves; else revert.

## Loop protocol (per iteration)

1. Read `ITERATION_LOG_C_<model>.md` and the leaderboard to find the
   current best combined value + the config that achieved it.
2. Propose one change. Categories:
   - **Task sampling**: `--selection round_robin` vs `bandit`.
   - **Task weights**: `--task-weights "mimic4_phenotyping=3,mimic4_los=1.5"`.
     (Mirror the GRU multitask findings as a prior.)
   - **GRPO knobs**: `--group-size`, `--kl-beta`, `--temperature`,
     `--max-new-tokens`.
   - **Schedule**: `--total-steps`, `--learning-rate`, `--warmup-steps`.
   - **Capacity**: `--lora-r` / `--lora-alpha`.
3. Launch:
   ```bash
   sbatch qwen_experiments/scripts/run_multi_grpo.sbatch \
       <model_tag> <iter_idx> \
       --extra-tag <short_label> \
       --<flag> <new_value>
   ```
4. After the job completes, read `run_meta.json::metrics.best_combined`
   and `best_per_task`.
5. Decision — KEEP if `best_combined` improves; REVERT otherwise.
6. Append row to `ITERATION_LOG_C_<model>.md`:

```
| iter | commit_sha | job_id | change | combined | Δ | per_task | decision |
|------|------------|--------|--------|----------|---|----------|----------|
```

## Parity guardrails vs GRU multitask study

- The GRU study used round-robin → task-weighted (phen 2x, los 1.5x) →
  longer training → hidden 384 → phen 3x → 7200s T_max=25, landing at
  combined test 0.7671.
- Track C should **not** mirror those changes blindly: LLM has different
  dynamics. But if no idea proposed, start from `{selection=bandit,
  weights=default}` and let the bandit discover the weighting.

## Stop conditions

- 3 iterations no-improvement + ≥5 iterations total → declare saturation.
- Any single task's val primary collapses by >0.03 vs its best on the
  shared adapter → flag potential **negative transfer**, consider
  adding per-task residual adapter as the next change (requires
  extending the code — not a vanilla iteration).
