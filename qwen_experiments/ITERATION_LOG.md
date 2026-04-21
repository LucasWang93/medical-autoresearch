# Qwen_experiments iteration log (entry point)

One row per completed run (any track). For detailed per-task autonomous
iteration traces, see:

- `ITERATION_LOG_B_<model>_<task>.md` — Track B (single-task GRPO)
- `ITERATION_LOG_C_<model>.md` — Track C (multi-task GRPO)

For the machine-readable leaderboard (appended by every script), see
`leaderboard.tsv`.

## High-level milestones

*(fill in as runs complete)*

- [ ] 2026-04-21 — skeleton + prompts + GRPO core merged; data caches
      building; smoke test submitted.
- [ ] Track A — all 5 tasks × {0, 5} shots × {4B, 9B} evaluated.
- [ ] Track B — iter 0 on 9B for each task.
- [ ] Track C — iter 0 on 9B.
- [ ] Track B — first round of hyperparam iterations (≥3 per task).
- [ ] Track C — first round of multi-task iterations (≥3 total).
