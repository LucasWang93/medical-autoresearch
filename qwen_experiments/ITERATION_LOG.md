# Qwen_experiments iteration log (entry point)

One row per completed run (any track). For detailed per-task autonomous
iteration traces, see:

- `ITERATION_LOG_B_<model>_<task>.md` — Track B (single-task GRPO)
- `ITERATION_LOG_C_<model>.md` — Track C (multi-task GRPO)

For the machine-readable leaderboard (appended by every script), see
`leaderboard.tsv`.

## High-level milestones

- [x] 2026-04-21 — skeleton + prompts + GRPO core merged; data caches
      building; smoke test submitted.
- [x] Track A — all 5 tasks × {0, 5} shots × {4B, 9B} evaluated. (20 runs,
      commit 6ff402b.)
- [x] Track B — iter 0 on 4B + 9B for each task. (10 runs, commit d667674.)
- [x] Track C — iter 0 on 4B + 9B. (2 runs, commit ed8b0f0. Buggy bandit;
      see INSIGHTS.md §F3.)
- [ ] Track B — first round of hyperparam iterations (≥3 per task) —
      BLOCKED on reward-shaping fixes (INSIGHTS.md §5).
- [ ] Track C — first round of multi-task iterations (≥3 total) —
      BLOCKED on same.

## Iter 0 synthesis

See `INSIGHTS.md` for the full write-up. TL;DR: mode collapse (every
LOS run converges to F1_macro=0.18152866, every drugrec run to
Jaccard=0.180) driven by class-imbalanced flat reward + format bonus +
ineffective KL clamp. 9B regresses vs its 0-shot baseline at lr=1e-5.
Bandit was buggy for all of iter 0 (fix landed mid-run).
