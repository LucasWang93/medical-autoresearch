# Qwen3.5 GRPO Study — Code Map + Iter 0 Insights

Last updated: 2026-04-22, after Track C iter 0 landed.

## 1. Code map

```
qwen_experiments/
├── src/
│   ├── data.py            (209)  MIMIC-IV split loader + TextSample cache
│   ├── prompts.py         (283)  chat messages + answer-parsers (per task)
│   ├── rewards.py         (108)  per-task reward + format bonus
│   ├── metrics.py         (161)  sklearn metrics (binary/multi-class/label)
│   ├── models.py          (152)  Qwen3.5 load + LoRA apply + adapter_disabled
│   ├── grpo_core.py       (227)  GRPO loss (group norm, k3 KL, response mask)
│   ├── baseline_fewshot.py(412)  Track A — 0/5-shot inference
│   ├── grpo_single.py     (309)  Track B — single-task GRPO+LoRA
│   ├── grpo_multi.py      (350)  Track C — multi-task GRPO+UCB bandit
│   └── runmeta.py         (181)  run-dir, meta, leaderboard.tsv writer
├── scripts/               sbatch templates for h200
├── runs/{A_baseline,B_single,C_multi,_meta}/
├── leaderboard.tsv        49 rows (1 header + 48 runs)
└── INSIGHTS.md            ← you are here
```

Key invariants across the three tracks:

- **Reward is task-specific** (`rewards.py::compute_reward`) with a flat
  `FORMAT_BONUS = 0.05` on successful parse.
- **Reference model** = same weights with LoRA disabled
  (`adapter_disabled` context manager in grpo_single/multi).
- **KL estimator** = k3 low-variance form `exp(r) − r − 1`.
- **Advantage** = within-group z-score, per-task.

## 2. Iter 0 result table

| Run | Metric | Value | Δ vs GRU baseline |
|---|---|---|---|
| A 9b mortality 0-shot | AUROC | 0.636 | −0.325 |
| A 9b readmission 0-shot | AUROC | 0.577 | −0.092 |
| A 9b LOS 0-shot | F1_macro | 0.202 | −0.280 |
| A 9b drugrec 0-shot | Jaccard | 0.124 | −0.042 (parse only 1%) |
| B 4b mortality | AUROC | **0.924** | −0.037 |
| B 4b readmission | AUROC | 0.519 | −0.150 |
| B 9b mortality | AUROC | 0.505 | −0.456 ← regression |
| B 9b readmission | AUROC | 0.500 | −0.169 |
| B {4b,9b} LOS | F1_macro | 0.182 (identical) | −0.300 (collapse) |
| B {4b,9b} drugrec | Jaccard | 0.180 (identical) | +0.014 (collapse) |
| B {4b,9b} phenotyping | F1_samples | 0.130–0.133 | n/a (no GRU ref) |
| C 4b multi | combined | 0.410 | mort 0.823, others collapsed |
| C 9b multi | combined | 0.329 | all tasks collapsed |

## 3. What the train logs tell us (4b Track C, 400 steps, 3h18m)

Bandit task selection (buggy, fix landed mid-run):
- Step 80: mortality picked 100 times, all others 0.
- Step 160: mortality 100, los 54, readmission 1, phenotyping 1.
- Step 399: mortality 0, readmission 1, los 0, phenotyping 0 — essentially
  degenerate.

KL to base model (λ=0.04):
- step 1: 0.0
- step 135: 0.74
- step 399: **44.4** ← massive drift. At this KL, the "KL regularizer" is
  clipped by the `.clamp(-20, 20)` inside `kl_k3`, so the pull back toward
  the reference is near-constant and ineffective.

Training reward (averaged, with format bonus):
- step 1: 0.39
- step 135: 0.53
- step 399: 0.92 ← near-maximal.

But val primary:
- mortality AUROC drops from 0.82 @ step 160 → 0.49 @ step 320 while
  parse_rate goes from 0.36 → 0.99 and `accuracy` jumps to 0.99 (= class
  prevalence).

Diagnosis: the model learns to always output the majority class. On
mortality the positive rate is ~1%, so "always predict 0" gives
`correctness = 0.99`, plus `format_bonus = 0.05`, total reward 1.04 —
nearly the maximum achievable. **The reward signal does not distinguish a
good classifier from a constant predictor.**

## 4. Four compounding failure modes in iter 0

### F1 — Class-imbalanced flat reward (main driver)
`_binary_reward` returns 1 if correct else 0. With 99% negative class, the
majority predictor gets 0.99 reward and dominates the group once the
policy lands on it. No advantage for learning to discriminate.

### F2 — Format bonus reinforces degenerate output
Any parseable output gets +0.05. "predict class 0" parses fine, so the
first degenerate sample often has the highest group reward → advantage
amplifies it → the whole group converges on it within ~50 steps.

### F3 — Bandit exploration bug (fixed mid-run, commit 293f1f3)
`UCBTaskBandit.counts[t]` was only incremented inside `update_after_eval`,
not on every `select()`. For the first `eval_every=80` steps, the bandit
kept returning the first task with `counts=0` (mortality). After the
first eval, mortality had 100 pulls and kept dominating UCB. Both iter 0
Track C runs used the buggy bandit.

### F4 — 9B lr=1e-5 overshoots
9B Qwen has ~2.5× the capacity of 4B; the same LR pushes the LoRA
adapters far off manifold. The 9B model regresses vs its own 0-shot
baseline on mortality (0.505 vs 0.636) and readmission (0.500 vs 0.577)
while 4B stays at or above its own 0-shot numbers.

## 5. Fixes proposed for iter 1 (not yet implemented)

Priority order, smallest first:

1. **Reward shaping to fight class collapse** (`rewards.py`):
   - Binary: reward = `2 * (TP_rate + TN_rate)/2 - 1` (balanced accuracy),
     OR give +1 only when matching the minority class correctly.
   - LOS: keep the ordinal partial credit, but add −0.2 if the prediction
     equals the modal bucket across the group (anti-mode-collapse signal
     computed at advantage time).
   - Drugrec/phenotyping: penalize empty predictions (current F1 reward
     gives 0 on empty, which is fine, but the length penalty only kicks
     in at `>2×|gold|`; tighten to `>1.5×`).
   - Optional: halve `FORMAT_BONUS` to 0.025 once parse_rate > 0.8.

2. **9B learning rate** (`grpo_single.py`, `grpo_multi.py`):
   - Cut default `--learning-rate 1e-5` → `5e-6` for 4B, `1e-6` for 9B.
   - Or reduce `--lora-r 32` → `8` for 9B.

3. **Bandit** — already fixed in 293f1f3, but verify with a short
   smoke run before committing iter 1 GPU time.

4. **KL clip tightening** (`grpo_core.py::kl_k3`): current `.clamp(-20, 20)`
   lets `exp(r)` run up to e^20. Drop to `.clamp(-5, 5)` so KL penalty
   actually bites when the policy drifts.

5. **Eval improvement for binary tasks**: the Track A 0-shot path uses
   first-token `{yes,no}` softmax to get probabilities (needed for AUROC).
   The GRPO eval path uses hard generation → 0/0.5/1 probs → AUROC is
   noisy. For iter 1 B/C eval, reuse the Track A probability path.

## 6. What transfers from the GRU study

- **A2C-GAE beat REINFORCE** on the GRU side. GRPO is closer to REINFORCE
  than A2C; if iter 1 with the fixes above still under-performs, consider
  swapping in a per-step baseline (running-mean reward) to reduce variance
  beyond group-norm.
- **Code masking 10%** was effective regularization. Could translate to
  "randomly drop 10% of codes in the prompt" — not tried here.
- **sqrt inverse-freq class weights** helped LOS on GRU. Analogous move on
  the LLM side would be to sample LOS prompts inversely proportional to
  their true-bucket frequency at training time.

## 7. Open questions the data doesn't yet answer

- **Is mode collapse a 4B+9B issue or LLM-wide?** Would need a 32B run to
  disentangle capacity from reward design.
- **Does the format bonus help at all?** Could run a single ablation
  with `FORMAT_BONUS = 0` to see if parse_rate still climbs (probably yes,
  via group-norm on correctness alone).
- **Would supervised warm-up** (1 epoch SFT on train labels) avoid the
  collapse trap entirely? Common trick in GRPO pipelines, not tried here.

## 8. What's committed on master right now

```
ed8b0f0  qwen Track C iter 0: 4b + 9b multi-task baseline
293f1f3  qwen: fix UCBTaskBandit only exploring on eval, not on select
d667674  qwen Track B iter 0: sweep complete (10 runs, after 2 bug fixes)
139f22c  qwen: phenotyping primary = f1_samples
19a684d  qwen: fix eval skipped when all groups degenerate
6ff402b  qwen Track A: sweep complete (20 runs)
170a954  qwen: fix duplicate step kwarg in now_log
b313e06  qwen: rename torch_dtype → dtype (transformers 5.5.4)
```

All adapter binaries (~170–350 MB each) are NOT committed — only
`metrics.json`, `run_meta.json`, `train_log.jsonl` are tracked per run.
