#!/bin/bash
# End-to-end launcher for all three tracks. Idempotent — each sbatch
# creates its own timestamped run dir; re-running this just adds more.
#
# Usage:
#   ./launch_all.sh A           # Track A sweep (20 jobs)
#   ./launch_all.sh B           # Track B iter 0 for all 5 tasks × both models
#   ./launch_all.sh C           # Track C iter 0 for both models
#   ./launch_all.sh all         # all three (A then B then C)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PHASE="${1:-all}"

TASKS=(mimic4_mortality mimic4_readmission mimic4_los mimic4_phenotyping mimic4_drugrec)
MODELS=(qwen35-4b qwen35-9b)

launch_A() {
  echo "=== Track A sweep: ${#MODELS[@]} models × ${#TASKS[@]} tasks × 2 shot counts ==="
  for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
      for shots in 0 5; do
        sbatch "${SCRIPT_DIR}/run_baseline.sbatch" "$model" "$task" "$shots" --n-eval 500
      done
    done
  done
}

launch_B() {
  echo "=== Track B iter 0: ${#MODELS[@]} models × ${#TASKS[@]} tasks ==="
  for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
      sbatch "${SCRIPT_DIR}/run_single_grpo.sbatch" "$model" "$task" 0 --extra-tag baseline
    done
  done
}

launch_C() {
  echo "=== Track C iter 0: ${#MODELS[@]} models × multi-task ==="
  for model in "${MODELS[@]}"; do
    sbatch "${SCRIPT_DIR}/run_multi_grpo.sbatch" "$model" 0 --extra-tag baseline
  done
}

case "$PHASE" in
  A) launch_A ;;
  B) launch_B ;;
  C) launch_C ;;
  all) launch_A; launch_B; launch_C ;;
  *) echo "Unknown phase: $PHASE"; exit 1 ;;
esac

echo ""
echo "Submitted. Watch:  squeue -u \$USER"
echo "Results:           tail -f qwen_experiments/leaderboard.tsv"
