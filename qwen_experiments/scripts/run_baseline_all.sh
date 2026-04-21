#!/bin/bash
# Dispatch all Track-A runs for a given model across 5 tasks × {0, 5} shots.
#
# Usage: ./run_baseline_all.sh qwen35-9b [n_eval]

set -euo pipefail

MODEL_TAG="${1:?model tag}"
N_EVAL="${2:-500}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for TASK in mimic4_mortality mimic4_readmission mimic4_los mimic4_phenotyping mimic4_drugrec; do
  for SHOTS in 0 5; do
    echo "sbatch ${TASK} ${SHOTS}"
    sbatch "${SCRIPT_DIR}/run_baseline.sbatch" "${MODEL_TAG}" "${TASK}" "${SHOTS}" --n-eval "${N_EVAL}"
  done
done
