#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --account=pi_yz875
#SBATCH --job-name=qwen-A
#SBATCH --output=/nfs/roberts/project/pi_yz875/sw2572/codes/auto-research-health/medical-autoresearch/qwen_experiments/runs/_meta/baseline_%j.log

# Usage:
#   sbatch run_baseline.sbatch <model_tag> <task> <shots> [--n-eval N]
#
# model_tag: qwen35-4b | qwen35-9b
# task:      mimic4_{mortality,readmission,los,phenotyping,drugrec}
# shots:     0 or 5
#
# Example:
#   sbatch run_baseline.sbatch qwen35-9b mimic4_mortality 0

set -euo pipefail

MODEL_TAG="${1:?model tag (qwen35-4b|qwen35-9b)}"
TASK="${2:?task name (mimic4_*)}"
SHOTS="${3:-0}"
shift 3 || true

MA_ROOT=/nfs/roberts/project/pi_yz875/sw2572/codes/auto-research-health/medical-autoresearch
QEXP_ROOT="${MA_ROOT}/qwen_experiments"

mkdir -p "${QEXP_ROOT}/runs/_meta"
cd "${MA_ROOT}"

source /nfs/roberts/project/pi_yz875/sw2572/gpus/env_qwen35.sh

echo "============================================"
echo "Track A baseline   Job: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)   GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Model: ${MODEL_TAG}  Task: ${TASK}  Shots: ${SHOTS}"
echo "============================================"
nvidia-smi -L || true

cd "${MA_ROOT}"
python -m qwen_experiments.src.baseline_fewshot \
  --model-tag "${MODEL_TAG}" \
  --task "${TASK}" \
  --shots "${SHOTS}" \
  "$@"
