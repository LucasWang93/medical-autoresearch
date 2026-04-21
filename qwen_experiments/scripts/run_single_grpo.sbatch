#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --account=pi_yz875
#SBATCH --job-name=qwen-B
#SBATCH --output=/nfs/roberts/project/pi_yz875/sw2572/codes/auto-research-health/medical-autoresearch/qwen_experiments/runs/_meta/single_grpo_%j.log

# Usage:
#   sbatch run_single_grpo.sbatch <model_tag> <task> <iter_idx> [extra --args]
set -euo pipefail

MODEL_TAG="${1:?model tag (qwen35-4b|qwen35-9b)}"
TASK="${2:?task}"
ITER_IDX="${3:-0}"
shift 3 || true

MA_ROOT=/nfs/roberts/project/pi_yz875/sw2572/codes/auto-research-health/medical-autoresearch
cd "${MA_ROOT}"
mkdir -p "${MA_ROOT}/qwen_experiments/runs/_meta"

source /nfs/roberts/project/pi_yz875/sw2572/gpus/env_qwen35.sh

echo "============================================"
echo "Track B GRPO    Job: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)   GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Model: ${MODEL_TAG}  Task: ${TASK}  Iter: ${ITER_IDX}"
echo "Extra args: $*"
echo "============================================"
nvidia-smi -L || true

python -m qwen_experiments.src.grpo_single \
  --model-tag "${MODEL_TAG}" \
  --task "${TASK}" \
  --iter-idx "${ITER_IDX}" \
  "$@"
