#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --account=pi_yz875
#SBATCH --job-name=qwen-C
#SBATCH --output=/nfs/roberts/project/pi_yz875/sw2572/codes/auto-research-health/medical-autoresearch/qwen_experiments/runs/_meta/multi_grpo_%j.log

# Usage:
#   sbatch run_multi_grpo.sbatch <model_tag> <iter_idx> [extra --args]
set -euo pipefail

MODEL_TAG="${1:?model tag}"
ITER_IDX="${2:-0}"
shift 2 || true

MA_ROOT=/nfs/roberts/project/pi_yz875/sw2572/codes/auto-research-health/medical-autoresearch
cd "${MA_ROOT}"
mkdir -p "${MA_ROOT}/qwen_experiments/runs/_meta"

source /nfs/roberts/project/pi_yz875/sw2572/gpus/env_qwen35.sh

echo "============================================"
echo "Track C multi-task GRPO    Job: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)   GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Model: ${MODEL_TAG}  Iter: ${ITER_IDX}"
echo "Extra args: $*"
echo "============================================"
nvidia-smi -L || true

python -m qwen_experiments.src.grpo_multi \
  --model-tag "${MODEL_TAG}" \
  --iter-idx "${ITER_IDX}" \
  "$@"
