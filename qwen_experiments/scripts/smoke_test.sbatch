#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --account=pi_yz875
#SBATCH --job-name=qwen-smoke
#SBATCH --output=/nfs/roberts/project/pi_yz875/sw2572/codes/auto-research-health/medical-autoresearch/qwen_experiments/runs/_meta/smoke_%j.log

# Lightweight end-to-end smoke test:
#   * Track A baseline eval on 4B, 8 samples of mortality, 0 shots
#   * Track B one GRPO step on 4B, 2 prompts × group 4

set -euo pipefail

MA_ROOT=/nfs/roberts/project/pi_yz875/sw2572/codes/auto-research-health/medical-autoresearch
cd "${MA_ROOT}"
mkdir -p "${MA_ROOT}/qwen_experiments/runs/_meta"

source /nfs/roberts/project/pi_yz875/sw2572/gpus/env_qwen35.sh

echo "====== Smoke 1/2: Track A (4B, mortality, 0-shot, n=8) ======"
python -m qwen_experiments.src.baseline_fewshot \
  --model-tag qwen35-4b --task mimic4_mortality --shots 0 --n-eval 8

echo ""
echo "====== Smoke 2/2: Track B (4B, mortality, 3 steps, G=4) ======"
python -m qwen_experiments.src.grpo_single \
  --model-tag qwen35-4b --task mimic4_mortality \
  --iter-idx 0 --extra-tag smoke \
  --group-size 4 --total-steps 3 --eval-every 3 --eval-n 8 \
  --prompts-per-step 1 --max-new-tokens 16
