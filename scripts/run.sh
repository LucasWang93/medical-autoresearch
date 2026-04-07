#!/bin/bash
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --account=pi_yz875
#SBATCH --job-name=med-ar
#SBATCH --output=results/run_%j.log

# Medical AutoResearch — single experiment launcher
#
# Usage:
#   sbatch scripts/run.sh                     # default: gpu_devel, 10 min
#   sbatch --partition=gpu --time=00:30:00 scripts/run.sh   # override
#   sbatch scripts/run.sh --task mortality_prediction       # pass args to train.py
#
# Partitions:
#   gpu_devel   — quick test (5-10 min)
#   gpu         — standard experiments (30 min)
#   gpu_rtx6000 — RTX 6000 nodes
#   gpu_h200    — H200 for large models

set -euo pipefail

# Use hardcoded BASEDIR since Slurm may change $0
BASEDIR="/nfs/roberts/project/pi_yz875/sw2572/codes/auto-research-health/medical-autoresearch"
cd "${BASEDIR}"

echo "=========================================="
echo "Medical AutoResearch — Experiment Run"
echo "=========================================="
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION:-local}"
echo "GPU:       ${CUDA_VISIBLE_DEVICES:-none}"
echo "Time:      $(date)"
echo "Dir:       $(pwd)"
echo "=========================================="

# Load system modules (PyTorch with matching CUDA)
module purge 2>/dev/null || true
module load StdEnv 2>/dev/null || true
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0 2>/dev/null || true

# Install missing deps into user site
pip install --user scikit-learn datasets pandas 2>/dev/null || true

# GPU diagnostics
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    print(f'VRAM: {vram / 1024**3:.1f} GB')
else:
    print('WARNING: No CUDA — training on CPU (will be slow)')
"

echo "=========================================="
echo "Starting training..."
echo "=========================================="

# Run training (pass through any extra args)
python train.py "$@"

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
