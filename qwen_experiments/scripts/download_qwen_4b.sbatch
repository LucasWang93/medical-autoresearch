#!/bin/bash
#SBATCH --partition=day
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --account=pi_yz875
#SBATCH --job-name=qwen35-4b-dl
#SBATCH --output=/nfs/roberts/project/pi_yz875/sw2572/codes/auto-research-health/medical-autoresearch/qwen_experiments/runs/_meta/download_qwen35_4b_%j.log

set -euo pipefail

mkdir -p /nfs/roberts/project/pi_yz875/sw2572/codes/auto-research-health/medical-autoresearch/qwen_experiments/runs/_meta

source /nfs/roberts/project/pi_yz875/sw2572/gpus/env_qwen35.sh

TARGET=/nfs/roberts/project/pi_yz875/sw2572/models/Qwen3.5-4B
echo "Downloading Qwen3.5-4B to ${TARGET}"
echo "Node: $(hostname)  Job: ${SLURM_JOB_ID:-local}  Time: $(date)"

hf download Qwen/Qwen3.5-4B --local-dir "${TARGET}"

echo "Download complete. Final size:"
du -sh "${TARGET}"
