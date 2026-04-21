#!/bin/bash
#SBATCH --partition=day
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --account=pi_yz875
#SBATCH --job-name=qwen-datacache
#SBATCH --output=/nfs/roberts/project/pi_yz875/sw2572/codes/auto-research-health/medical-autoresearch/qwen_experiments/runs/_meta/datacache_%j.log

# Pre-build the TextSample JSONL caches for all 5 MIMIC-IV tasks.
# Without this, each Track-A/B/C GPU job would redo the CSV read.

set -euo pipefail

MA_ROOT=/nfs/roberts/project/pi_yz875/sw2572/codes/auto-research-health/medical-autoresearch
cd "${MA_ROOT}"

source /nfs/roberts/project/pi_yz875/sw2572/gpus/env_qwen35.sh

for TASK in mimic4_mortality mimic4_readmission mimic4_los mimic4_phenotyping mimic4_drugrec; do
  echo "=== Building cache for ${TASK} ==="
  python -c "
import sys; sys.path.insert(0, '.')
from qwen_experiments.src.data import build_text_dataset
s = build_text_dataset('${TASK}', force=False)
print(f'[${TASK}] samples: {len(s)}')
"
done

echo "Caches built:"
ls -la "${MA_ROOT}/qwen_experiments/.cache/data/"
