# medical-autoresearch

Autonomous clinical prediction research loop built from the `autoresearch`
single-file pattern and `PyHealth` task infrastructure.

## What matters

- `prepare.py`: fixed infrastructure for task registration, synthetic EHR data,
  PyHealth loading, evaluation, and standardized result printing.
- `train.py`: the research surface. Model, RL policy, reward usage, and training
  loop all live here.
- `program.md`: instructions for autonomous experiment iteration.
- `scripts/run.sh`: Slurm launcher for cluster experiments.

## Available tasks

- `drug_recommendation`: multilabel recommendation with DDI-aware reward.
- `mortality_prediction`: binary prediction from visit history.
- `readmission_prediction`: binary 30-day readmission prediction.
- `length_of_stay`: multiclass length-of-stay bucket prediction.

## Quick start

```bash
python train.py --time-budget 60 --n-patients 500
python train.py --task mortality_prediction --time-budget 60 --n-patients 500
```

To use real MIMIC data through the local `PyHealth` checkout:

```bash
python train.py --use-pyhealth --data-root /path/to/mimiciii
```

If PyHealth loading fails, the code falls back to synthetic data.

## Tests

Run the local regression and smoke tests with:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The tests cover:

- reward and metric regressions
- PyHealth data normalization into the internal tensor format
- end-to-end `train.py` smoke execution
