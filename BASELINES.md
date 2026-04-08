# Baseline Results

All baselines use the same model architecture: **REINFORCE-based Clinical RL** with 120s training budget on a single GPU.

## Model Architecture

```
Input (ICD codes per visit)
    → CodeEmbedding (sum-pooling over codes, dim=128)
    → GRU (hidden=128, 1 layer)
    → PolicyAgent (RL history selection, 10 actions)
    → Fusion (attended history + current visit)
    → Task Head (linear → binary/multiclass/multilabel output)
```

- **CodeEmbedding**: `nn.Embedding` + sum-pooling over codes per visit + dropout (0.3)
- **GRU**: 1-layer, hidden_dim=128, processes visit sequence
- **PolicyAgent**: REINFORCE-based attention over visit history. Selects which past visits to attend to (10 discrete actions). Uses entropy regularization (0.01) and learned baseline for variance reduction.
- **Task Head**: Linear projection from fused representation to output logits
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5), gradient clipping (max_norm=1.0)
- **Total parameters**: ~289K (binary/multiclass) or ~440K (multilabel with drug history)

## Training Setup

- **Budget**: 120 seconds wall-clock training
- **Epochs**: Typically 1 epoch on MIMIC-IV (322K samples), 2-5 epochs on SUPPORT2 (9K samples)
- **Batch size**: 32
- **Loss**: task_loss (BCE/CE) + RL policy gradient loss (REINFORCE with baseline)
- **Evaluation**: On held-out test set (10% of data), patient-level random split

## Results by Data Source

### MIMIC-IV 3.1 (223,452 patients, 322,576 samples)

Real longitudinal EHR data from Beth Israel Deaconess Medical Center. Each patient has >= 2 hospital admissions. Features are ICD diagnosis codes (top 500) and ICD procedure codes (top 200) accumulated across visits.

| Task | Type | Metric | Baseline | Job ID | Notes |
|------|------|--------|----------|--------|-------|
| `mimic4_mortality` | binary | AUROC | **0.961** | 7601425 | In-hospital mortality. High AUROC partly because discharge diagnoses reflect the full hospital course including terminal events. |
| `mimic4_readmission` | binary | AUROC | **0.669** | 7601440 | 30-day readmission. Inherently hard to predict — many readmissions are driven by social factors not captured in ICD codes. |
| `mimic4_los` | 4-class | F1_macro | **0.481** | 7601441 | Length-of-stay buckets: <3d, 3-7d, 7-14d, >14d. Well above random (0.25). |
| `mimic4_drugrec` | multilabel | Jaccard | **0.166** | 7601442 | Drug recommendation across 300 drugs. Large label space makes this the hardest task. |

### SUPPORT2 (9,105 patients, tabular cross-sectional)

Public dataset of seriously ill patients from 5 US medical centers (HuggingFace: `jarrydmartinx/support2`). Features are discretized vitals, labs, severity scores, and demographics — treated as "pseudo-visits" for our GRU pipeline.

| Task | Type | Metric | Baseline | Job ID | Notes |
|------|------|--------|----------|--------|-------|
| `support2_mortality` | binary | AUROC | **0.916** | 7574470 | In-hospital mortality. Strong signal from severity scores (APS, SCOMA). |
| `support2_dzclass` | 4-class | F1_macro | **0.779** | 7592232 | Disease classification (ARF/MOSF, COPD/CHF/Cirrhosis, Cancer, Coma). Leak-fixed: dzgroup/dzclass excluded from features. |
| `support2_survival` | binary | AUROC | **0.936** | 7592233 | 2-month survival prediction. Leak-fixed: sps/surv/prg scores excluded from features. |

### Synthetic EHR (2,000 generated patients)

Self-contained synthetic data with disease archetypes and condition-to-drug mappings. Useful for development and debugging without real data access.

| Task | Type | Metric | Baseline | Job ID | Notes |
|------|------|--------|----------|--------|-------|
| `drug_recommendation` | multilabel | Jaccard | **0.654** | 7439080 | 60s budget, 150-drug label space. Synthetic archetypes provide learnable structure. |

## Improvement Opportunities

Ranked by potential impact:

1. **mimic4_drugrec (0.166)** — Largest gap. Try: transformer attention over visits, longer training (600s+), drug co-occurrence priors, separate drug history encoder.

2. **mimic4_los (0.481)** — 4-class with room to grow. Try: ordinal regression (LOS is ordered), demographic features from patients table, admission type features.

3. **mimic4_readmission (0.669)** — Hard task but addressable. Try: time-aware visit encoding (gap between admissions), more training time, ensemble approaches.

4. **support2_dzclass (0.779)** — Try: feature interaction layers, ensemble of feature groups, class-balanced sampling.

5. **mimic4_mortality (0.961)** — Already high. Marginal gains possible with: prospective-only diagnosis codes (exclude current visit), calibration tuning.

## Reproducing Baselines

```bash
# MIMIC-IV tasks (requires local MIMIC-IV data)
for task in mimic4_mortality mimic4_readmission mimic4_los mimic4_drugrec; do
  sbatch --partition=gpu --time=00:20:00 scripts/run.sh --task $task --time-budget 120
done

# SUPPORT2 tasks (auto-downloads from HuggingFace)
for task in support2_mortality support2_dzclass support2_survival; do
  sbatch --partition=gpu_devel --time=00:10:00 scripts/run.sh --task $task --time-budget 120
done

# Synthetic task
sbatch scripts/run.sh --task drug_recommendation --time-budget 60 --n-patients 2000
```

## Data Leakage Notes

- **SUPPORT2 dzclass**: Initially scored 1.000 F1_macro because `dzgroup` and `dzclass` were in the input features (the label is derived from these). Fixed by excluding them via `_SUPPORT2_EXCLUDE`.
- **SUPPORT2 survival**: Initially scored 0.975 AUROC because `sps` (survival probability score) was in the input. Fixed by excluding `sps`, `surv2m`, `surv6m`, `prg2m`, `prg6m`.
- **MIMIC-IV mortality**: AUROC 0.961 uses discharge diagnoses which can include terminal events. This is the standard benchmark setup but should be noted when comparing to prospective models.
