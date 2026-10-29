# Runtime Validation Report

**Generated:** 2026-10-29T10:36:35+05:30  
**Target:** `D:\MIND-AID`

## Summary

| Step | Status | Exit Code | Details |
|---|---|---:|---|
| Environment | PASS | 0 | Python 3.13.3; required research packages installed; Node.js 22.13.0 available; Docker unavailable and not required for this workflow |
| Deduplicated split | PASS | 0 | Regenerated `data/splits/train.csv` and `data/splits/external_validation.csv` with random state 42 |
| Strict external validation | PASS | 0 | 715 valid external rows; 0 duplicate test rows; 0 train–test overlap |
| Strict model training | PASS | 0 | 2,835 input rows; 19 exact normalized duplicate rows removed before fitting |
| External evaluation | PASS | 0 | Strict external metrics and robustness report regenerated |
| Explainability audit | PASS | 0 | 715 samples audited |
| Resource benchmark | PASS | 0 | Three seeds: 13, 42, 73; transformer resource comparison regenerated |
| Visual report | PASS | 0 | HTML report and charts regenerated |
| API health | PASS | 0 | Existing service at `127.0.0.1:8000/health` returned HTTP 200 |
| API text analysis | PASS | 0 | `/analyze` returned a valid text-only screening response |
| Second API startup | BLOCKED | 1 | Port 8000 was already in use; existing service was left running |

## Latest Strict External Results

- Test samples: 715
- Accuracy: 0.7231
- Balanced accuracy: 0.7191
- Weighted precision: 0.7335
- Weighted recall: 0.7231
- Weighted F1: 0.7183
- ROC-AUC: 0.8257
- Brier score: 0.1794
- Accuracy 95% bootstrap CI: 0.6881–0.7552
- F1 95% bootstrap CI: 0.6831–0.7512

## Explainability Audit

- Mean top-keyword Jaccard stability: 0.8338
- Mean probability change after masking: 0.1520
- Prediction flip rate after masking: 23.36%
- Misclassified examples: 198

## Robustness

- Punctuation removal: 0.0% prediction flips
- Spelling noise: 2.0% prediction flips
- Slang/abbreviation substitutions: 2.9% prediction flips
- Word dropout: 9.8% prediction flips
- Stopword removal with negations preserved: 27.6% prediction flips
- 20-token truncation: 31.3% prediction flips; weighted F1 0.6673

## Generated Artifacts

- `research/reports/research_evaluation_dreaddit_adapted_strict.json`
- `research/reports/external_dataset_validation_dreaddit_adapt_strict.json`
- `research/reports/explainability_audit_dreaddit_adapted_strict.json`
- `research/reports/model_benchmark_dreaddit_adapted_strict.json`
- `research/reports/metrics_dreaddit_adapt_strict.json`
- `research/reports/metrics_internal_deduplicated.json`
- `research/reports/dreaddit_strict/visual_report.html`
- `research/charts/ablation_comparison_dreaddit_adapted_strict.png`
- `research/charts/generalization_comparison.png`
- `research/charts/robustness_perturbation_chart.png`
- `research/charts/calibration_curve.png`

## Interpretation

The strict external evaluation is the primary evidence for generalization claims. Internal holdout results are diagnostic only. The API response is a research screening estimate and is not a clinical diagnosis.
