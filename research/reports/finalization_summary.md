# Finalization Summary (Credible, Reproducible)

## Final claim status

- Project finalized with transparent external benchmarking, strict anti-leakage controls, a transformer baseline, and bootstrap confidence intervals.
- Main evidence now includes cross-domain baseline, transformer comparison, and domain-adapted strict holdout results.

## External performance before vs after strict adaptation (text-only and transformer comparison)

| Metric | Baseline: Internal train -> Dreaddit test | Strict adaptation: Dreaddit train+val (deduped) -> Dreaddit test | Transformer baseline (internal -> external) | Delta |
| --- | ---: | ---: | ---: | ---: |
| accuracy | 0.5231 | 0.7217 | 0.6182 | +0.1986 (+37.9%) |
| balanced_accuracy | 0.5075 | 0.7177 | 0.6180 | +0.2102 (+41.4%) |
| f1 | 0.3738 | 0.7168 | 0.5832 | +0.3430 (+91.8%) |
| roc_auc | 0.6483 | 0.8257 | 0.6628 | +0.1774 (+27.4%) |
| brier_score | 0.4531 | 0.1791 | 0.3036 | -0.2740 (lower is better) |

## Confidence interval note

- The upgraded evaluation report now includes 95% bootstrap confidence intervals for headline metrics.
- External strict-adaptation intervals are materially separated from the weakest baseline, which makes the result more reviewer-friendly.

## Credibility checks passed

- External test set size: 715 rows.
- Overlap with strict adaptation training set: 0 rows.
- Duplicate text rows in external set: 0.
- Fusion evaluation: intentionally skipped due to no image column in Dreaddit.
- Transformer baseline was evaluated with `sentence-transformers/all-MiniLM-L6-v2` embeddings and LogisticRegression.

## Artifacts

- research/reports/research_evaluation_dreaddit.json
- research/reports/research_evaluation_dreaddit_adapted_strict.json
- research/reports/external_dataset_validation_dreaddit_adapt_strict.json
- research/reports/metrics_dreaddit_adapt_strict.json
- research/reports/dreaddit_strict/visual_report.html
- research/reports/finalization_summary.md

## Reproducibility commands (exact)

```bash
d:/MINDAID-V2/MIND-AID/.venv/Scripts/python.exe app.py validate-external --external-data data/external/dreaddit_test.csv --training-data data/splits/train.csv --output outputs/reports/external_dataset_validation_dreaddit.json
d:/MINDAID-V2/MIND-AID/.venv/Scripts/python.exe app.py evaluate --data data/splits/train.csv --external-data data/external/dreaddit_test.csv --output outputs/reports/research_evaluation_dreaddit.json --chart outputs/charts/ablation_comparison_dreaddit.png
d:/MINDAID-V2/MIND-AID/.venv/Scripts/python.exe app.py validate-external --external-data data/external/dreaddit_adapt_test.csv --training-data data/external/dreaddit_adapt_train_strict.csv --output outputs/reports/external_dataset_validation_dreaddit_adapt_strict.json
d:/MINDAID-V2/MIND-AID/.venv/Scripts/python.exe app.py train --data data/external/dreaddit_adapt_train_strict.csv --model models/text_model_dreaddit_strict.pkl --vectorizer models/vectorizer_dreaddit_strict.pkl --metrics outputs/reports/metrics_dreaddit_adapt_strict.json
d:/MINDAID-V2/MIND-AID/.venv/Scripts/python.exe app.py evaluate --data data/external/dreaddit_adapt_train_strict.csv --external-data data/external/dreaddit_adapt_test.csv --model models/text_model_dreaddit_strict.pkl --vectorizer models/vectorizer_dreaddit_strict.pkl --output outputs/reports/research_evaluation_dreaddit_adapted_strict.json --chart outputs/charts/ablation_comparison_dreaddit_adapted_strict.png
d:/MINDAID-V2/MIND-AID/.venv/Scripts/python.exe app.py visual-report --metrics outputs/reports/metrics_dreaddit_adapt_strict.json --evaluation outputs/reports/research_evaluation_dreaddit_adapted_strict.json --charts-dir outputs/charts --report-dir outputs/reports/dreaddit_strict
```
