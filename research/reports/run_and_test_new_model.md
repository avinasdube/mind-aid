# How to Run and Test the Upgraded MIND-AID Model

## What changed in the implementation

The upgraded pipeline now includes:
1. The existing TF-IDF + calibrated LinearSVC text model.
2. A transformer sentence-embedding baseline using `sentence-transformers/all-MiniLM-L6-v2` and LogisticRegression.
3. Bootstrap 95% confidence intervals for headline metrics.
4. Updated reports and charts saved in `research/reports` and `research/charts`.

## How to run the models

All commands below assume you are in the repository root. The strict Dreaddit CSV files live under `data/external/`, not in the root folder.

### 1) Activate the environment

```bash
source .venv/Scripts/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Train the strict model again

```bash
python app.py train --data data/external/dreaddit_adapt_train_strict.csv --model models/text_model_dreaddit_strict.pkl --vectorizer models/vectorizer_dreaddit_strict.pkl --metrics research/reports/metrics_dreaddit_adapt_strict.json
```

### 4) Validate the strict external split

```bash
python app.py validate-external --external-data data/external/dreaddit_adapt_test.csv --training-data data/external/dreaddit_adapt_train_strict.csv --output research/reports/external_dataset_validation_dreaddit_adapt_strict.json
```

### 5) Run external evaluation

```bash
python app.py evaluate --data data/external/dreaddit_adapt_train_strict.csv --external-data data/external/dreaddit_adapt_test.csv --model models/text_model_dreaddit_strict.pkl --vectorizer models/vectorizer_dreaddit_strict.pkl --output research/reports/research_evaluation_dreaddit_adapted_strict.json --chart research/charts/ablation_comparison_dreaddit_adapted_strict.png
```

### 6) Regenerate visual report

```bash
python app.py visual-report --metrics research/reports/metrics_dreaddit_adapt_strict.json --evaluation research/reports/research_evaluation_dreaddit_adapted_strict.json --charts-dir research/charts --report-dir research/reports/dreaddit_strict
```

## How to test the upgrade

### A. Syntax check

```bash
python -m py_compile src/research_eval.py
```

### B. Confirm transformer baseline appears in the JSON report

Open `research/reports/research_evaluation_dreaddit_adapted_strict.json` and confirm these blocks exist:
1. `transformer_sentence_embedding_baseline`
2. `confidence_intervals`

### C. Confirm the strict external metrics

Expected upgraded strict external results are approximately:
1. Text-only accuracy: 0.7217
2. Text-only ROC-AUC: 0.8257
3. Transformer baseline accuracy: 0.7524
4. Transformer baseline ROC-AUC: 0.8474

### D. Confirm the report artifacts

Check these files:
1. `research/reports/dreaddit_strict/visual_report.html`
2. `research/reports/dreaddit_strict/analysis_summary.md`
3. `research/charts/generalization_comparison.png`
4. `research/charts/robustness_perturbation_chart.png`

## How to explain the impact

1. The transformer baseline gives a stronger modern comparison than TF-IDF alone.
2. Strict adaptation still performs best overall on the external benchmark.
3. Bootstrap confidence intervals make the results more defensible in a paper review.
4. The upgraded report is now more credible because it includes both point estimates and uncertainty.
