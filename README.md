# MIND-AID

MIND-AID is a multimodal depression-risk screening research system that combines text modeling with optional facial-emotion cues for decision-support prototyping.

## Overview

The project provides an end-to-end workflow for:

- text preprocessing and model training
- risk scoring and lightweight explainability
- optional confidence-aware text+face fusion
- external validation and anti-leakage checks
- chart and report generation for research communication

MIND-AID is intended for research and decision support. It is not a clinical diagnostic tool.

## Research Status

The current implementation includes:

- strict external benchmarking on Dreaddit
- overlap checks before external evaluation
- a legacy sentence-embedding baseline for comparison
- a fine-tuned DistilBERT model with token-level SHAP explanations
- bootstrap 95% confidence intervals for headline external metrics

Text-only external benchmark (Dreaddit test):

| Metric | Baseline: internal train -> Dreaddit test | Strict adaptation: Dreaddit train+val (deduped) -> Dreaddit test |
| --- | ---: | ---: |
| accuracy | 0.5231 | 0.7231 |
| balanced_accuracy | 0.5075 | 0.7191 |
| f1 | 0.3738 | 0.7183 |
| roc_auc | 0.6479 | 0.8257 |
| brier_score | 0.4531 | 0.1794 |

Transformer baseline external benchmark (same test split, different training setups):

| Metric | Transformer baseline: internal train -> Dreaddit test | Transformer baseline: strict Dreaddit train+val -> Dreaddit test |
| --- | ---: | ---: |
| accuracy | 0.6182 | 0.7524 |
| balanced_accuracy | 0.6090 | 0.7519 |
| f1 | 0.5832 | 0.7523 |
| roc_auc | 0.6628 | 0.8474 |
| brier_score | 0.3036 | 0.1599 |

Notes:

- strict training set: 2,816 rows after normalization and removal of 19 duplicate rows
- strict external test size: 715 samples
- strict train vs strict test overlap: 0 rows
- fusion was skipped for Dreaddit external claims because no image column is available
- the table above reports the strict TF-IDF external evaluation
- the legacy sentence-embedding baseline is retained for research comparison
- the production prediction path defaults to calibrated TF-IDF + LinearSVC; fine-tuned DistilBERT is explicitly selectable

Latest fine-tuned DistilBERT external evaluation on the same 715-row test set:

| Metric | Fine-tuned DistilBERT |
| --- | ---: |
| accuracy | 0.7916 |
| weighted F1 | 0.7914 |
| ROC-AUC | 0.8863 |

The DistilBERT configuration used `distilbert-base-uncased`, two epochs, batch
size 8, maximum sequence length 256, learning rate `2e-5`, seed 42, and CPU
execution. These strict results are separate from the original accepted-paper
results and must not be silently substituted for them.

## Core Features

- TF-IDF (uni/bi-gram) + calibrated LinearSVC text pipeline
- optional DeepFace-based facial emotion analysis
- confidence-aware multimodal fusion strategy
- keyword-level explanation output
- external dataset validator (schema, overlap, image readiness)
- robust evaluation reporting with bootstrap confidence intervals
- React frontend + FastAPI backend interface

## Technology Stack

- Python 3.10+
- FastAPI
- scikit-learn
- pandas, numpy
- sentence-transformers
- transformers, torch
- SHAP
- DeepFace, OpenCV
- matplotlib, seaborn
- React (Vite)

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quickstart

Train the default text model:

```bash
python app.py train --data data/processed/depression_dataset.csv --model models/text_model.pkl --vectorizer models/vectorizer.pkl --metrics research/reports/metrics_train.json
```

Start backend API:

```bash
python -m uvicorn api_server:app --reload --host 127.0.0.1 --port 8000
```

Start frontend:

```bash
cd frontend
npm install
npm run dev
```

Frontend URL: <http://127.0.0.1:5173>

The frontend defaults to the strict TF-IDF model and provides a model selector
for the fine-tuned DistilBERT model. DistilBERT requests run token-level SHAP
explanations and therefore take longer. To select a different API port for the
frontend, set `VITE_API_BASE_URL` before starting Vite:

```powershell
$env:VITE_API_BASE_URL = "http://127.0.0.1:8001"
cd frontend
npm.cmd run dev -- --host 127.0.0.1 --port 5174
```

The API accepts `model_name=tfidf` or `model_name=distilbert` as form data on
`POST /analyze`. The response identifies the selected model and explanation
method in `model_label` and `explanation_method`.

## Dataset Handling

If you need a strict holdout from the packaged dataset:

```bash
python app.py split-dataset --input data/processed/depression_dataset.csv --train-output data/splits/train.csv --validation-output data/splits/external_validation.csv --test-size 0.2
```

Expected schema is `text,label` for text-only evaluation. Optional fusion-related columns include `face_score`, `face_confidence`, and `image_path`.

Before splitting or fitting, the workflow normalizes text with the project
preprocessor and removes exact duplicate normalized texts. The split and
training metrics record the number of removed rows. Near-duplicate posts are
not removed using an arbitrary similarity threshold; external validation
reports exact overlap checks, and near-duplicate similarity audits should be
reported separately when available.

## Research Evaluation Workflow

Use `external_validation` as the primary evidence block for paper claims. Treat `internal_diagnostic_in_sample` as diagnostic only.

The reproducibility commands also include:

```bash
python app.py explainability-audit --data data/external/dreaddit_adapt_test.csv --model models/text_model_dreaddit_strict.pkl --vectorizer models/vectorizer_dreaddit_strict.pkl --output research/reports/explainability_audit_dreaddit_adapted_strict.json

python app.py benchmark-model --data data/external/dreaddit_adapt_train_strict.csv --external-data data/external/dreaddit_adapt_test.csv --output research/reports/model_benchmark_dreaddit_adapted_strict.json
```

The default API explanation path remains lightweight coefficient-based TF-IDF
attribution. A separate reproducible SHAP audit is available for the trained
linear model; it is not invoked on every API request.

Reproducible SHAP attributions for the shipped TF-IDF model can be generated with:

```bash
python app.py shap-audit --data data/external/dreaddit_adapt_test.csv --model models/text_model_dreaddit_strict.pkl --vectorizer models/vectorizer_dreaddit_strict.pkl --output research/reports/shap_audit_dreaddit_adapted_strict.json
```

This uses `shap.LinearExplainer` on the averaged linear coefficients from the
calibrated `LinearSVC` folds and explains the uncalibrated decision function.
It is a valid SHAP explanation of the current TF-IDF model, but it is not a
SHAP explanation of a fine-tuned DistilBERT model.

The accepted-paper DistilBERT workflow is separate. Fine-tune the model on the
strict training split and evaluate it on the untouched external split:

```bash
python app.py train-distilbert --data data/external/dreaddit_adapt_train_strict.csv --external-data data/external/dreaddit_adapt_test.csv --output-dir models/distilbert_dreaddit_strict --metrics research/reports/metrics_distilbert_dreaddit_strict.json --epochs 2 --batch-size 8 --max-length 256
```

Then generate token-level SHAP values:

```bash
python app.py distilbert-shap-audit --data data/external/dreaddit_adapt_test.csv --model-dir models/distilbert_dreaddit_strict --output research/reports/shap_distilbert_dreaddit_strict.json --max-samples 10 --max-evals 200
```

This uses `shap.Explainer` with `shap.maskers.Text` and the Partition
algorithm to explain class-1 probability. The generated artifact records the
model directory, sample count, tokenizer-based tokens, base probability, and
per-token SHAP values. SHAP is audited on a representative subset because
token masking is substantially slower than ordinary inference. The latest audit
covered 10 representative samples and had a maximum probability additivity
error of approximately `2.22e-16`.

The latest TF-IDF SHAP audit covered all 715 external samples and had a maximum
decision-function additivity error of approximately `1.55e-15`. TF-IDF SHAP
explains the uncalibrated linear decision function, whereas DistilBERT SHAP
explains class-1 probability. Neither attribution method is clinical evidence
or a diagnosis.

Validate strict external split:

```bash
python app.py validate-external --external-data data/external/dreaddit_adapt_test.csv --training-data data/external/dreaddit_adapt_train_strict.csv --output research/reports/external_dataset_validation_dreaddit_adapt_strict.json
```

Train strict-adapted model:

```bash
python app.py train --data data/external/dreaddit_adapt_train_strict.csv --model models/text_model_dreaddit_strict.pkl --vectorizer models/vectorizer_dreaddit_strict.pkl --metrics research/reports/metrics_dreaddit_adapt_strict.json
```

Run strict external evaluation:

```bash
python app.py evaluate --data data/external/dreaddit_adapt_train_strict.csv --external-data data/external/dreaddit_adapt_test.csv --model models/text_model_dreaddit_strict.pkl --vectorizer models/vectorizer_dreaddit_strict.pkl --output research/reports/research_evaluation_dreaddit_adapted_strict.json --chart research/charts/ablation_comparison_dreaddit_adapted_strict.png
```

Generate visual report package:

```bash
python app.py visual-report --metrics research/reports/metrics_dreaddit_adapt_strict.json --evaluation research/reports/research_evaluation_dreaddit_adapted_strict.json --charts-dir research/charts --report-dir research/reports/dreaddit_strict
```

Optional cross-domain baseline run:

```bash
python app.py validate-external --external-data data/external/dreaddit_test.csv --training-data data/splits/train.csv --output research/reports/external_dataset_validation_dreaddit.json

python app.py evaluate --data data/splits/train.csv --external-data data/external/dreaddit_test.csv --output research/reports/research_evaluation_dreaddit.json --chart research/charts/ablation_comparison_dreaddit.png
```

## Key Output Artifacts

- `research/reports/research_evaluation_dreaddit.json`
- `research/reports/research_evaluation_dreaddit_adapted_strict.json`
- `research/reports/metrics_dreaddit_adapt_strict.json`
- `research/reports/external_dataset_validation_dreaddit.json`
- `research/reports/external_dataset_validation_dreaddit_adapt_strict.json`
- `research/reports/dreaddit_strict/visual_report.html`
- `research/reports/explainability_audit_dreaddit_adapted_strict.json`
- `research/reports/model_benchmark_dreaddit_adapted_strict.json`
- `research/reports/shap_audit_dreaddit_adapted_strict.json`
- `research/reports/metrics_distilbert_dreaddit_strict.json`
- `research/reports/shap_distilbert_dreaddit_strict.json`
- `research/charts/ablation_comparison_dreaddit.png`
- `research/charts/ablation_comparison_dreaddit_adapted_strict.png`
- `research/charts/generalization_comparison.png`
- `research/charts/robustness_perturbation_chart.png`

## Face-Emotion Troubleshooting

If face detection frequently returns fallback behavior:

1. Ensure dependencies are installed in the active environment.
2. Verify DeepFace import:

```bash
python -c "from deepface import DeepFace; print('ok')"
```

3. Use clear, front-facing, well-lit images.
4. Check `face_note` fields in API responses for explicit fallback reason.

Note: `tf-keras` is required with TensorFlow 2.21+ for DeepFace/RetinaFace compatibility.

## Screenshots

Update these image paths if you maintain UI screenshots:

- `docs/images/dashboard-home.png`
- `docs/images/analysis-results.png`
- `docs/images/report-download.png`

## Future Enhancements

- human-in-the-loop review for high-risk cases
- multilingual and localized support
- session-level temporal trend modeling
- asynchronous or cached DistilBERT SHAP explanations for faster API responses
- secure auth and session management
- cloud deployment with CI/CD and observability

## Disclaimer

MIND-AID is a research and decision-support prototype. It does not provide medical diagnosis and must not replace professional mental health care.
