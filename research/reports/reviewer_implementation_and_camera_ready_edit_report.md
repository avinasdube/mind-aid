# Reviewer-Driven Implementation, Validation, and Camera-Ready Edit Report

**Project:** MIND-AID  
**Repository:** `D:\MIND-AID`  
**Purpose:** Provide a single, repository-readable source of truth for reviewing the implemented reviewer-requested changes and editing the already accepted five-page camera-ready paper without rewriting it.

## 1. How to use this report

This report is intended to be read together with the original accepted manuscript:

`D:\MINDAID-V2\MIND-AID-CAMERA-READY\MIND-AID_Camera-Ready_Paper.docx`

The accepted DOCX remains the source of truth. This report does **not** authorize replacing the paper with the earlier working Markdown draft. The recommended editing method is:

1. Make a copy of the accepted DOCX.
2. Locate the existing section or paragraph named below.
3. Replace ambiguous or inaccurate wording with the compact text supplied here.
4. Add only the reviewer-specific evidence required.
5. Keep the title, authors, section order, references, accepted tables, and existing figures unless an explicit instruction below says to update a caption or an already existing chart.
6. Confirm that the final document remains five pages.

Reviewer Comments 1–2 were not available in the supplied materials. This report therefore addresses Comments 3–10 only and does not invent responses for Comments 1–2.

---

## 2. Executive summary of the major changes

The project now contains two explicitly selectable text models:

| Model | Role | Explanation |
|---|---|---|
| TF-IDF + calibrated LinearSVC | Default/deployed model and lightweight baseline | TF-IDF feature value multiplied by learned linear coefficient |
| Fine-tuned DistilBERT | Optional selectable semantic model and research comparison | Token-level SHAP using `shap.Explainer`, `shap.maskers.Text`, and Partition |

The React frontend allows the user to select either model. The API response records the selected model and the explanation method. The default remains TF-IDF for backward compatibility; selecting DistilBERT is an explicit user choice.

The repository now includes:

- Exact normalized duplicate removal before splitting and fitting.
- Strict external leakage checks.
- Strict external evaluation on a 715-row unseen test set.
- Reproducible TF-IDF SHAP explanations.
- A reproducible fine-tuned DistilBERT training workflow.
- Reproducible token-level DistilBERT SHAP explanations.
- Explainability output filtering that removes stopwords, special tokens, duplicates, and unreadable fragments.
- Resource and repeated-seed benchmarking.
- Robustness perturbation testing.
- React model selection and model-specific result display.
- A broader `.gitignore` for local datasets, model weights, outputs, credentials, caches, and uploaded files.

The original accepted internal results and the new strict external results are different experiments. They must not be silently merged.

---

## 3. Original accepted-paper claims that must be preserved carefully

The accepted DOCX claims:

- A TF-IDF + LinearSVC baseline.
- A fine-tuned DistilBERT model.
- Coefficient-based feature importance.
- SHAP-based token attribution.
- Original accepted Phase 1 TF-IDF accuracy of approximately 99.3%.
- Original accepted Phase 2 DistilBERT accuracy of approximately 98.9%.

The repository originally did not reproduce the DistilBERT/SHAP claim. It contained a sentence-embedding baseline using `sentence-transformers/all-MiniLM-L6-v2`, not the fine-tuned DistilBERT model described in the accepted paper.

The repository now contains a separate fine-tuning and SHAP workflow that supports the DistilBERT/SHAP claim. However, the newly generated strict external experiment is not necessarily the same protocol as the original accepted Phase 2 experiment. Therefore:

- Keep the original accepted Phase 1 and Phase 2 tables unchanged.
- Add the new strict results as an explicitly labelled supplementary validation experiment in prose.
- Do not replace 99.3% or 98.9% with 72.31% or 79.16% without a separate author decision.
- Do not describe the deployed TF-IDF API as if it were the DistilBERT model.

---

## 4. Implemented code changes

### 4.1 Leakage control and duplicate handling

Relevant files:

- `src/dataset_split.py`
- `src/train_model.py`
- `src/validate_dataset.py`

Behavior:

- Text is normalized using the project preprocessor.
- Empty rows are removed.
- Exact duplicate normalized texts are removed before splitting or fitting.
- Training metadata records input rows and removed duplicate rows.
- External validation reports duplicate test rows and exact normalized train–test overlap.
- TF-IDF is fitted on training text and only transformed on held-out text.

This is exact duplicate control, not a claim that near-duplicates or user-level duplicates have been completely eliminated. Author identifiers are unavailable, so user-independent evaluation cannot be guaranteed.

### 4.2 TF-IDF model and explanation

Relevant files:

- `src/train_model.py`
- `src/predict_text.py`
- `src/explain.py`
- `src/shap_explainability.py`

The TF-IDF pipeline uses:

- TF-IDF uni-grams and bi-grams.
- Calibrated `LinearSVC`.
- Class-weight balancing.
- Probability output through calibration.

The default lightweight explanation path uses:

```text
TF-IDF feature value × learned linear coefficient
```

The separate TF-IDF SHAP audit uses:

- `shap.LinearExplainer`.
- Averaged linear coefficients from the five calibrated LinearSVC fold estimators.
- The uncalibrated linear decision function as the explained output.

This is a valid SHAP explanation of the current TF-IDF linear model. It is not a DistilBERT explanation.

### 4.3 Fine-tuned DistilBERT

Relevant file:

- `src/distilbert_explainability.py`

Training configuration used for the newest saved model:

- Base checkpoint: `distilbert-base-uncased`
- Labels: binary class 0 and class 1
- Training data: `data/external/dreaddit_adapt_train_strict.csv`
- External data: `data/external/dreaddit_adapt_test.csv`
- Seed: 42
- Epochs: 2
- Batch size: 8
- Maximum sequence length: 256
- Learning rate: `2e-5`
- Device: CPU
- Training rows after normalization and duplicate removal: 2,816
- External rows: 715

Saved model directory:

`models/distilbert_dreaddit_strict/`

The model weights are approximately 255.43 MiB. This is a separate research model and is selectable from the frontend but is not silently substituted for the default TF-IDF model.

### 4.4 DistilBERT SHAP

Relevant file:

- `src/distilbert_explainability.py`

The DistilBERT SHAP workflow uses:

- `shap.Explainer`.
- `shap.maskers.Text`.
- Partition algorithm.
- Class-1 probability as the explained output.

The report stores:

- Input text.
- True label.
- Predicted label.
- Class-1 probability.
- Base class-1 probability.
- Additivity error.
- Readable top tokens.
- Positive and negative SHAP values.

The display layer filters:

- English stopwords.
- Special tokens.
- Duplicate tokens.
- Empty tokens.
- Very short non-content fragments.

Raw SHAP values remain available in the JSON report; filtering is only for readable keyword presentation.

### 4.5 API model selection

Relevant file:

- `api_server.py`

`POST /analyze` now accepts form data:

```text
model_name=tfidf
```

or:

```text
model_name=distilbert
```

The default is `tfidf`.

The response includes:

- `model`
- `model_label`
- `explanation_method`
- `keywords`
- `depression_probability`
- `risk_label`
- `crisis_flag`

The API rejects unsupported model names with HTTP 400.

The response identifies the explanation method:

```text
TF-IDF feature value × learned linear coefficient
```

or:

```text
Token-level SHAP Partition attribution
```

### 4.6 React frontend model selector

Relevant file:

- `frontend/src/App.jsx`

The interface now provides:

- `TF-IDF + LinearSVC (fast)`
- `Fine-tuned DistilBERT (semantic)`

The results page displays the selected model and explanation method. The frontend keeps TF-IDF as the default and makes DistilBERT selection explicit.

### 4.7 Repository publication protection

Relevant file:

- `.gitignore`

New ignore coverage includes:

- `data/`
- `models/`
- `outputs/`
- CSV files.
- Model weights and checkpoints.
- Generated charts and JSON reports.
- Frontend dependencies and build output.
- Uploaded images and runtime logs.
- `.env` files and credentials.
- Certificates and private keys.
- Caches and downloaded model stores.
- Local DOCX/PDF/archive files.

`.gitignore` does not remove files already tracked by Git. Any already-tracked private or generated files must be removed from the Git index separately with `git rm --cached` after review.

---

## 5. Latest verified strict results

### 5.1 Strict external data controls

Source artifact:

`research/reports/external_dataset_validation_dreaddit_adapt_strict.json`

Verified values:

| Check | Result |
|---|---:|
| Raw external rows | 715 |
| Valid external rows | 715 |
| Duplicate external-test rows | 0 |
| Exact normalized train–test overlap | 0 |
| Strict training rows before duplicate removal | 2,835 |
| Training duplicate rows removed before fitting | 19 |
| Image rows | 0 |
| Image-related warning | Fusion ablations skipped because no image column exists |

The external report status is `warning`, not `error`, solely because the Dreaddit external data contains no image column. This is expected for text-only external evaluation.

### 5.2 Strict TF-IDF external evaluation

Source artifact:

`research/reports/research_evaluation_dreaddit_adapted_strict.json`

The primary external validation block reports:

| Metric | Value |
|---|---:|
| Test samples | 715 |
| Accuracy | 72.31% |
| Balanced accuracy | 71.91% |
| Weighted precision | 73.35% |
| Weighted recall | 72.31% |
| Weighted F1 | 71.83% |
| ROC-AUC | 82.57% |
| Brier score | 0.1794 |
| Accuracy bootstrap 95% CI | 68.81%–75.52% |
| Weighted F1 bootstrap 95% CI | 68.31%–75.12% |

External confusion matrix:

```text
              predicted 0   predicted 1
actual 0           206          140
actual 1            58          311
```

Interpretation:

- The strict TF-IDF model detects many class-1 examples.
- It also produces a substantial number of false positives, which is consistent with the positive-paragraph test.
- It should be described as a lightweight baseline/deployed default, not as a clinically reliable diagnostic model.

### 5.3 Fine-tuned DistilBERT external evaluation

Source artifact:

`research/reports/metrics_distilbert_dreaddit_strict.json`

| Metric | Value |
|---|---:|
| Base model | `distilbert-base-uncased` |
| Training rows | 2,816 |
| External rows | 715 |
| Epochs | 2 |
| Batch size | 8 |
| Maximum sequence length | 256 |
| Learning rate | `2e-5` |
| Device | CPU |
| Accuracy | 79.16% |
| Weighted F1 | 79.14% |
| ROC-AUC | 88.63% |
| Saved model size | approximately 255.43 MiB |

Interpretation:

- DistilBERT performed better than strict TF-IDF on this same external test split.
- This is a comparison on the current strict experiment, not a replacement for the accepted Phase 2 result.
- No paired significance test was performed.
- Peak resident memory was not measured.
- The result should not be presented as proof of clinical validity.

### 5.4 TF-IDF SHAP audit

Source artifact:

`research/reports/shap_audit_dreaddit_adapted_strict.json`

| Measure | Result |
|---|---:|
| Samples | 715 |
| Explainer | `shap.LinearExplainer` |
| Explained output | Uncalibrated LinearSVC decision function |
| Feature count | 111,583 |
| Background rows | 100 |
| Maximum absolute additivity error | `1.55 × 10^-15` |

This confirms numerical SHAP additivity for the TF-IDF linear explanation path.

### 5.5 DistilBERT SHAP audit

Source artifact:

`research/reports/shap_distilbert_dreaddit_strict.json`

| Measure | Result |
|---|---:|
| Samples | 10 representative external examples |
| Explainer | `shap.Explainer` |
| Masker | `shap.maskers.Text` |
| Algorithm | Partition |
| Explained output | Class-1 probability |
| Maximum absolute additivity error | `2.22 × 10^-16` |

The 10-example SHAP report is a representative audit, not a claim that all 715 external examples were individually explained with SHAP.

Example readable tokens include:

- Positive class example: `helpless`, `die`, `wanna`, `swear`, `scared`.
- Negative class examples: `helped`, `prepared`, `hours`, `friend`, `retirement`, `knowledge`, `employees`.

These are model-associated tokens only. They are not validated clinical indicators.

### 5.6 Robustness results

Latest strict TF-IDF perturbation results:

| Perturbation | Prediction flip rate |
|---|---:|
| Punctuation removal | 0.0% |
| Synthetic spelling noise | approximately 2.0% |
| Slang/abbreviation substitution | approximately 2.9% |
| Word dropout | approximately 9.8% |
| Stopword removal preserving negations | approximately 27.6% |
| Truncation to 20 tokens | approximately 31.3% |
| Truncated weighted F1 | 0.6673 |

Sarcasm, naturally occurring spelling variation, and indirect emotional expressions were not independently annotated or evaluated.

### 5.7 Explainability audit beyond SHAP

Source artifact:

`research/reports/explainability_audit_dreaddit_adapted_strict.json`

Verified values:

- Samples: 715.
- Mean top-keyword Jaccard stability after punctuation perturbation: 0.8338.
- Mean absolute probability change after masking top keywords: 0.1520.
- Prediction flip rate after masking: 23.36%.
- Misclassified examples: 198.

This audit is a faithfulness/stability proxy. It is not clinical validation.

---

## 6. Latest application and frontend tests

### 6.1 Backend and frontend environment used

The repository was tested on isolated ports because port 8000 was occupied by a different checkout:

- Current MIND-AID backend: `http://127.0.0.1:8001`
- Current MIND-AID frontend: `http://127.0.0.1:5174`
- Existing other-checkout backend: `D:\MINDAID-V2` on port 8000

The other-checkout process was not terminated.

### 6.2 Build and startup checks

Backend:

```text
GET /health → HTTP 200
```

Frontend:

```text
Vite production build → PASS
31 modules transformed
```

The frontend build command on Windows is:

```powershell
cd D:\MIND-AID\frontend
npm.cmd run build
```

### 6.3 Positive paragraph test

The positive paragraph supplied during testing expresses optimism, gratitude, hope, excitement, and future goals.

TF-IDF selection:

- Probability: approximately 81.5%.
- Risk: High Risk.
- Explanation tokens included `waiting`, `memories`, `leave`, `happening`, and `fully`.
- This is a demonstrated TF-IDF false positive caused by word-level associations and limited context understanding.

DistilBERT selection:

- Probability: approximately 5.8% in the React frontend test.
- Risk: Low Risk.
- Explanation tokens included `going`, `future`, `leave`, `know`, `make`, `opportunity`, `happiness`, and `little`.
- This result is more consistent with the paragraph’s meaning.

### 6.4 Distress paragraph test

Input:

```text
I feel hopeless and unable to cope
```

DistilBERT result:

- Probability: approximately 97.6%.
- Risk: High Risk.
- SHAP tokens: `feel`, `hopeless`, `unable`, `cope`.

This confirms that the selectable DistilBERT model responds strongly to explicit distress language while producing a low score for the positive paragraph.

### 6.5 Frontend behavior

The React UI was verified to:

- Render the model selector.
- Submit `model_name`.
- Disable model selection while analyzing.
- Display the selected model.
- Display the selected explanation method.
- Display model-specific keywords.
- Display the risk category, probability, confidence, score bars, disclaimer, and report control.

The frontend does not currently display a full SHAP plot. It displays the top readable tokens returned by the API. Full SHAP values remain in the research JSON artifact.

---

## 7. Exact paper edits by reviewer comment

The following instructions are deliberately compact to preserve the mandatory five-page length.

### Comment 3 — Data leakage and overestimated performance

**Edit location:** Existing “Dataset description” and “Training and evaluation procedure” sections.

**Add or replace with:**

> To reduce leakage risk, text was normalized before exact duplicate removal, and duplicate removal was performed before model fitting. For the TF-IDF pipeline, the vocabulary and feature weights were fitted only on the training partition and applied to held-out data using transformation only. In the strict external evaluation, the test set contained 715 samples, with zero exact normalized train–test overlap and zero duplicate rows within the external test set. These strict external results are therefore treated as the primary generalization evidence. Because author identifiers were unavailable, user-level independence could not be guaranteed.

**Add one result sentence:**

> On this strict external split, the TF-IDF model achieved 72.31% accuracy, 71.91% balanced accuracy, 71.83% weighted F1, 82.57% ROC-AUC, and a Brier score of 0.1794.

**Do not change:** Existing accepted Phase 1/Phase 2 tables.

**Do not add:** A new leakage table.

### Comment 4 — Novelty and research contribution

**Edit location:** Final paragraph of the Introduction.

**Add:**

> The contribution of this work is methodological and integrative rather than the invention of a new TF-IDF, LinearSVC, DistilBERT, or SHAP algorithm. The study integrates classical and transformer-based text modeling, multimodal screening, leakage-aware external evaluation, and model-specific explainability within one reproducible workflow.

**Do not add:** A new contribution section.

### Comment 5 — Explainability validation

**Edit location:** Existing Explainability subsection and Figure 3 caption/discussion.

**Add:**

> Explainability is model-specific. For the deployed TF-IDF model, local feature influence is estimated by multiplying each active TF-IDF feature value by its learned linear coefficient. For the selectable fine-tuned DistilBERT model, token-level explanations are generated using SHAP’s text masker and Partition explainer, with class-1 probability as the explained output.

**Add compact evidence:**

> The TF-IDF SHAP audit covered 715 external samples and had a maximum additivity error of approximately \(1.55 \times 10^{-15}\). The DistilBERT SHAP audit covered 10 representative external examples and had a maximum probability additivity error of approximately \(2.22 \times 10^{-16}\). Displayed tokens were filtered for readability by removing stopwords, special tokens, and duplicate entries.

**Add interpretation limitation:**

> These values identify tokens that influence model output and should not be interpreted as clinically validated symptoms.

**Figure 3 action:** Update only the caption so it names the actual method:

- If it shows TF-IDF: “Coefficient-based TF-IDF feature attribution for the deployed LinearSVC model.”
- If it shows DistilBERT: “Token-level SHAP attribution for the fine-tuned DistilBERT model.”
- If it combines both: “Model-specific explainability: coefficient-based TF-IDF attribution for the deployed model and token-level SHAP attribution for the selectable fine-tuned DistilBERT model.”

**Do not add:** A second explainability figure.

### Comment 6 — Fairness of model comparison

**Edit location:** Existing Phase 1/Phase 2 comparative discussion.

**Add one compact paragraph:**

> For the strict external comparison, both text models were evaluated on the same 715-row test split. The TF-IDF model used 111,583 features, with a 4.47 MB classifier and 2.84 MB vectorizer. The fine-tuned DistilBERT model used `distilbert-base-uncased`, two training epochs, batch size 8, maximum sequence length 256, and CPU execution; its saved model weights occupied approximately 255.43 MiB. On the strict external split, DistilBERT achieved 79.16% accuracy, 79.14% weighted F1, and 88.63% ROC-AUC, compared with 72.31%, 71.83%, and 82.57% for TF-IDF. No paired significance test or peak-memory measurement is claimed.

**Do not add:** A new resource table.

### Comment 7 — Clinical relevance and ethics

**Edit location:** Existing Limitations, Conclusion, or Future Scope paragraph.

**Add:**

> The dataset labels are source-dataset annotations and should not be interpreted as clinician-confirmed psychiatric diagnoses. MIND-AID is intended for research and decision-support prototyping rather than autonomous diagnosis. Real deployment would require consent or lawful data governance, privacy protection, bias auditing, escalation protocols, qualified human review, and prospective clinical validation.

**Add if space permits:**

> Both TF-IDF and DistilBERT outputs represent model estimates of dataset-associated risk patterns, not medical conclusions.

### Comment 8 — Robustness and real-world applicability

**Edit location:** Existing Results Discussion or Limitations/Future Work.

**Add:**

> In the strict external robustness audit, punctuation removal caused 0% prediction flips, synthetic spelling noise caused 2.0% flips, slang or abbreviation substitution caused 2.9% flips, word dropout caused 9.8% flips, and stopword removal while preserving negations caused 27.6% flips. Truncation to 20 tokens caused 31.3% flips and reduced weighted F1 to 0.6673. These results indicate sensitivity to information loss, particularly when negation or longer context is removed. Sarcasm, naturally occurring spelling variation, and indirect emotional expressions were not independently annotated or evaluated.

**Chart action:** Do not add a new robustness chart if the paper has no existing robustness figure. If an existing robustness figure is already present, update that existing figure rather than increasing the page count.

### Comment 9 — Comparison with existing research

**Edit location:** Immediately before or after existing Table 4.

**Add:**

> Results from different studies are contextual rather than directly comparable because datasets, label policies, preprocessing procedures, class distributions, and train–test splits differ. The fairest comparisons in this study are the TF-IDF and fine-tuned DistilBERT models evaluated on the same 715-row strict external test split.

**Optional, only if space permits:**

> On this common split, DistilBERT achieved 79.16% accuracy and 88.63% ROC-AUC, while TF-IDF achieved 72.31% accuracy and 82.57% ROC-AUC.

**Do not change:** Existing Table 4 values.

### Comment 10 — Reproducibility and implementation details

**Edit location:** End of Methodology or existing reproducibility paragraph.

**Add:**

> Reproducibility is supported through fixed random seed 42, exact normalized duplicate removal before fitting, training-only TF-IDF fitting, and an untouched external test split. The strict TF-IDF model uses TF-IDF uni- and bi-grams with calibrated LinearSVC. The transformer experiment fine-tunes `distilbert-base-uncased` for two epochs using batch size 8, maximum sequence length 256, and learning rate \(2 \times 10^{-5}\). TF-IDF explanations use `shap.LinearExplainer`, while DistilBERT explanations use `shap.Explainer` with `shap.maskers.Text` and the Partition algorithm. The frontend permits explicit selection of either model, and the API response records the selected model and explanation method.

**Add a short artifact statement:**

> The corresponding models, metrics, and explanation reports are stored in the repository under `models/` and `research/reports/`.

**Critical clarification:**

> The default deployed API model is TF-IDF, while fine-tuned DistilBERT is available as an explicitly selectable research model with a separate token-level SHAP workflow.

---

## 8. Table and figure update decisions

### Existing Phase 1 table

**Keep unchanged.**

Do not replace the original accepted TF-IDF values with strict external values. If space permits, add a short footnote:

> These values correspond to the original internal experimental protocol and are distinct from the strict external validation results reported separately.

### Existing Phase 2 table

**Keep unchanged.**

Do not replace the accepted DistilBERT values with the newest strict external values. The new strict fine-tuning run is an additional experiment with a different data/evaluation protocol.

### Existing Table 4

**Keep unchanged.**

Add the contextual-comparison sentence before or after it.

### Existing Figure 3

Update the caption only if it currently claims a method that does not match the image. Do not add another figure. The caption must distinguish TF-IDF coefficient attribution from DistilBERT SHAP attribution.

### Robustness chart

Project artifact:

`research/charts/robustness_perturbation_chart.png`

Recommended paper action: report robustness values in prose. Add or replace a figure only if the existing paper already contains a robustness figure and the replacement does not increase the paper beyond five pages.

### Other charts

Do not automatically insert the following into the paper:

- `generalization_comparison.png`
- `calibration_curve.png`
- `ablation_comparison_dreaddit_adapted_strict.png`

They are supporting repository artifacts, not automatic replacements for accepted paper figures.

---

## 9. Five-page preservation rules

To keep the paper at five pages:

1. Do not add new tables.
2. Do not add new figures.
3. Add strict metrics once, in a compact paragraph.
4. Combine comparison, resource, and reproducibility details where possible.
5. Put ethics and deployment limitations into the existing limitations paragraph.
6. Update Figure 3’s caption rather than adding a second explainability figure.
7. Keep original accepted tables and figures unchanged unless a caption correction is necessary.
8. Remove repeated or vague wording instead of appending duplicate explanations.
9. Do not repeat the same accuracy and ROC-AUC values in every section.
10. After editing, check the page count in the Word document and shorten prose before changing layout or font sizes.

---

## 10. Known limitations and claims that must not be made

Do not claim:

- The system is a clinical diagnostic tool.
- The dataset labels are clinician-confirmed diagnoses.
- SHAP proves clinical relevance.
- SHAP identifies depression symptoms.
- The TF-IDF model understands paragraph-level sentiment.
- DistilBERT is used for every API request.
- The new strict metrics replace the accepted metrics.
- The models were statistically significantly different without a paired test.
- Peak memory was measured.
- User-level independence was guaranteed.
- Sarcasm and indirect expressions were fully evaluated.
- The 10-example DistilBERT SHAP audit is a 715-example full SHAP audit.

Use:

> Model-associated lexical or token-level evidence

instead of:

> Clinically validated symptom evidence

Use:

> Screening estimate

instead of:

> Diagnosis

---

## 11. Reproduction commands

Run from the repository root.

### Validate strict external data

```powershell
.\.venv\Scripts\python.exe app.py validate-external `
  --external-data data\external\dreaddit_adapt_test.csv `
  --training-data data\external\dreaddit_adapt_train_strict.csv `
  --output research\reports\external_dataset_validation_dreaddit_adapt_strict.json
```

### Train strict TF-IDF

```powershell
.\.venv\Scripts\python.exe app.py train `
  --data data\external\dreaddit_adapt_train_strict.csv `
  --model models\text_model_dreaddit_strict.pkl `
  --vectorizer models\vectorizer_dreaddit_strict.pkl `
  --metrics research\reports\metrics_dreaddit_adapt_strict.json
```

### Evaluate strict TF-IDF

```powershell
.\.venv\Scripts\python.exe app.py evaluate `
  --data data\external\dreaddit_adapt_train_strict.csv `
  --external-data data\external\dreaddit_adapt_test.csv `
  --model models\text_model_dreaddit_strict.pkl `
  --vectorizer models\vectorizer_dreaddit_strict.pkl `
  --output research\reports\research_evaluation_dreaddit_adapted_strict.json `
  --chart research\charts\ablation_comparison_dreaddit_adapted_strict.png
```

### Train strict DistilBERT

```powershell
.\.venv\Scripts\python.exe app.py train-distilbert `
  --data data\external\dreaddit_adapt_train_strict.csv `
  --external-data data\external\dreaddit_adapt_test.csv `
  --output-dir models\distilbert_dreaddit_strict `
  --metrics research\reports\metrics_distilbert_dreaddit_strict.json `
  --epochs 2 `
  --batch-size 8 `
  --max-length 256
```

### Generate DistilBERT SHAP

```powershell
.\.venv\Scripts\python.exe app.py distilbert-shap-audit `
  --data data\external\dreaddit_adapt_test.csv `
  --model-dir models\distilbert_dreaddit_strict `
  --output research\reports\shap_distilbert_dreaddit_strict.json `
  --max-samples 10 `
  --max-evals 200
```

### Run the selectable application

Backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api_server:app `
  --host 127.0.0.1 `
  --port 8001
```

Frontend:

```powershell
cd frontend
$env:VITE_API_BASE_URL = "http://127.0.0.1:8001"
npm.cmd run dev -- --host 127.0.0.1 --port 5174
```

Open:

`http://127.0.0.1:5174`

---

## 12. Final paper consistency statement

The final camera-ready revision should communicate the following:

> The original accepted Phase 1 and Phase 2 results are preserved as originally reported. Newly added strict external experiments provide leakage-aware supplementary evidence. On the same 715-row strict external test split, the fine-tuned DistilBERT model achieved 79.16% accuracy, while the strict TF-IDF model achieved 72.31% accuracy. TF-IDF explanations use coefficient-based attribution, whereas DistilBERT explanations use token-level SHAP. The frontend permits explicit model selection, and neither explanation method constitutes clinical evidence or a medical diagnosis.

This report is the implementation-and-evidence reference for the next manual edit of the accepted camera-ready DOCX.
