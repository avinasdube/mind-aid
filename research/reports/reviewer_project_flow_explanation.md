# MIND-AID Project Working and End-to-End Flow (Reviewer Presentation Version)

## 1) What this project does
MIND-AID is a research prototype for early depression-risk screening support.

In simple terms:
- A user gives text (and optionally an image).
- The system estimates risk from text.
- If image evidence is usable, it can combine text and face-emotion signals.
- It returns a risk label, confidence score, and important keywords that influenced the decision.

Important note for reviewers:
This is a decision-support research tool, not a clinical diagnosis system.

## 2) Why this project was built
Many ML papers report very high scores on the same dataset they train on, but models often fail on new external data (domain shift problem).

This project was designed to answer a practical research question:
How reliable is the model on unseen external data, and how much can strict adaptation improve it?

## 3) Full project flow from scratch

### Step A: Data preparation
1. Internal dataset is prepared and cleaned.
2. For strict experiments, data is split into train and unseen validation.
3. External datasets are validated for schema and overlap risk before testing.

Why this matters:
If training and test data overlap, reported performance is misleading.

### Step B: Text model training
1. Text is normalized and vectorized with TF-IDF (unigrams and bigrams).
2. A calibrated linear classifier is trained.
3. Calibration is used to make risk probabilities more meaningful.
4. Training outputs model artifacts and metrics.

There is now also a transformer sentence-embedding baseline in the evaluation pipeline. It is not the main deployed model, but it provides a stronger modern benchmark for comparison.

### Step C: Evaluation strategy (core research logic)
The project uses two different evaluation scenarios on Dreaddit test data:

1. Cross-domain baseline:
- Train on internal data.
- Test on Dreaddit external test set.

2. Strict domain adaptation:
- Train on deduplicated Dreaddit train+validation data.
- Test on Dreaddit test set.
- Verify zero overlap before final claim.

Why this matters:
The baseline measures domain shift pain.
The strict adaptation measures how much performance can be recovered when train domain is aligned carefully.

### Step D: Prediction flow (runtime)
1. User submits text (optional image).
2. Backend loads model and vectorizer.
3. Text risk probability and class are computed.
4. Explainability module extracts top meaningful keywords.
5. If image is provided and face confidence is valid, fusion can combine signals.
6. API returns:
- risk label
- risk score
- confidence
- explanation keywords
- fusion metadata (if used)

### Step E: Reporting and visual outputs
1. Evaluation JSON reports are generated.
2. Charts are created for ablations and calibration.
3. Visual HTML report is generated for reviewer-friendly inspection.

## 4) Most important result (for presentation)
The strongest evidence is external validation, not internal in-sample scores.

External comparison on Dreaddit test (715 samples):

| Metric | Baseline (Internal Train -> Dreaddit Test) | Strict Adaptation (Deduped Dreaddit Train+Val -> Dreaddit Test) | Transformer Baseline |
| --- | ---: | ---: | ---: |
| Accuracy | 0.5231 | 0.7217 | 0.6182 |
| Balanced Accuracy | 0.5075 | 0.7177 | 0.6180 |
| F1 | 0.3738 | 0.7168 | 0.5832 |
| ROC-AUC | 0.6483 | 0.8257 | 0.6628 |
| Brier Score (lower better) | 0.4531 | 0.1791 | 0.3036 |

Interpretation in plain language:
- The baseline performs weakly on external data, showing clear domain shift.
- Strict adaptation greatly improves both classification quality and probability reliability.
- The transformer baseline is stronger than the old cross-domain text-only baseline, which makes the comparison more credible for reviewers.
- Overlap check passed: strict train vs external test overlap = 0.

The updated reports also include 95% bootstrap confidence intervals, which help show that the gains are not just random variation.

## 5) How to explain this to a reviewer in 60 seconds
Use this short script:
1. We built MIND-AID as a reproducible depression-risk screening framework.
2. We intentionally tested external generalization, not only in-sample accuracy.
3. Cross-domain baseline dropped strongly on Dreaddit external data.
4. After strict deduplicated adaptation, external metrics improved substantially (accuracy 0.523 to 0.720, ROC-AUC 0.648 to 0.825, Brier 0.453 to 0.179).
5. We treat internal near-perfect scores as diagnostic only, and use external validation for primary claims.
6. The project includes explainability, reproducibility commands, and visual reports for transparent review.

## 6) Strengths of this work
1. External-first evaluation mindset.
2. Anti-leakage overlap checks.
3. Reproducible pipeline and generated artifacts.
4. Interpretable output (keywords + confidence).
5. Practical API and frontend integration.

## 7) Honest limitations (important for credibility)
1. Strongest benchmark claims are text-only because Dreaddit external files do not include image columns.
2. This is a screening support system, not a clinical diagnostic tool.
3. More external datasets and stronger deep baseline comparisons would improve publication strength further.

## 8) Key files a reviewer can inspect quickly
1. research/reports/research_evaluation_dreaddit.json
2. research/reports/research_evaluation_dreaddit_adapted_strict.json
3. research/reports/external_dataset_validation_dreaddit_adapt_strict.json
4. research/reports/finalization_summary.md
5. research/reports/dreaddit_strict/visual_report.html
6. research/charts/
7. research/reports/paper_ieee_submission_draft.md

## 9) One-line final takeaway
MIND-AID demonstrates that strict, leakage-aware domain adaptation can convert a weak cross-domain baseline into a substantially more reliable external depression-risk screening model, with transparent and reproducible evidence.
