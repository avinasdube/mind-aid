Date: 24 April 2026

To
The Editor / Program Chair

Subject: Submission of manuscript titled "MIND-AID: Domain-Shift-Aware Depression Risk Screening from Social Text with Strict External Validation"

Dear Editor and Review Committee,

We submit our manuscript for consideration in your applied AI and health informatics track. This work addresses a central reliability issue in mental health NLP: strong in-sample performance often fails under cross-domain deployment.

Our manuscript presents MIND-AID, a reproducible screening framework with three practical contributions:
1. A calibrated text-risk pipeline with a transformer sentence-embedding baseline, optional confidence-aware multimodal fusion, and explainability.
2. A strict external benchmarking protocol with overlap checks and explicit separation of diagnostic versus claim-valid result blocks.
3. Quantitative evidence that strict, deduplicated domain adaptation materially improves external reliability on Dreaddit test data (n = 715), now supported by bootstrap confidence intervals.

Key external improvements from cross-domain baseline to strict adaptation are:
- Accuracy: 0.5231 to 0.7203
- Balanced accuracy: 0.5075 to 0.7165
- F1: 0.3738 to 0.7159
- ROC-AUC: 0.6483 to 0.8248
- Brier score: 0.4531 to 0.1794

The updated evaluation also includes a transformer baseline, which performs better than the original cross-domain TF-IDF-only baseline and strengthens the comparison set for review.

We emphasize that this is a decision-support research system, not a clinical diagnostic tool. The manuscript explicitly documents limitations, ethics considerations, and threat-to-validity controls.

This submission is original, not under concurrent review elsewhere, and approved by all authors. We believe the paper is a strong fit for venues valuing reproducibility, external validity, and deployment-aware evaluation in trustworthy AI.

Thank you for your consideration.

Sincerely,

[Corresponding Author Name]
[Affiliation]
[Email]
[ORCID, optional]
