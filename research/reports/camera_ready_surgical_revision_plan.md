# Surgical Camera-Ready Revision Plan

## Source of truth

- Source manuscript: `D:\MINDAID-V2\MIND-AID-CAMERA-READY\MIND-AID_Camera-Ready_Paper.docx`
- Preserve the original title exactly: **MIND-AID: An explainable machine learning approach for early depression detection from user-generated text**
- Preserve the original authors, affiliations, section order, tables, figures, references, and accepted Phase 1/Phase 2 result values unless a reviewer-required factual correction is explicitly approved.
- The source DOCX remains unchanged. A revised copy will be created only after this plan is approved.

## Items not available

- Reviewer Comments 1–2 were not supplied. No edits or claims will be invented for them.
- The final revision will be marked incomplete for those two items unless the comments are recovered.

## Surgical changes by reviewer item

### Comment 3 — Data leakage and overestimated performance

**Original location:** Section 3, “Dataset description,” and “Training and evaluation procedure.”

**Planned edit:**

1. Replace the broad statement that duplicate posts were removed with an evidence-qualified statement:
   - exact normalized duplicate removal is applied in the corrected split/training workflow;
   - the original packaged dataset contained repeated normalized texts and therefore the original internal result should not be described as duplicate-free unless its exact preprocessing run can be reproduced;
   - the strict external evaluation is the primary leakage-controlled evidence.
2. Add that TF–IDF vocabulary and weights are fitted on the training partition only and applied to validation/test text by `transform`.
3. Add strict external controls:
   - test size: 715;
   - exact normalized train–test overlap: 0;
   - duplicate external-test rows: 0;
   - character n-gram similarity audit maximum: 0.621, with no pair at or above 0.80.
4. Add a limitation that the internal split is not user-independent because author identifiers are unavailable.

**Do not change:** Original title, table layout, or accepted Phase 1/Phase 2 values unless the author decides that the original values must be replaced by the newly regenerated strict-evaluation values.

**Approved presentation:** prose only. Keep all existing tables and figures unchanged; do not add a new Comment 3 table.

### Comment 4 — Novelty and research contribution

**Original location:** End of Section 1 and opening of Section 2.

**Planned edit:**

Add one concise clarification that the contribution is the integrated comparison and leakage-aware evaluation protocol, not a new TF–IDF, LinearSVC, DistilBERT, or SHAP algorithm. Preserve the existing contribution narrative otherwise.

### Comment 5 — Explainability validation

**Original location:** Section 3, “Explainability,” and Figure 3 caption/discussion.

**Planned edit:**

1. Distinguish statistically influential lexical features from clinically validated indicators.
2. Add the verified audit findings for the strict external set:
   - 715 samples;
   - mean top-keyword Jaccard stability: 0.834;
   - mean absolute probability change after masking top keywords: 0.152;
   - prediction flip rate after masking: 23.4%;
   - 198 misclassified examples.
3. Add one representative correct and one representative misclassified interpretation without presenting “depression,” “anxiety,” or “help” as diagnostic evidence.

**Implementation update:** A separate reproducible fine-tuned DistilBERT and token-level SHAP workflow is now present in the repository. It uses `shap.Explainer` with `shap.maskers.Text` and the Partition algorithm, explains class-1 probability, and writes the model metrics and token-attribution artifact to:

- `research/reports/metrics_distilbert_dreaddit_strict.json`
- `research/reports/shap_distilbert_dreaddit_strict.json`

The generated strict external run uses 715 samples, achieves 0.7916 accuracy, 0.7914 weighted F1, and 0.8863 ROC-AUC, and produces token-level SHAP values for a documented 10-example representative subset. The SHAP probability additivity error is at most 2.22e-16. The audited examples produce readable, deduplicated content tokens after stopword filtering (for example, `helpless`, `die`, `wanna`, `swear`, and `scared` for one positive example). These values are new strict-evaluation evidence and must not silently replace the accepted Phase 2 table. The paper may retain the DistilBERT/SHAP claim only with the implementation details and evidence now recorded; it must not imply that the deployed TF–IDF API model is the DistilBERT model.

### Comment 6 — Fairness of model comparison

**Original location:** Phase 1/Phase 2 results and comparative discussion.

**Planned edit:**

Add a compact resource paragraph or table, without replacing the accepted performance tables:

- hardware: Windows 11, AMD Ryzen 5 7535HS, 12 logical processors, 15.32 GB RAM;
- software: Python 3.13.3, NumPy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, joblib 1.5.3, sentence-transformers 5.4.1, PyTorch 2.11.0;
- strict TF–IDF model: 4.47 MB classifier, 2.84 MB vectorizer, 111,583 features, 0.154–0.177 ms/sample inference;
- transformer comparator: 384-dimensional embeddings, 40.12 s embedding-plus-logistic-regression fitting, 14.28 ms/sample encoding;
- repeated seeds: 13, 42, and 73;
- no significance test claimed; peak resident memory was not measured.

### Comment 7 — Clinical relevance and ethics

**Original location:** Dataset description, Introduction, and Conclusion/Future Scope.

**Planned edit:**

Add concise language stating that dataset labels are annotations from the source dataset, not clinician-confirmed psychiatric diagnoses. Add that the system is research/decision support only, and real deployment would require privacy safeguards, consent or lawful data governance, bias auditing, escalation protocols, qualified human review, and prospective clinical validation.

### Comment 8 — Robustness and real-world applicability

**Original location:** Results discussion and Conclusion/Future Scope.

**Planned edit:**

Add a short robustness paragraph using the strict external audit:

- punctuation removal: 0% prediction flips;
- spelling noise: 2.0% flips;
- slang/abbreviation substitutions: 2.9% flips;
- word dropout: 9.8% flips;
- stopword removal while preserving negations: 27.6% flips;
- 20-token truncation: 31.3% flips and F1 = 0.6673.

State explicitly that sarcasm, naturally occurring spelling variation, and indirect emotional expressions were not independently annotated or evaluated.

### Comment 9 — Comparison with existing research

**Original location:** “Comparative discussion” and Table 4.

**Planned edit:**

Retain Table 4, but add a sentence immediately before or after it stating that values from different studies are contextual only because datasets, label policies, preprocessing, and splits differ. State that the fair comparisons are the models evaluated on the same test split in this study.

### Comment 10 — Reproducibility and implementation details

**Original location:** Section 3 and the final paragraph of Section 5.

**Planned edit:**

Add a compact reproducibility paragraph covering preprocessing, exact deduplication, TF–IDF fitting order, model settings, seeds, environment, hardware, and repository commands/artifacts.

Update the SHAP implementation statement to distinguish the fine-tuned DistilBERT SHAP audit from the deployed TF–IDF coefficient explanation path. Cite the reproducibility commands and artifacts above.

## Formatting and language-only corrections

- Correct “post-analysis interpretability interpretation” to “post-analysis interpretability.”
- Correct “The issue with comparing results” to more precise academic wording.
- Remove the duplicated period in “practice..”
- Clarify class ordering for precision, recall, and F1 as class 0 (non-depression) followed by class 1 (depression), or macro/weighted averaging where applicable.
- Ensure every new paragraph refers to the relevant table or figure without renumbering existing figures unnecessarily.

## Approval gates before DOCX editing

1. Confirm whether the original accepted Phase 1/Phase 2 metrics remain unchanged in the camera-ready manuscript.
   2. Confirm the final wording for the original SHAP claim. Reproducible DistilBERT/SHAP evidence is now available, but the accepted Phase 2 metrics and the newly regenerated strict external metrics are different experiments.
3. Approve the insertion locations and compact wording.
4. Only then create a separate revised DOCX; the original DOCX will not be overwritten.
