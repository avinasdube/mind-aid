# MIND-AID: Domain-Shift-Aware Depression Risk Screening from Social Text with Strict External Validation

## Abstract
Automated depression-risk screening from social text is promising for early triage, but real-world deployment is limited by domain shift, evaluation leakage, and the ethical risks of inferring mental-health status from sensitive language. We present MIND-AID not as a new classification algorithm or clinical diagnostic system, but as an integrated, leakage-aware evaluation framework that combines calibrated text classification, a transformer sentence-embedding comparator, optional multimodal fusion, lightweight local attribution, robustness perturbation checks, and strict external validation safeguards. We evaluate two settings on Dreaddit external test data (n = 715): (1) a cross-domain baseline trained on an internal dataset and (2) strict domain-adapted training with duplicate normalization before model fitting. External text-only performance improves from accuracy 0.5231, balanced accuracy 0.5075, F1 0.3738, ROC-AUC 0.6483, and Brier score 0.4531 to accuracy 0.7231, balanced accuracy 0.7191, F1 0.7183, ROC-AUC 0.8257, and Brier score 0.1794 after strict adaptation, with zero exact overlap between strict training and external test sets. The transformer baseline provides an independent modern comparison, while bootstrap confidence intervals quantify uncertainty in the headline metrics. These results show that in-sample scores can overestimate deployment readiness and that leakage-aware adaptation improves out-of-domain reliability without establishing clinical validity. MIND-AID is released as a reproducible pipeline with command-level benchmarking, chart generation, and visual reporting.

## Index Terms
Mental health NLP, domain shift, external validation, depression risk screening, calibration, explainable AI

## I. Introduction
Language-based depression-risk screening systems are increasingly explored for decision support. However, many reported systems are evaluated in settings that do not reflect deployment, where domain mismatch and hidden overlap can cause severe performance collapse. In practice, robust external performance and transparent evaluation controls are more important than high in-sample accuracy.

This work investigates a deployment-relevant question: how much cross-domain degradation occurs when a model trained on one corpus is tested on another, and how much can be recovered using strict leakage-aware adaptation? To answer this, we develop MIND-AID, an end-to-end framework that emphasizes external validity, calibration quality, and reproducibility.

The main contributions are methodological rather than algorithmic:
1. An integrated risk-screening workflow that connects calibrated TF-IDF inference, an independently implemented transformer comparator, optional confidence-aware multimodal fusion, and local coefficient-based attribution.
2. A leakage-aware evaluation protocol that removes exact normalized duplicates before fitting, checks train–test overlap, separates in-sample diagnostics from headline claims, and reports external uncertainty and perturbation sensitivity.
3. A reproducible evidence package showing how domain shift changes conclusions on Dreaddit and how strict adaptation affects external discrimination and calibration.

## II. Related Work
Prior work in mental health NLP demonstrates that social media language can support risk inference, but generalization across datasets remains a key limitation. Distribution shift, label policy differences, lexical drift, and overlap leakage can all produce inflated estimates of model quality. Calibration research further shows that probability quality is essential when model outputs are used for triage-oriented decisions.

MIND-AID is positioned as an applied evaluation framework focused on robust external claims rather than benchmark-only optimization. Its novelty is the integration and operationalization of these controls in one reproducible decision-support workflow; TF-IDF, LinearSVC, sentence embeddings, confidence weighting, and local feature attribution are established techniques and are not claimed as algorithmic innovations. Accuracy values from earlier studies are treated as contextual evidence only because their datasets, label policies, preprocessing, and evaluation splits are not identical to ours.

## III. Method
### A. System Components
MIND-AID includes:
1. Text preprocessing and normalization.
2. TF-IDF feature extraction with unigrams and bigrams.
3. Calibrated linear classifier for probability outputs.
4. Transformer sentence-embedding baseline for comparison.
5. Optional facial-emotion branch with confidence-aware fusion.
6. Keyword-level explanation module with an external explanation audit.
7. Automated charting and HTML visual reporting.

### B. Prediction and Calibration
The text branch produces risk probability and label. Calibration is applied to improve probability reliability. The API returns risk score, confidence, and explanation features suitable for audit and triage support.

### C. Fusion Policy
When valid image evidence exists, confidence-aware weighting combines text and face channels. If image evidence is missing or low confidence, the system remains text-dominant to avoid injecting noisy modality signals.

### D. Explainability Validation
Explanations are local TF-IDF feature attributions computed as feature value multiplied by the learned linear coefficient. On the strict external test set, we measured top-keyword stability after punctuation perturbation and a masking-based faithfulness proxy. Across 715 samples, mean top-keyword Jaccard stability was 0.834, masking the reported keywords changed the positive probability by 0.152 on average, and the predicted label changed in 23.4% of cases. These are model-behavior measures, not clinical validation. Terms such as “anxiety,” “help,” or “feelings” are therefore interpreted as statistically influential features in this dataset, not as evidence of a diagnosis.

## IV. Experimental Design
### A. Data and Evaluation Scenarios
We evaluate two external-test scenarios on Dreaddit test data (715 samples):
1. Cross-domain baseline: train on internal split, test on Dreaddit test.
2. Strict adaptation: train on deduplicated Dreaddit train-plus-validation, test on Dreaddit test.

Strict overlap checks are performed before benchmarking. Observed overlap between strict train and strict external test is zero. The cross-domain TF-IDF baseline, strict-adaptation TF-IDF model, and sentence-transformer comparator are all evaluated on the same 715-row Dreaddit test split; these are the primary fair comparisons.

The binary labels are dataset annotations used for supervised benchmarking; they are not clinician-confirmed psychiatric diagnoses. Accordingly, all reported outputs are risk-screening predictions under the dataset's labeling policy and must not be interpreted as diagnostic evidence.

### B. Metrics
Primary external metrics:
1. Accuracy
2. Balanced accuracy
3. F1 score
4. ROC-AUC
5. Brier score
6. 95% bootstrap confidence intervals for the above metrics
7. Repeated-seed resource benchmarking, including training time, inference latency, artifact size, and feature count

Internal in-sample results are reported only as diagnostics and are excluded from headline claims.

## V. Results
### A. External Validation (Primary Evidence)
| Metric | Cross-Domain Baseline | Strict Adaptation | Transformer Baseline | Delta |
| --- | ---: | ---: | ---: |
| Accuracy | 0.5231 | 0.7231 | 0.6182 | +0.2000 |
| Balanced Accuracy | 0.5075 | 0.7191 | 0.6180 | +0.2116 |
| F1 | 0.3738 | 0.7183 | 0.5832 | +0.3445 |
| ROC-AUC | 0.6483 | 0.8257 | 0.6628 | +0.1774 |
| Brier Score | 0.4531 | 0.1794 | 0.3036 | -0.2737 |

Strict adaptation yields gains across discrimination and calibration metrics. The baseline model under domain mismatch exhibits strong class imbalance behavior on the external set, while strict adaptation improves class balance and stability. The result is evidence for the evaluation protocol and its domain-shift analysis, not evidence that the underlying linear classifier is novel.

The transformer baseline is a stronger modern comparison than the TF-IDF-only cross-domain baseline, but strict adaptation still performs best overall on external reliability and calibration.

Results from prior studies are not used to establish a direct accuracy ranking because their experimental conditions differ. The same-dataset comparisons above provide stronger evidence for the relative effects of domain adaptation and representation choice within the present evaluation protocol.

### B. Robustness Summary
On the strict external test set, punctuation removal and whitespace collapse caused no prediction flips. Word dropout reduced F1 to 0.7123 with a 9.8% flip rate; synthetic spelling noise produced a 1.96% flip rate, and slang/abbreviation substitutions produced a 2.94% flip rate. Removing stopwords while preserving common negations caused a 27.6% flip rate, while truncating texts to 20 tokens caused a 31.3% flip rate and reduced F1 to 0.6673. These results indicate sensitivity to context reduction and preprocessing choices. Sarcasm, naturally occurring spelling variation, and indirect emotional expressions were not independently annotated or evaluated, so robustness to those phenomena remains unresolved.

The updated report additionally includes 95% bootstrap confidence intervals, which strengthen the claim that the observed improvements are not just point-estimate noise.

### C. In-Sample Diagnostics
Near-ceiling in-sample scores are observed in both scenarios, reinforcing that external validation is required for realistic claims.

### D. Explainability Audit
The explanation audit found 198 misclassified external examples. In a representative false positive, terms such as “feelings,” “need,” and “boyfriend” contributed to a positive prediction for a non-risk-labeled post; in a representative false negative, the model relied on context-specific terms including “school,” “job,” and “lease.” These examples illustrate both the usefulness and the limits of lexical attribution: the features expose model behavior, but they do not establish clinically meaningful symptoms.

### E. Resource and Seed Comparison
On the reported Windows 11 environment (Python 3.13.3, NumPy 2.4.4, pandas 3.0.2), three deterministic seeds produced identical strict external metrics: accuracy 0.7231, balanced accuracy 0.7191, weighted F1 0.7183, and ROC-AUC 0.8257. Training required 13.57–14.36 s per run; inference required 0.154–0.177 ms per sample. The serialized classifier and vectorizer occupied 4.47 MB and 2.84 MB, respectively, with 111,583 TF-IDF features. The sentence-transformer comparator used 384-dimensional embeddings, required 40.12 s for embedding plus logistic-regression fitting, and required 14.28 ms per external sample for encoding. Its cached encoder size was not measured. Peak resident memory was not measured, and no paired significance test is claimed.

## VI. Discussion
This study shows that high in-sample performance can mask substantial deployment risk. Under strict external evaluation, cross-domain performance is weak, but leakage-aware adaptation provides meaningful recovery. The observed improvements in ROC-AUC and Brier score indicate better ranking quality and probability reliability, both important in triage-oriented workflows.

From a research perspective, the strongest contribution is methodological discipline: explicit anti-leakage checks, external-first claims, calibration and perturbation reporting, and reproducible evidence packaging. The explanation module identifies statistically influential TF-IDF features for individual predictions; these features are not presented as clinically validated indicators.

## VII. Limitations and Ethics
1. The strongest benchmark is text-only because the Dreaddit external split does not provide image columns; fusion claims are therefore not made for that benchmark.
2. MIND-AID is a screening support system, not a clinical diagnostic tool.
3. The dataset labels are not clinical diagnoses, and the text may contain sensitive personal or mental-health information. Any real deployment would require lawful data governance, privacy protection, consent or an appropriate legal basis, bias auditing, escalation protocols, and human oversight.
4. False positives may create stigma or unnecessary intervention, while false negatives may delay support; model outputs must therefore remain advisory and subject to qualified review.
5. Additional out-of-domain datasets and prospective clinical validation are needed to establish transportability or safety. Comparisons with prior publications remain contextual unless their datasets and protocols are reproduced.
6. Explanation stability and masking sensitivity are preliminary model-faithfulness checks; human and clinical validation remain necessary.

## VIII. Reproducibility and Artifacts
Key generated artifacts include:
1. research/reports/research_evaluation_dreaddit.json
2. research/reports/research_evaluation_dreaddit_adapted_strict.json
3. research/reports/external_dataset_validation_dreaddit_adapt_strict.json
4. research/reports/metrics_dreaddit_adapt_strict.json
5. research/reports/dreaddit_strict/visual_report.html
6. research/reports/explainability_audit_dreaddit_adapted_strict.json
7. research/reports/model_benchmark_dreaddit_adapted_strict.json

The reported environment used Windows 11, Python 3.13.3, 12 logical processors, and 15.32 GB RAM. Package versions were NumPy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, joblib 1.5.3, sentence-transformers 5.4.1, PyTorch 2.11.0, Matplotlib 3.10.8, Seaborn 0.13.2, and FastAPI 0.136.1. The text model uses lowercase URL removal, punctuation removal, whitespace normalization, exact normalized-text deduplication, TF-IDF unigrams and bigrams, class-balanced LinearSVC, and five-fold sigmoid calibration. The default split and bootstrap seed is 42; repeated resource runs use seeds 13, 42, and 73. The transformer comparator uses all-MiniLM-L6-v2, batch size 64, normalized embeddings, and LogisticRegression with max_iter=2000 and class_weight=balanced.

The current implementation does not execute SHAP. Its explanation method is the deterministic local attribution “TF-IDF feature value × learned linear coefficient”; therefore, SHAP-specific configuration is not applicable and no SHAP claim is made. All benchmark steps are executable through the `app.py` commands documented in the project README, including `split-dataset`, `train`, `evaluate`, `validate-external`, `explainability-audit`, and `benchmark-model`.

## IX. Conclusion
MIND-AID demonstrates that external, leakage-aware evaluation changes the practical conclusions of depression-risk NLP systems. Cross-domain baseline performance is weak despite excellent in-sample diagnostics, while strict adaptation significantly improves external reliability. The framework prioritizes transparent claims, reproducibility, and deployment-aware evaluation practices, but its predictions remain research-only risk estimates rather than diagnoses or substitutes for professional care.

## Acknowledgment
This manuscript reports a research prototype for decision support. No clinical deployment claim is made.

## References (IEEE style)
[1] M. Turcan and K. McKeown, "Dreaddit: A Reddit Dataset for Stress Analysis in Social Media," in Proceedings of the 10th International Workshop on Health Text Mining and Information Analysis (LOUHI), 2019.

[2] C. Coppersmith, K. Dredze, and C. Harman, "Quantifying Mental Health Signals in Twitter," in Proceedings of the Workshop on Computational Linguistics and Clinical Psychology (CLPsych), 2014.

[3] C. Coppersmith, M. T. Just, and G. T. Nguyen, "Social Media and Clinical Psychology," in Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (HLT-NAACL) Workshop on CLPsych, 2015.

[4] A. T. Beck, R. A. Steer, and G. K. Brown, Manual for the Beck Depression Inventory-II. San Antonio, TX, USA: Psychological Corp., 1996.

[5] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," in Proceedings of the 34th International Conference on Machine Learning (ICML), 2017, pp. 1321-1330.

[6] J. Platt, "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods," in Advances in Large Margin Classifiers, A. Smola, P. Bartlett, B. Scholkopf, and D. Schuurmans, Eds. Cambridge, MA, USA: MIT Press, 1999, pp. 61-74.

[7] A. Niculescu-Mizil and R. Caruana, "Predicting Good Probabilities With Supervised Learning," in Proceedings of the 22nd International Conference on Machine Learning (ICML), 2005, pp. 625-632.

[8] T. Fawcett, "An Introduction to ROC Analysis," Pattern Recognition Letters, vol. 27, no. 8, pp. 861-874, 2006.

[9] T. Saito and M. Rehmsmeier, "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets," PLOS ONE, vol. 10, no. 3, e0118432, 2015.

[10] M. T. Ribeiro, S. Singh, and C. Guestrin, "Why Should I Trust You?: Explaining the Predictions of Any Classifier," in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2016, pp. 1135-1144.

[11] M. A. Hall, "Correlation-Based Feature Selection for Machine Learning," Ph.D. dissertation, Univ. Waikato, Hamilton, New Zealand, 1999.

[12] T. B. Brown et al., "Language Models are Few-Shot Learners," in Advances in Neural Information Processing Systems (NeurIPS), 2020.

[13] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in Proceedings of NAACL-HLT, 2019, pp. 4171-4186.

[14] K. Ethayarajh, "How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings," in Proceedings of EMNLP-IJCNLP, 2019, pp. 55-65.

[15] M. A. Haque, A. Milstein, and H. Fei, "Do We Need Clinical Labels for Mental Health Prediction from Social Media?," in Proceedings of EMNLP Findings, 2021.

Note: Verify venue-specific formatting and bibliographic metadata (page numbers, DOI, workshop volume identifiers) before final submission.
