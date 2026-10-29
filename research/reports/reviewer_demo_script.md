# MIND-AID Reviewer Demo Script

## 30-Second Version
MIND-AID is a depression-risk screening research prototype built for external validity, not just in-sample accuracy. The key result is that a model trained on one corpus performs weakly on Dreaddit test data, but strict domain-adapted training with overlap checks substantially improves external performance. We also added a transformer sentence-embedding baseline and bootstrap confidence intervals, which makes the comparison more credible for review.

## 60-Second Version
MIND-AID is a reproducible depression-risk screening framework that takes text, and optionally image cues, to produce a risk label, confidence score, and explanation keywords. The important research question is generalization: can a model trained in one domain still work on another?

We tested two main scenarios on Dreaddit external data. First, a cross-domain baseline trained on internal data performed weakly. Second, a strict deduplicated adaptation using Dreaddit train-plus-validation significantly improved external metrics. The upgraded evaluation also includes a transformer sentence-embedding baseline, so the paper is not comparing only against a simple TF-IDF model. On top of that, the latest report includes 95% bootstrap confidence intervals for the headline metrics, which makes the results more statistically defensible.

The main message for reviewers is simple: in-sample scores are not enough, leakage control matters, and strict adaptation can recover a lot of lost performance under domain shift.

## 2-Minute Version
MIND-AID is a practical research system for early depression-risk screening from social text. It includes text preprocessing, calibrated linear modeling, optional multimodal fusion, explainability, and automated visual reporting.

The project was designed to answer a specific reviewer-facing question: how much does performance drop when a model leaves its training domain, and how much can we recover with strict adaptation?

The strongest evidence comes from external validation on Dreaddit test data. The cross-domain baseline trained on the internal split shows weak external generalization. After training on a deduplicated strict Dreaddit split, the model improves substantially on external accuracy, F1, ROC-AUC, and Brier score. The upgraded pipeline also adds a transformer sentence-embedding baseline, which improves the comparison set and strengthens the paper’s credibility. Finally, bootstrap confidence intervals were added so the report includes uncertainty, not only point estimates.

If I were summarizing the contribution in one line, I would say: MIND-AID shows that a leakage-aware, externally validated workflow produces much more believable results than in-sample metrics alone.

## Reviewer Talking Points
1. External validation is the primary claim block.
2. Strict train-test overlap was checked and found to be zero.
3. The transformer baseline gives a stronger modern benchmark.
4. Bootstrap confidence intervals improve statistical credibility.
5. Fusion is intentionally skipped on Dreaddit external tests because the dataset has no image column.
6. The project is a screening aid, not a diagnostic tool.

## Key Numbers to Mention
1. Cross-domain baseline accuracy: 0.5231
2. Strict adapted text-only accuracy: 0.7217
3. Strict adapted text-only ROC-AUC: 0.8257
4. Transformer baseline accuracy: 0.7524
5. Transformer baseline ROC-AUC: 0.8474
6. External test size: 715 rows
7. Strict train-external overlap: 0 rows
