# MIND-AID Final Submission Checklist

## Manuscript
1. Confirm the title is final.
2. Replace all author placeholders.
3. Verify abstract numbers match the latest upgraded evaluation.
4. Make sure the transformer baseline and confidence intervals are mentioned.
5. Ensure the limitations section clearly says this is not a clinical diagnostic tool.

## Results and Claims
1. Use strict external validation as the headline result.
2. Keep in-sample diagnostics clearly labeled as diagnostic only.
3. Mention zero overlap between strict train and strict external test.
4. Include the transformer baseline as the modern comparison point.
5. Do not claim multimodal superiority on Dreaddit external results because there is no image column.

## Figures and Tables
1. Confirm the generalization chart looks clean and readable.
2. Confirm the calibration chart is present.
3. Confirm the ablation chart is present.
4. Confirm the visual HTML report renders all images correctly.
5. Keep all captions short, factual, and publication style.

## Reproducibility
1. Include exact run commands in the appendix or methods.
2. Point to the research folder artifacts, not stale output paths.
3. Verify the external validation JSON includes confidence intervals.
4. Verify the strict evaluation JSON includes the transformer baseline block.
5. Confirm the visual report links are correct.

## Files to Include in Submission Package
1. research/reports/paper_ieee_submission_draft.md
2. research/reports/reviewer_project_flow_explanation.md
3. research/reports/finalization_summary.md
4. research/reports/research_evaluation_dreaddit.json
5. research/reports/research_evaluation_dreaddit_adapted_strict.json
6. research/reports/external_dataset_validation_dreaddit_adapt_strict.json
7. research/reports/dreaddit_strict/visual_report.html
8. research/charts/generalization_comparison.png
9. research/charts/robustness_perturbation_chart.png
10. research/charts/ablation_comparison_dreaddit_adapted_strict.png

## Final Technical Checks
1. Run a syntax check on the updated evaluation module.
2. Re-run training only if the strict model artifacts need refreshing.
3. Re-run evaluation if the training data or model changes.
4. Regenerate the visual report after evaluation.
5. Open the HTML report once before submission to confirm no broken charts.

## Final Review Questions
1. Does the paper explain the domain shift problem clearly?
2. Are the main claims based on external validation rather than in-sample metrics?
3. Do the reported numbers match the latest JSON artifacts?
4. Is the transformer baseline described as a comparison, not the main deployed model?
5. Are all paths in the submission package current and consistent?
