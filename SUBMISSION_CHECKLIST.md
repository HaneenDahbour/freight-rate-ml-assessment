# Submission checklist

- [x] Reproducible code, pinned CatBoost dependency, run instructions and CI contract checks.
- [x] `outputs/validation_predictions.csv` with exactly 12,000 unique IDs and positive numeric rates.
- [x] `outputs/december_chart_inputs.csv` with 31 unchanged scenario rows and positive numeric rates.
- [x] `report/assessment_report.pdf` with validation design and the chart from the supplied `score.py`.
- [x] Saved model bundle and independent replay of the 12,000 predictions with byte-identical output.
- [x] Supplied `score.py` validates both output CSVs and creates `scorer_results/candidate_december.png`.
- [x] Publish the code in an accessible GitHub repository under the applicant's account. Keep raw assessment CSVs out of a public repository unless sharing is permitted.
- [ ] Record the 2-3 minute Loom using `report/loom_outline.md` and verify that its link can be opened by the reviewer.

The historical September and October metrics in the report are measured on labeled development rows. There is no local final-validation score because the 12,000 target rates were not provided.
