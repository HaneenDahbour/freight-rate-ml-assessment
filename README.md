# Freight rate prediction assessment

This repository trains freight-rate regression models on January-October 2025 labeled loads, evaluates future-month holdouts, then predicts 12,000 November-December loads and the fixed 31-day December scenario. No target values from the 12,000 final loads are available locally.

## Run

Use Python 3.11 or 3.12. Install `requirements.txt`, then place the assessment's original CSVs in `data/` under the names `train_test.csv`, `validation.csv`, `validation_predictions_template.csv`, and `december_chart_inputs.csv`. The downloaded hyphenated filenames are renamed only for this directory. From the repository root:

```bash
python -m pip install -r requirements.txt
python src/pipeline.py
python src/verify_outputs.py
```

Outputs are `outputs/validation_predictions.csv`, `outputs/december_chart_inputs.csv`, `artifacts/backtests.csv`, and `artifacts/run.json`. Run the supplied assessment scorer to validate the two CSVs and create `scorer_results/candidate_december.png`:

```bash
python score.py --predictions outputs/validation_predictions.csv --december-predictions outputs/december_chart_inputs.csv
```

The original `score.py` passed on both outputs. `verify_outputs.py` adds an independent check against the input files and also creates an optional local chart; the report embeds the supplied scorer's chart.

## Modeling contract

Each fold trains on dates strictly before its held-out month. July and August select a model by mean dollar MAE; September and October report untouched forward checks. The model is then refit on all 48,000 labeled rows. Every fold independently learns the city-to-coordinate lookup and numeric imputation from its own training history. The target is posted dollars per mile, transformed back to dollars for all scoring and outputs. Candidate models are a constant median, Ridge, histogram gradient boosting, and CatBoost when installed.

The load-validation model considers `quote_signal` and `market_index`, which are supplied for those loads. The fixed December file lacks both fields, so a second estimator is selected using the shared feature contract: pickup, delivery, equipment, distance, weight, derived coordinates, and calendar features. Both selections use only July-August backtests, and neither draws on the hidden targets. The saved model bundle routes each input schema to its appropriate estimator. Negative source weights are converted to absolute values; missing weight is imputed inside the fitted estimator. Original CSVs stay untouched. Only dates change in the 31-row December scenario.

Run metadata includes SHA-256 hashes of the four inputs, feature names, selected models, seed, and data-quality counts. `artifacts/backtests.csv` holds the chronological evidence. The frozen bundle supports independent batch inference:

```bash
python src/predict.py --model artifacts/model.joblib --input data/validation.csv --output outputs/replayed.csv
```

That command reproduces the 12,000 predictions exactly. A production environment would also retain immutable artifacts, schedule batch jobs, monitor input shifts and errors after labels arrive, and gate promotion on future-month checks.

## Expected submission

Share this repository, `outputs/validation_predictions.csv`, `report/assessment_report.pdf`, and a 2-3 minute Loom URL recorded by the applicant. The report includes the chart. Supply the original scorer run result once `score.py` is available.
