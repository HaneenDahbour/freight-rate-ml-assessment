# 2-3 minute Loom outline

**0:00-0:25 - The data and task.** Show the repository and explain that 48,000 labeled loads cover January-October 2025; 12,000 November-December loads have hidden rates. The second output is a fixed route on each day of December.

**0:25-0:50 - Data quality.** Show `artifacts/run.json`: 292 negative development weights, 300 missing weights, 374 missing market indices. Explain absolute weight and fold-fitted imputation. The fixed December scenario lacks `market_index` and `quote_signal`, so a separate model uses its available features.

**0:50-1:35 - Model and validation.** Show `src/pipeline.py` and the backtest table. July and August select among median, Ridge, histogram boosting, CatBoost, and a market-signal CatBoost using dollar MAE. September and October are forward checks after selection. Quote the measured figures in the report, including the October variability. The 12,000-row score is hidden and cannot be claimed.

**1:35-2:10 - Reproducibility and outputs.** Run or show `src/verify_outputs.py`, the 12,000-row CSV, and the December chart in the PDF. Show the run manifest with input hashes and the frozen model used by `src/predict.py`. Mention a new dataset can be scored without fitting.

**2:10-2:35 - Operational next steps.** Explain that a real batch service would version model/data artifacts, monitor input changes and delayed-label error, and retrain behind a forward-evaluation gate. Close with the report and repository run instructions.

Record the Loom yourself and add its accessible URL to the submission. Do not present this outline as a recorded video.
