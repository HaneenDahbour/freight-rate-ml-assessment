"""Generate assessment PDF from measured backtests and the verified chart."""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report/assessment_report.pdf"
W, H = A4
NAVY = colors.HexColor("#073746")
TEAL = colors.HexColor("#087984")
GRAY = colors.HexColor("#415460")
PALE = colors.HexColor("#EAF3F3")
pdfmetrics.registerFont(TTFont("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))


def line(c, txt, x, y, size=10, color=GRAY, bold=False):
    c.setFillColor(color)
    c.setFont("DejaVu-Bold" if bold else "DejaVu", size)
    c.drawString(x, y, txt)


def wrapped(c, txt, x, y, width=492, leading=15, size=10):
    words = txt.split()
    row = ""
    for word in words:
        next_row = (row + " " + word).strip()
        if stringWidth(next_row, "DejaVu", size) > width and row:
            line(c, row, x, y, size)
            y -= leading
            row = word
        else:
            row = next_row
    if row:
        line(c, row, x, y, size)
        y -= leading
    return y


def page(c, section, page_no):
    c.setFillColor(NAVY)
    c.rect(0, H-84, W, 84, fill=1, stroke=0)
    line(c, "FREIGHT RATE PREDICTION", 49, H-47, 17, colors.white, True)
    line(c, section.upper(), 49, H-68, 9, colors.HexColor("#B4DDDE"), True)
    c.setStrokeColor(colors.HexColor("#D7E6E7"))
    c.line(49, 47, W-49, 47)
    line(c, "Spotter ML Engineer assessment | development evidence", 49, 32, 8)
    line(c, str(page_no), W-58, 32, 8)


def build():
    scores = pd.read_csv(ROOT / "artifacts/backtests.csv")
    info = json.loads((ROOT / "artifacts/run.json").read_text())
    selected = info["selected_model"]
    c = canvas.Canvas(str(OUT), pagesize=A4)
    c.setTitle("Freight Rate Prediction - Validation and December Scenario")
    page(c, "From data to predictions", 1)
    y = H-119
    line(c, "An auditable batch prediction pipeline", 49, y, 14, NAVY, True); y -= 29
    y = wrapped(c, "Goal: estimate the posted USD rate for each freight load. I treated this as a future-load regression task: inspect the labeled history, repair identifiable data problems, compare models on later months, fit on all labeled history, then produce and verify the requested CSVs.",49,y); y -= 13
    line(c, "1  Understand one source record",49,y,12,NAVY,True); y -= 22
    y = wrapped(c, "Example TR-000001: Richmond to Baltimore, 274.3 miles, Dry Van, 30,658 lb, January 1, 2025; market_index 0.95684 and quote_signal 2.39595. Its known posted_rate is $645.41, or approximately $2.353 per mile. The validation set has the same prediction task but no posted_rate.",49,y); y -= 13
    line(c, "2  Clean without leaking the future",49,y,12,NAVY,True); y -= 22
    y = wrapped(c, f"In 48,000 labeled rows, {info['negative_weight_rows_corrected']} weights were negative; I used their magnitude as pounds. There were {info['missing_weight_rows']} missing weights and {info['missing_market_index_rows']} missing market indices. Missing values were handled by the relevant fitted model; numeric market-signal fill values were learned from each fold's past training rows. Raw CSVs were not overwritten.",49,y); y -= 13
    line(c, "3  Represent the load",49,y,12,NAVY,True); y -= 22
    y = wrapped(c, "Inputs include route cities, distance, equipment, weight, calendar date, and training-derived geographic coordinates. For the main validation loads, market_index and quote_signal are also supplied. I excluded load_id: it identifies a row but is not a freight-price signal. The target is dollars per mile; predictions are multiplied by each load's distance to return dollars.",49,y); y -= 13
    line(c, "4  Compare, select, and fit",49,y,12,NAVY,True); y -= 22
    y = wrapped(c, "I compared a median baseline, Ridge, histogram gradient boosting, and CatBoost using July and August forward-month MAE in dollars. CatBoost with market signals had the lowest two-month mean ($138.29). I checked this selected family on September and October, then refit on all 48,000 labeled loads. December has no market signals, so its fixed scenario uses the separately selected shared-feature CatBoost.",49,y); y -= 13
    line(c, "5  Deliver and verify",49,y,12,NAVY,True); y -= 22
    y = wrapped(c, "The batch run saved a reusable model, 12,000 positive load predictions matched to the template IDs, 31 daily December predictions, and a manifest of input hashes and measured backtests. The supplied score.py validated both files and generated the chart in this report. Final hidden-set error is calculated only after submission.",49,y)
    c.showPage()
    page(c, "Method and data", 2)
    y = H-119
    line(c, "Objective and evaluation design", 49, y, 14, NAVY, True); y -= 26
    y = wrapped(c, "Predict posted freight rate in USD for 12,000 unlabeled November-December 2025 loads. The labeled 48,000-row development set covers January-October 2025. The 12,000 hidden targets are unavailable, so all dollar error figures below are historical backtests.", 49, y); y -= 13
    line(c, "Chronological protocol", 49, y, 12, NAVY, True); y -= 22
    for text in ["Selection: train before July, score every July row; train before August, score every August row. Select the lowest mean dollar MAE across those months.",
                 "Independent checks: train before September and score September; train before October and score October, using the previously selected model family.",
                 "Final fit: train the selected estimator on all labeled January-October rows, then predict the separate 12,000-row set and the fixed December scenario."]:
        y = wrapped(c, text, 61, y, 473); y -= 10
    line(c, "Feature availability and data quality", 49, y, 12, NAVY, True); y -= 22
    for text in ["The 31 December rows include only route, distance, equipment, weight and date. The separate December model uses only these shared attributes. The 12,000-row validation model can use market_index and quote_signal, which are available in that file.",
                 f"Development data contains {info['negative_weight_rows_corrected']} negative weights, {info['missing_weight_rows']} missing weights, and {info['missing_market_index_rows']} missing market indices. Weight magnitude is used; imputation is fitted on training rows within each fold.",
                 "December coordinates are recovered through a city lookup learned from labeled development data. Date, geography and route fields share the same transformation for both output files."]:
        y = wrapped(c, text, 61, y, 473); y -= 10
    c.showPage()
    page(c, "Model evidence", 3)
    y = H-119
    line(c, "Candidate selection and forward checks", 49, y, 14, NAVY, True); y -= 28
    line(c, "Model", 56, y, 9, NAVY, True)
    for x, title in [(306,"Jul MAE"),(378,"Aug MAE"),(454,"Mean")]: line(c,title,x,y,9,NAVY,True)
    y -= 17
    candidates = scores[scores.period.isin(["2025-07","2025-08"]) & ~scores.model.str.startswith("final_check:")]
    for name, group in candidates.groupby("model", sort=False):
        vals = group.set_index("period").mae_usd
        if name == selected:
            c.setFillColor(PALE); c.rect(49,y-5,W-98,19,fill=1,stroke=0)
        label = {"median_rate_per_mile":"Median rate/mile", "ridge_rate_per_mile":"Ridge rate/mile",
                 "hist_boost_rate_per_mile":"Hist. boosting rate/mile", "catboost_rate_per_mile":"CatBoost rate/mile",
                 "catboost_market_signals":"CatBoost + market signals"}[name]
        line(c,label,56,y,9,NAVY,name==selected)
        for x,value in [(306,vals["2025-07"]),(378,vals["2025-08"]),(454,vals.mean())]: line(c,f"${value:,.2f}",x,y,9,NAVY)
        y -= 23
    y -= 17
    line(c, "Validation model: " + selected.replace("_", " "), 49, y, 11, TEAL, True); y -= 22
    line(c, "December model: " + info["december_model"].replace("_", " "),49,y,9,GRAY); y -= 24
    final = scores[scores.model.str.startswith("final_check:")].set_index("period")
    for period in ["2025-09", "2025-10"]:
        row = final.loc[period]
        y = wrapped(c, f"{period} forward check ({int(row.holdout_rows):,} rows): MAE ${row.mae_usd:,.2f}; RMSE ${row.rmse_usd:,.2f}; median absolute error ${row.median_ae_usd:,.2f}.",49,y); y -= 8
    y -= 8
    line(c, "Interpretation", 49, y, 12, NAVY, True); y -= 22
    y = wrapped(c, "October error is materially higher than September. This indicates meaningful temporal variability and limits confidence in a single average backtest score. The validation model was chosen using July-August only; the October result was not used to tune this submitted run.",49,y); y -= 14
    y = wrapped(c, "The model predicts USD per mile, then multiplies by observed positive distance for the required USD output. Actual MAE and RMSE are always computed on dollar rates. Model choice uses MAE; RMSE exposes the effect of a smaller number of large misses.",49,y)
    c.showPage()
    page(c, "December scenario and operations", 4)
    y = H-119
    line(c, "Fixed December prediction scenario",49,y,14,NAVY,True); y-=22
    y = wrapped(c,"Lexington to Fort Wayne | 360 miles | Dry Van | 32,000 lb. Only the date changes from December 1 to 31, 2025. This scenario uses the independently selected shared-feature model; the chart shows predictions, not actual rates.",49,y)
    chart = ImageReader(str(ROOT / "scorer_results/candidate_december.png"))
    c.drawImage(chart,45, y-300, width=W-90, height=280, preserveAspectRatio=True, anchor="c")
    y -= 332
    line(c,"Reproducibility and delivery",49,y,12,NAVY,True); y-=22
    for text in ["One command trains and writes both outputs. The saved bundle routes complete load inputs to the market-signal model and fixed December inputs to the shared-feature model. A batch command scores new CSVs without refitting. The verifier checks IDs, rates, and scenario inputs.",
                 "The run manifest records input SHA-256 fingerprints, feature schema, seed, model choice, and data-quality counts. A real deployment would store immutable model and data versions, schedule batch scoring, monitor input drift and delayed-label errors, and promote a new model only after forward checks.",
                 "The supplied score.py validated 12,000 final predictions and all 31 fixed December predictions and produced the chart above. Spotter calculates final validation metrics after submission."]:
        y=wrapped(c,text,49,y); y-=9
    c.save()
    print(OUT)


if __name__ == "__main__":
    build()
