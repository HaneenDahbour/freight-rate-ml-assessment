"""Reproducible, time-aware freight-rate assessment pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
NUM = ["distance", "weight", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
       "month", "weekday", "day", "day_of_year", "route_miles_per_geodesic"]
CAT = ["pickup", "delivery", "equipment"]
FEATURES = NUM + CAT
SIGNALS = ["market_index", "quote_signal"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coordinates(dev: pd.DataFrame) -> dict:
    """Build location lookup solely from labeled development rows."""
    locations = []
    for side in ("pickup", "delivery"):
        part = dev[[side, side + "_lat", side + "_lon"]].copy()
        part.columns = ["city", "lat", "lon"]
        locations.append(part)
    table = pd.concat(locations, ignore_index=True).groupby("city")[["lat", "lon"]].median()
    return table.to_dict("index")


def features(frame: pd.DataFrame, location_lookup: dict) -> pd.DataFrame:
    out = frame.copy()
    dates = pd.to_datetime(out["date"], errors="raise")
    for side in ("pickup", "delivery"):
        for axis in ("lat", "lon"):
            col = side + "_" + axis
            if col not in out:
                out[col] = np.nan
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].fillna(out[side].map({k: v[axis] for k, v in location_lookup.items()}))
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").abs()
    out["distance"] = pd.to_numeric(out["distance"], errors="coerce")
    if out["distance"].isna().any() or (out["distance"] <= 0).any():
        raise ValueError("distance must be positive and present")
    out["month"] = dates.dt.month
    out["weekday"] = dates.dt.weekday
    out["day"] = dates.dt.day
    out["day_of_year"] = dates.dt.dayofyear
    lat1, lat2 = np.radians(out.pickup_lat), np.radians(out.delivery_lat)
    lon1, lon2 = np.radians(out.pickup_lon), np.radians(out.delivery_lon)
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    geo = 3958.8 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    out["route_miles_per_geodesic"] = out.distance / np.maximum(geo, 10)
    for col in CAT:
        out[col] = out[col].fillna("Unknown").astype(str)
    return out[FEATURES]


def signal_features(frame: pd.DataFrame, lookup: dict, medians: dict) -> pd.DataFrame:
    result = features(frame, lookup)
    for column in SIGNALS:
        if column not in frame:
            raise ValueError("required market signal missing: " + column)
        result[column] = pd.to_numeric(frame[column], errors="coerce").fillna(medians[column]).values
    return result


def signal_medians(training: pd.DataFrame) -> dict:
    return {column: float(training[column].median()) for column in SIGNALS}


def make_models():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="ignore")
    pre = ColumnTransformer([("num", numeric, NUM), ("cat", categorical, CAT)])
    models = {
        "median_rate_per_mile": DummyRegressor(strategy="median"),
        "ridge_rate_per_mile": Pipeline([("pre", pre), ("model", Ridge(alpha=30))]),
        "hist_boost_rate_per_mile": Pipeline([
            ("pre", ColumnTransformer([
                ("num", SimpleImputer(strategy="median", add_indicator=True), NUM),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])),
            ("model", HistGradientBoostingRegressor(max_iter=240, max_leaf_nodes=31,
                                                      learning_rate=.055, l2_regularization=5,
                                                      random_state=42))]),
    }
    try:
        from catboost import CatBoostRegressor
        models["catboost_rate_per_mile"] = CatBoostRegressor(
            iterations=550, depth=6, learning_rate=.045, loss_function="RMSE",
            l2_leaf_reg=6, random_seed=42, verbose=False, thread_count=4,
            cat_features=CAT)
    except ImportError:
        pass
    return models


def score(model, x_train, y_train, x_test, y_test, distance):
    model.fit(x_train, y_train)
    predicted = np.maximum(model.predict(x_test) * distance, 1)
    return {"mae_usd": float(mean_absolute_error(y_test, predicted)),
            "rmse_usd": float(np.sqrt(mean_squared_error(y_test, predicted))),
            "median_ae_usd": float(np.median(np.abs(y_test - predicted)))}


def validate_inputs(dev, unseen, template, december):
    if dev.load_id.duplicated().any() or unseen.load_id.duplicated().any():
        raise ValueError("duplicate load_id")
    if len(unseen) != 12000 or len(template) != 12000:
        raise ValueError("12,000 final rows required")
    if template.columns.tolist() != ["load_id", "predicted_rate"]:
        raise ValueError("template schema mismatch")
    if template.load_id.tolist() != unseen.load_id.tolist():
        raise ValueError("template ID order differs from validation file")
    if "posted_rate" in unseen or "posted_rate" not in dev:
        raise ValueError("unexpected target schema")
    if len(december) != 31 or december.date.tolist() != pd.date_range("2025-12-01", "2025-12-31").strftime("%Y-%m-%d").tolist():
        raise ValueError("December must have all 31 dates in order")
    if set(december.pickup) != {"Lexington"} or set(december.delivery) != {"Fort Wayne"}:
        raise ValueError("December route changed")


def run(root: Path):
    data = root / "data"
    paths = {name: data / name for name in
             ("train_test.csv", "validation.csv", "validation_predictions_template.csv", "december_chart_inputs.csv")}
    dev = pd.read_csv(paths["train_test.csv"])
    unseen = pd.read_csv(paths["validation.csv"])
    template = pd.read_csv(paths["validation_predictions_template.csv"])
    december = pd.read_csv(paths["december_chart_inputs.csv"])
    validate_inputs(dev, unseen, template, december)
    dates = pd.to_datetime(dev.date)
    if dates.min() != pd.Timestamp("2025-01-01") or dates.max() != pd.Timestamp("2025-10-31"):
        raise ValueError("unexpected development date bounds")
    if pd.to_datetime(unseen.date).min() <= dates.max():
        raise ValueError("final predictions must be strictly in the future")
    if dev.posted_rate.isna().any() or (dev.posted_rate <= 0).any():
        raise ValueError("target must be positive")
    comparisons = []
    model_builders = make_models
    # Each rolling fold independently learns city lookup, imputation and estimator.
    for start, stop in [("2025-07-01", "2025-08-01"),
                        ("2025-08-01", "2025-09-01"),
                        ("2025-09-01", "2025-10-01")]:
        past = dev.loc[dates < start]
        hold = dev.loc[(dates >= start) & (dates < stop)]
        lookup = coordinates(past)
        x_past, x_hold = features(past, lookup), features(hold, lookup)
        y_past = past.posted_rate / past.distance
        for name, model in model_builders().items():
            result = score(model, x_past, y_past, x_hold, hold.posted_rate, hold.distance)
            comparisons.append({"period": start[:7], "model": name,
                                "train_rows": len(past), "holdout_rows": len(hold), **result})
        if "catboost_rate_per_mile" in model_builders():
            medians = signal_medians(past)
            result = score(model_builders()["catboost_rate_per_mile"],
                           signal_features(past, lookup, medians), y_past,
                           signal_features(hold, lookup, medians), hold.posted_rate, hold.distance)
            comparisons.append({"period": start[:7], "model": "catboost_market_signals",
                                "train_rows": len(past), "holdout_rows": len(hold), **result})
    scores = pd.DataFrame(comparisons)
    # Choose on July/August only. September/October remain independent final checks.
    selection = scores[scores.period.isin(["2025-07", "2025-08"])].groupby("model").mae_usd.mean()
    selected = selection.idxmin()
    for start, stop in [("2025-09-01", "2025-10-01"), ("2025-10-01", "2025-11-01")]:
        past = dev.loc[dates < start]
        hold = dev.loc[(dates >= start) & (dates < stop)]
        lookup = coordinates(past)
        medians = signal_medians(past)
        if selected == "catboost_market_signals":
            x_past, x_hold = signal_features(past, lookup, medians), signal_features(hold, lookup, medians)
            estimator = model_builders()["catboost_rate_per_mile"]
        else:
            x_past, x_hold = features(past, lookup), features(hold, lookup)
            estimator = model_builders()[selected]
        result = score(estimator, x_past, past.posted_rate / past.distance,
                       x_hold, hold.posted_rate, hold.distance)
        scores = pd.concat([scores, pd.DataFrame([{"period": start[:7], "model": "final_check:" + selected,
                                                  "train_rows": len(past), "holdout_rows": len(hold), **result}])], ignore_index=True)
    lookup = coordinates(dev)
    medians = signal_medians(dev)
    if selected == "catboost_market_signals":
        model = model_builders()["catboost_rate_per_mile"]
        model.fit(signal_features(dev, lookup, medians), dev.posted_rate / dev.distance)
        validation_x = signal_features(unseen, lookup, medians)
        validation_features = FEATURES + SIGNALS
    else:
        model = model_builders()[selected]
        model.fit(features(dev, lookup), dev.posted_rate / dev.distance)
        validation_x = features(unseen, lookup)
        validation_features = FEATURES
    # The fixed December scenario has no market signals, so use the selected
    # best candidate trained on the common feature contract.
    core_selection = selection.drop(labels=["catboost_market_signals"], errors="ignore")
    core_name = core_selection.idxmin()
    core_model = model_builders()[core_name]
    core_model.fit(features(dev, lookup), dev.posted_rate / dev.distance)
    joblib.dump({"model": model, "december_model": core_model,
                 "city_lookup": lookup, "signal_medians": medians,
                 "feature_names": validation_features, "core_feature_names": FEATURES,
                 "target_unit": "USD_per_mile", "prediction_unit": "USD"},
                root / "artifacts/model.joblib")
    prediction = np.maximum(model.predict(validation_x) * unseen.distance, 1)
    template["predicted_rate"] = np.round(prediction, 2)
    december["predicted_rate"] = np.round(np.maximum(core_model.predict(features(december, lookup)) * december.distance, 1), 2)
    if not np.isfinite(template.predicted_rate).all() or not np.isfinite(december.predicted_rate).all():
        raise ValueError("non-finite prediction")
    (root / "outputs").mkdir(exist_ok=True)
    (root / "artifacts").mkdir(exist_ok=True)
    template.to_csv(root / "outputs/validation_predictions.csv", index=False)
    december.to_csv(root / "outputs/december_chart_inputs.csv", index=False)
    scores.to_csv(root / "artifacts/backtests.csv", index=False)
    run_info = {"selected_model": selected, "december_model": core_name,
                "selection_mae_usd": selection.to_dict(),
                "input_sha256": {k: sha256(v) for k, v in paths.items()},
                "features": validation_features, "december_features": FEATURES,
                "signal_medians": medians, "random_seed": 42,
                "final_development_rows": len(dev), "final_validation_rows": len(unseen),
                "negative_weight_rows_corrected": int((dev.weight < 0).sum()),
                "missing_weight_rows": int(dev.weight.isna().sum()),
                "missing_market_index_rows": int(dev.market_index.isna().sum()),
                "december_city_lookup": {k: lookup[k] for k in ("Lexington", "Fort Wayne")}}
    (root / "artifacts/run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(scores.round(2).to_string(index=False))
    print("Selected:", selected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    run(parser.parse_args().root)
