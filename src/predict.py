"""Batch inference from a frozen model bundle; never fits on scoring inputs."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pipeline import features, signal_features, FEATURES, SIGNALS


def predict(model_path: Path, input_path: Path, output_path: Path) -> None:
    bundle = joblib.load(model_path)
    if bundle["core_feature_names"] != FEATURES:
        raise ValueError("model feature schema is incompatible")
    frame = pd.read_csv(input_path)
    if "load_id" in frame:
        expected = FEATURES + SIGNALS if bundle["feature_names"] != FEATURES else FEATURES
        if bundle["feature_names"] != expected:
            raise ValueError("validation feature schema is incompatible")
        x = signal_features(frame, bundle["city_lookup"], bundle["signal_medians"]) if expected != FEATURES else features(frame, bundle["city_lookup"])
        model = bundle["model"]
    else:
        x = features(frame, bundle["city_lookup"])
        model = bundle["december_model"]
    values = np.maximum(model.predict(x) * frame.distance, 1)
    if "load_id" in frame:
        result = pd.DataFrame({"load_id": frame.load_id, "predicted_rate": np.round(values, 2)})
    else:
        result = frame.copy()
        result["predicted_rate"] = np.round(values, 2)
    if not np.isfinite(result.predicted_rate).all():
        raise ValueError("non-finite prediction")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predict(args.model, args.input, args.output)
