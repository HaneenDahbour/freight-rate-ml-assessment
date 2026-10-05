"""Validate assessment deliverables and draw the fixed December chart."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def verify(root: Path):
    reference = pd.read_csv(root / "data/validation.csv")
    predictions = pd.read_csv(root / "outputs/validation_predictions.csv")
    december = pd.read_csv(root / "outputs/december_chart_inputs.csv")
    original = pd.read_csv(root / "data/december_chart_inputs.csv")
    assert predictions.columns.tolist() == ["load_id", "predicted_rate"]
    assert len(predictions) == 12000
    assert predictions.load_id.tolist() == reference.load_id.tolist()
    assert predictions.load_id.is_unique
    assert np.isfinite(predictions.predicted_rate).all()
    assert (predictions.predicted_rate > 0).all()
    assert december.columns.tolist() == original.columns.tolist()
    assert len(december) == 31
    pd.testing.assert_frame_equal(december.drop(columns="predicted_rate"),
                                  original.drop(columns="predicted_rate"))
    assert np.isfinite(december.predicted_rate).all()
    assert (december.predicted_rate > 0).all()
    dates = pd.to_datetime(december.date)
    fig, ax = plt.subplots(figsize=(10.8, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    ax.plot(dates, december.predicted_rate, color="#064A56", linewidth=2.6,
            marker="o", markersize=3.2)
    floor = max(10.0, float(december.predicted_rate.min()) * .02)
    ax.fill_between(dates, december.predicted_rate, floor, color="#064A56", alpha=.08)
    ax.set_title("Candidate: December 2025 Predicted Load Rate", loc="left", fontsize=15, weight="bold")
    ax.set_ylabel("Predicted rate ($)")
    ax.grid(axis="y", color="#D9E2E4")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=35)
    ax.text(0, -.40, "Fixed inputs: Lexington to Fort Wayne | 360 miles | Dry Van | 32,000 lb | only date changes",
            transform=ax.transAxes, fontsize=9.5, color="#455A60")
    fig.tight_layout(rect=(0, .12, 1, 1))
    out = root / "report/candidate_december.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Validated 12,000 final predictions and 31 December predictions; chart:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    verify(parser.parse_args().root)
