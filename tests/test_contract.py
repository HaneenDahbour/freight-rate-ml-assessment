"""Small checks for real failure modes in batch prediction."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pipeline import coordinates, features, validate_inputs


class ContractTests(unittest.TestCase):
    def test_location_lookup_uses_only_supplied_development_rows(self):
        training = pd.DataFrame({"pickup": ["A"], "pickup_lat": [10.0], "pickup_lon": [20.0],
                                 "delivery": ["B"], "delivery_lat": [12.0], "delivery_lon": [22.0]})
        lookup = coordinates(training)
        self.assertEqual(set(lookup), {"A", "B"})
        scoring = pd.DataFrame({"pickup": ["A"], "delivery": ["B"], "distance": [360],
                                "weight": [-32000], "equipment": ["Dry Van"], "date": ["2025-12-01"]})
        transformed = features(scoring, lookup)
        self.assertEqual(transformed.iloc[0].pickup_lat, 10)
        self.assertEqual(transformed.iloc[0].weight, 32000)
        self.assertNotIn("quote_signal", transformed.columns)

    def test_duplicate_final_id_is_rejected(self):
        dev = pd.DataFrame({"load_id": ["TR-1"], "posted_rate": [1]})
        unseen = pd.DataFrame({"load_id": ["TE-1"] * 12000})
        template = pd.DataFrame({"load_id": ["TE-1"] * 12000,
                                 "predicted_rate": [None] * 12000})
        december = pd.DataFrame({"date": pd.date_range("2025-12-01", periods=31).strftime("%Y-%m-%d"),
                                 "pickup": ["Lexington"] * 31, "delivery": ["Fort Wayne"] * 31})
        with self.assertRaisesRegex(ValueError, "duplicate load_id"):
            validate_inputs(dev, unseen, template, december)


if __name__ == "__main__":
    unittest.main()
