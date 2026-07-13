from __future__ import annotations

import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from smartstress_langgraph.physio import FEATURE_NAMES, WesadAttentionPredictor


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "wesad_attention_v1_golden.json"


class WesadAttentionPredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = WesadAttentionPredictor()
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_matches_source_checkpoint_golden_probabilities(self) -> None:
        for sample in self.fixture["samples"]:
            with self.subTest(s17_index=sample["s17_index"]):
                prediction = self.predictor.predict(sample["features"])
                self.assertAlmostEqual(
                    prediction.probability,
                    sample["expected_probability"],
                    places=6,
                )

    def test_threshold_is_configurable(self) -> None:
        boundary = self.fixture["samples"][-1]
        self.assertTrue(self.predictor.predict(boundary["features"]).is_stress)
        with patch.dict(os.environ, {"SMARTSTRESS_STRESS_THRESHOLD": "0.75"}):
            strict_predictor = WesadAttentionPredictor()
        self.assertFalse(strict_predictor.predict(boundary["features"]).is_stress)

    def test_rejects_invalid_configured_threshold(self) -> None:
        with patch.dict(os.environ, {"SMARTSTRESS_STRESS_THRESHOLD": "1.2"}):
            with self.assertRaisesRegex(ValueError, "must be between 0 and 1"):
                WesadAttentionPredictor()

    def test_rejects_wrong_feature_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected 12 normalized features"):
            self.predictor.predict([1.0] * (len(FEATURE_NAMES) - 1))


if __name__ == "__main__":
    unittest.main()
