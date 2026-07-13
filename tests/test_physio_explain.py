from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from smartstress_langgraph.physio.explain import WesadGradientShapExplainer
from smartstress_langgraph.physio.model import FEATURE_NAMES, WesadAttentionPredictor


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "wesad_attention_v1_golden.json"


class PhysioExplainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sample = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["samples"][1]
        cls.features = sample["features"]
        cls.explanation = WesadGradientShapExplainer(
            WesadAttentionPredictor(),
            nsamples=64,
        ).explain(cls.features)

    def test_returns_one_finite_attribution_per_feature(self) -> None:
        self.assertEqual(tuple(self.explanation.attributions), FEATURE_NAMES)
        self.assertTrue(np.isfinite(list(self.explanation.attributions.values())).all())

    def test_top_drivers_are_ranked_by_absolute_attribution(self) -> None:
        self.assertEqual(len(self.explanation.top_drivers), 3)
        magnitudes = [abs(driver["attribution"]) for driver in self.explanation.top_drivers]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))
        self.assertTrue(all(driver["feature"] in FEATURE_NAMES for driver in self.explanation.top_drivers))


if __name__ == "__main__":
    unittest.main()
