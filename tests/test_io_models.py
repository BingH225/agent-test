from __future__ import annotations

import unittest

from pydantic import ValidationError

from smartstress_langgraph.io_models import SensorData


class SensorDataTests(unittest.TestCase):
    def test_accepts_exactly_twelve_normalized_features(self) -> None:
        sensor = SensorData(timestamp="2026-07-13T18:00:00Z", normalized_features=[1.0] * 12)
        self.assertEqual(sensor.to_payload(), {
            "normalized_features": [1.0] * 12,
            "sample_rate_hz": 700.0,
        })

    def test_rejects_wrong_feature_dimension(self) -> None:
        with self.assertRaises(ValidationError):
            SensorData(timestamp="now", normalized_features=[1.0] * 11)

    def test_accepts_raw_ecg_with_neutral_feature_baseline(self) -> None:
        sensor = SensorData(
            timestamp="now",
            raw_ecg=[0.0] * 14_000,
            sample_rate_hz=700,
            baseline_features=[1.0] * 12,
        )
        self.assertEqual(len(sensor.to_payload()["raw_ecg"]), 14_000)

    def test_raw_ecg_requires_baseline(self) -> None:
        with self.assertRaisesRegex(ValidationError, "requires exactly one"):
            SensorData(timestamp="now", raw_ecg=[0.0] * 14_000)

    def test_rejects_ambiguous_input(self) -> None:
        with self.assertRaisesRegex(ValidationError, "exactly one"):
            SensorData(
                timestamp="now",
                normalized_features=[1.0] * 12,
                raw_ecg=[0.0] * 14_000,
                baseline_features=[1.0] * 12,
            )


if __name__ == "__main__":
    unittest.main()
