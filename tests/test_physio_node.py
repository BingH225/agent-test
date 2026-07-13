from __future__ import annotations

import json
from pathlib import Path
import unittest

from smartstress_langgraph.nodes.physio_sense_node import physio_sense_node


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "wesad_attention_v1_golden.json"


class PhysioSenseNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        samples = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["samples"]
        cls.stress_sample = next(sample for sample in samples if sample["wesad_label"] == 2)

    def test_runs_frozen_model_and_records_provenance(self) -> None:
        timestamp = "2026-07-13T18:00:00Z"
        updates = physio_sense_node({
            "raw_sensor_input": {
                "timestamp": timestamp,
                "normalized_features": self.stress_sample["features"],
                "sample_rate_hz": 700.0,
            },
            "stress_history": [],
            "stress_timestamps": [],
            "audit_trail": [],
        })
        self.assertAlmostEqual(
            updates["current_stress_prob"],
            self.stress_sample["expected_probability"],
            places=6,
        )
        self.assertTrue(updates["stress_detected"])
        self.assertEqual(updates["physio_model_id"], "wesad_attention_v1")
        self.assertEqual(updates["physio_input_source"], "normalized_features")
        self.assertEqual(updates["stress_timestamps"], [timestamp])
        self.assertIsNone(updates["raw_sensor_input"])

    def test_no_sensor_input_does_not_create_fake_probability(self) -> None:
        self.assertEqual(physio_sense_node({"stress_history": []}), {})

    def test_invalid_payload_is_logged_without_new_history_item(self) -> None:
        updates = physio_sense_node({
            "raw_sensor_input": {"normalized_features": [1.0] * 11},
            "current_stress_prob": 0.42,
            "stress_history": [0.42],
            "stress_timestamps": ["earlier"],
            "error_log": [],
            "audit_trail": [],
        })
        self.assertNotIn("current_stress_prob", updates)
        self.assertNotIn("stress_history", updates)
        self.assertIn("PhysioSense inference failure", updates["error_log"][-1])
        self.assertIsNone(updates["raw_sensor_input"])


if __name__ == "__main__":
    unittest.main()
