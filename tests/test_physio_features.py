from __future__ import annotations

import unittest

import numpy as np

from smartstress_langgraph.physio import (
    FEATURE_NAMES,
    extract_ecg_features,
    normalize_features,
    prepare_physio_features,
)


SOURCE_PIPELINE_GOLDEN = (
    1.2170650749066934,
    0.09684842937804759,
    0.20999999999999996,
    6.013058313699288,
    672.0,
    0.5813148788927336,
    0.8270588235294118,
    0.0681490326962859,
    0.8298617886342846,
    0.28449502133712656,
    0.17421690915954322,
    0.21780308554195235,
)


def make_synthetic_ecg(sample_rate_hz: int = 700, duration: int = 30) -> np.ndarray:
    time = np.arange(sample_rate_hz * duration) / sample_rate_hz
    ecg = 0.015 * np.sin(2 * np.pi * 0.25 * time)
    ecg += 0.005 * np.sin(2 * np.pi * 50 * time)
    rr_pattern = (0.78, 0.82, 0.91, 0.74, 0.87, 0.80, 0.95, 0.76)
    peak_time = 0.6
    index = 0
    while peak_time < duration - 0.4:
        ecg += 1.2 * np.exp(-0.5 * ((time - peak_time) / 0.012) ** 2)
        ecg -= 0.18 * np.exp(-0.5 * ((time - (peak_time - 0.028)) / 0.010) ** 2)
        ecg += 0.25 * np.exp(-0.5 * ((time - (peak_time + 0.20)) / 0.045) ** 2)
        peak_time += rr_pattern[index % len(rr_pattern)]
        index += 1
    return ecg


class PhysioFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ecg = make_synthetic_ecg()
        cls.extracted = extract_ecg_features(cls.ecg)

    def test_matches_source_preprocessing_features(self) -> None:
        np.testing.assert_allclose(
            self.extracted,
            SOURCE_PIPELINE_GOLDEN,
            rtol=1e-8,
            atol=1e-10,
        )

    def test_direct_normalized_features_preserve_order(self) -> None:
        values = tuple(float(index + 1) for index in range(len(FEATURE_NAMES)))
        prepared = prepare_physio_features(normalized_features=values)
        self.assertEqual(prepared.values, values)
        self.assertEqual(prepared.source, "normalized_features")
        self.assertEqual(prepared.as_dict()["mean_hr"], 1.0)
        self.assertEqual(prepared.as_dict()["sum_psd"], 12.0)

    def test_raw_ecg_is_normalized_by_neutral_baseline(self) -> None:
        prepared = prepare_physio_features(
            raw_ecg=self.ecg,
            baseline_features=self.extracted,
        )
        np.testing.assert_allclose(prepared.values, np.ones(12), atol=1e-12)
        self.assertEqual(prepared.source, "raw_ecg")

    def test_normalization_rejects_zero_baseline(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not contain zero"):
            normalize_features([1.0] * 12, [0.0] + [1.0] * 11)

    def test_raw_ecg_requires_one_baseline_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires exactly one"):
            prepare_physio_features(raw_ecg=self.ecg)

    def test_short_ecg_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 20 seconds"):
            extract_ecg_features(self.ecg[: 10 * 700])


if __name__ == "__main__":
    unittest.main()
