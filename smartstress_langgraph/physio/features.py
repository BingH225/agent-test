"""ECG-to-feature contract compatible with ``wesad_attention_v1``.

The frozen model was trained with the source project's exact feature formulas.
This module intentionally preserves those formulas, then normalizes every
feature by a user-specific neutral baseline as described in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Sequence

import heartpy
import numpy as np
from scipy import fft, signal, stats

from .model import FEATURE_NAMES


MODEL_SAMPLE_RATE_HZ = 700.0
MIN_ECG_WINDOW_SECONDS = 20.0


@dataclass(frozen=True)
class PreparedPhysioFeatures:
    """Normalized model features plus preprocessing provenance."""

    values: tuple[float, ...]
    source: Literal["normalized_features", "raw_ecg"]
    raw_values: tuple[float, ...] | None = None
    baseline_values: tuple[float, ...] | None = None

    def as_dict(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.values))


def validate_feature_vector(
    values: Sequence[float],
    *,
    name: str = "features",
) -> tuple[float, ...]:
    """Validate and freeze a vector in the model's canonical 12-feature order."""
    if len(values) != len(FEATURE_NAMES):
        raise ValueError(
            f"{name} must contain exactly {len(FEATURE_NAMES)} values in this order: "
            f"{', '.join(FEATURE_NAMES)}"
        )
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numeric values")
    return tuple(float(value) for value in array)


def _best_tinn(intervals: np.ndarray) -> tuple[float, float]:
    kernel = stats.gaussian_kde(intervals)
    axis = np.linspace(np.min(intervals), np.max(intervals), len(intervals))
    density = kernel.evaluate(axis)
    spacing = axis[1] - axis[0]
    maximum_index = int(np.argmax(density))
    maximum_position = axis[maximum_index]
    maximum_value = float(np.amax(density))
    left_axis = axis[: maximum_index + 1]
    right_axis = axis[maximum_index:]
    hrv_index = len(intervals) / maximum_value

    left_errors: list[tuple[float, float]] = []
    for index in range(0, len(left_axis) - 1):
        intercept = left_axis[index]
        slope = maximum_value / (maximum_position - intercept)
        approximation = np.clip(
            slope * spacing * np.arange(-index, -index + maximum_index + 1),
            0,
            None,
        )
        squared_error = np.square(density[: maximum_index + 1] - approximation)
        integrated = (np.delete(squared_error, -1) + np.delete(squared_error, 0)) / 2
        left_errors.append((float(np.linalg.norm(integrated)), float(intercept)))

    right_errors: list[tuple[float, float]] = []
    for index in range(1, len(right_axis)):
        intercept = right_axis[index]
        slope = maximum_value / (maximum_position - intercept)
        approximation = np.clip(
            slope * spacing * np.arange(-index, len(right_axis) - index),
            0,
            None,
        )
        squared_error = np.square(density[maximum_index:] - approximation)
        integrated = (np.delete(squared_error, -1) + np.delete(squared_error, 0)) / 2
        right_errors.append((float(np.linalg.norm(integrated)), float(intercept)))

    if not left_errors or not right_errors:
        raise ValueError("ECG intervals do not contain enough variation to compute TINN")
    best_left = min(left_errors, key=lambda item: item[0])[1]
    best_right = min(right_errors, key=lambda item: item[0])[1]
    return float(best_right - best_left), float(hrv_index)


def _pairwise_nn50(intervals: np.ndarray) -> tuple[float, float]:
    # Preserve the training pipeline's all-pairs comparison. Changing this to
    # adjacent RR differences would require retraining the frozen checkpoint.
    count = float(np.sum(np.abs(intervals[:, None] - intervals[None, :]) > 0.05))
    if count == 0:
        count = 1.0
    return count, count / float(len(intervals) ** 2)


def _frequency_features(intervals: np.ndarray) -> tuple[float, float, float]:
    mean_interval = float(np.mean(intervals))
    transformed = np.asarray(fft.fft(intervals - mean_interval))
    frequencies = fft.fftfreq(len(intervals), mean_interval)[: len(intervals) // 2]
    power = (2 / len(transformed)) * np.abs(transformed)[: len(intervals) // 2]
    return (
        float(np.mean(frequencies)),
        float(np.std(frequencies)),
        float(np.sum(power)),
    )


def _resample_to_model_rate(ecg: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    if np.isclose(sample_rate_hz, MODEL_SAMPLE_RATE_HZ):
        return ecg
    ratio = Fraction(MODEL_SAMPLE_RATE_HZ / sample_rate_hz).limit_denominator(10_000)
    return np.asarray(signal.resample_poly(ecg, ratio.numerator, ratio.denominator))


def extract_ecg_features(
    ecg: Sequence[float],
    *,
    sample_rate_hz: float = MODEL_SAMPLE_RATE_HZ,
) -> tuple[float, ...]:
    """Calculate the frozen model's 12 raw features from one ECG window.

    Input windows must contain at least 20 seconds. Signals sampled at another
    positive rate are resampled to the model's 700 Hz training rate first.
    """
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be a positive finite number")
    signal_array = np.asarray(ecg, dtype=np.float64)
    if signal_array.ndim != 1 or not np.isfinite(signal_array).all():
        raise ValueError("ecg must be a one-dimensional sequence of finite numbers")
    minimum_samples = int(np.ceil(MIN_ECG_WINDOW_SECONDS * sample_rate_hz))
    if len(signal_array) < minimum_samples:
        raise ValueError(
            f"ecg must contain at least {MIN_ECG_WINDOW_SECONDS:g} seconds "
            f"({minimum_samples} samples at {sample_rate_hz:g} Hz)"
        )

    model_signal = _resample_to_model_rate(signal_array, float(sample_rate_hz))
    try:
        working_data, _ = heartpy.process(model_signal, MODEL_SAMPLE_RATE_HZ)
        peaks = np.asarray(working_data["peaklist"], dtype=np.float64)
        intervals = np.diff(peaks) / MODEL_SAMPLE_RATE_HZ
        if len(intervals) < 3 or np.any(intervals <= 0):
            raise ValueError("too few valid heartbeat intervals")

        frequencies = 1.0 / intervals
        tinn, hrv_index = _best_tinn(intervals)
        nn50, pnn50 = _pairwise_nn50(intervals)
        fft_mean, fft_std, sum_psd = _frequency_features(intervals)
        # The source training code labels RMS(RR) as RMSSD. Preserve that exact
        # value here so inference remains compatible with the frozen weights.
        training_rmssd = float(np.sqrt(np.mean(np.square(intervals))))
        features = (
            float(np.mean(frequencies)),
            float(np.std(frequencies)),
            tinn,
            hrv_index,
            nn50,
            pnn50,
            float(np.mean(intervals)),
            float(np.std(intervals)),
            training_rmssd,
            fft_mean,
            fft_std,
            sum_psd,
        )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Unable to extract ECG features: {exc}") from exc
    return validate_feature_vector(features, name="extracted ECG features")


def normalize_features(
    raw_features: Sequence[float],
    baseline_features: Sequence[float],
) -> tuple[float, ...]:
    """Normalize raw features by the same user's neutral-baseline features."""
    raw = np.asarray(validate_feature_vector(raw_features, name="raw_features"))
    baseline = np.asarray(
        validate_feature_vector(baseline_features, name="baseline_features")
    )
    if np.any(np.isclose(baseline, 0.0)):
        raise ValueError("baseline_features must not contain zero values")
    normalized = raw / baseline
    return validate_feature_vector(normalized, name="normalized_features")


def prepare_physio_features(
    *,
    normalized_features: Sequence[float] | None = None,
    raw_ecg: Sequence[float] | None = None,
    sample_rate_hz: float = MODEL_SAMPLE_RATE_HZ,
    baseline_features: Sequence[float] | None = None,
    baseline_ecg: Sequence[float] | None = None,
    baseline_sample_rate_hz: float | None = None,
) -> PreparedPhysioFeatures:
    """Resolve either direct 12-D input or raw ECG into normalized model input."""
    has_features = normalized_features is not None
    has_ecg = raw_ecg is not None
    if has_features == has_ecg:
        raise ValueError("Provide exactly one of normalized_features or raw_ecg")

    if normalized_features is not None:
        if baseline_features is not None or baseline_ecg is not None:
            raise ValueError("Baseline input is only valid with raw_ecg")
        values = validate_feature_vector(normalized_features, name="normalized_features")
        return PreparedPhysioFeatures(values=values, source="normalized_features")

    has_baseline_features = baseline_features is not None
    has_baseline_ecg = baseline_ecg is not None
    if has_baseline_features == has_baseline_ecg:
        raise ValueError(
            "raw_ecg requires exactly one of baseline_features or baseline_ecg"
        )

    raw_values = extract_ecg_features(raw_ecg, sample_rate_hz=sample_rate_hz)  # type: ignore[arg-type]
    if baseline_features is not None:
        baseline_values = validate_feature_vector(
            baseline_features,
            name="baseline_features",
        )
    else:
        baseline_values = extract_ecg_features(
            baseline_ecg,  # type: ignore[arg-type]
            sample_rate_hz=(
                baseline_sample_rate_hz
                if baseline_sample_rate_hz is not None
                else sample_rate_hz
            ),
        )
    values = normalize_features(raw_values, baseline_values)
    return PreparedPhysioFeatures(
        values=values,
        source="raw_ecg",
        raw_values=raw_values,
        baseline_values=baseline_values,
    )
