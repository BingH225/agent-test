"""SHAP explanations for the frozen physiological stress classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import shap
import torch

from .model import FEATURE_NAMES, WesadAttentionPredictor


@dataclass(frozen=True)
class StressExplanation:
    """Feature-level SHAP attribution output for one inference."""

    attributions: dict[str, float]
    top_drivers: tuple[dict[str, float | str], ...]
    background: str = "neutral_normalized_unit_vector"


class WesadGradientShapExplainer:
    """Gradient SHAP using the paper's neutral normalized baseline."""

    def __init__(
        self,
        predictor: WesadAttentionPredictor,
        *,
        nsamples: int = 128,
    ) -> None:
        if nsamples <= 0:
            raise ValueError("nsamples must be positive")
        self.predictor = predictor
        self.nsamples = int(nsamples)
        background = torch.ones(
            (1, len(FEATURE_NAMES)),
            dtype=torch.float32,
            device=predictor.device,
        )
        self._explainer = shap.GradientExplainer(predictor.model, background)

    def explain(
        self,
        features: Sequence[float],
        *,
        top_k: int = 3,
    ) -> StressExplanation:
        if len(features) != len(FEATURE_NAMES):
            raise ValueError(f"Expected {len(FEATURE_NAMES)} features")
        if not 1 <= top_k <= len(FEATURE_NAMES):
            raise ValueError(f"top_k must be between 1 and {len(FEATURE_NAMES)}")

        values = torch.tensor(
            [list(features)],
            dtype=torch.float32,
            device=self.predictor.device,
        )
        shap_values = self._explainer.shap_values(
            values,
            nsamples=self.nsamples,
            rseed=0,
        )
        flattened = np.asarray(shap_values, dtype=np.float64).reshape(-1)
        if len(flattened) != len(FEATURE_NAMES) or not np.isfinite(flattened).all():
            raise RuntimeError("SHAP returned an invalid attribution vector")

        attributions = {
            feature_name: float(attribution)
            for feature_name, attribution in zip(FEATURE_NAMES, flattened)
        }
        ranked_indices = np.argsort(np.abs(flattened))[::-1][:top_k]
        top_drivers = tuple(
            {
                "feature": FEATURE_NAMES[index],
                "feature_value": float(features[index]),
                "attribution": float(flattened[index]),
                "direction": (
                    "increases_stress_probability"
                    if flattened[index] >= 0
                    else "decreases_stress_probability"
                ),
            }
            for index in ranked_indices
        )
        return StressExplanation(
            attributions=attributions,
            top_drivers=top_drivers,
        )
