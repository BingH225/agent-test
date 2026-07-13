"""Frozen Attention-DNN used by the paper's PhysioSense agent.

The architecture and weights are migrated from ``smart-stress-model`` without
retraining.  The bundled checkpoint is the S17 holdout checkpoint that produced
the paper-reported 95.78% F1 score.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn


MODEL_ID = "wesad_attention_v1"
FEATURE_NAMES = (
    "mean_hr",
    "std_hr",
    "tinn",
    "hrv_index",
    "nn50",
    "pnn50",
    "mean_hrv",
    "std_hrv",
    "rmssd",
    "fft_mean",
    "fft_std",
    "sum_psd",
)

_MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = _MODULE_DIR / "artifacts" / "wesad_attention_v1.json"


class AttentionStressClassifier(nn.Module):
    """12-feature self-attention binary classifier from the source project."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = 32
        self.num_heads = 4
        self.embedding = nn.Linear(1, self.embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.nnECG = nn.Sequential(
            nn.Linear(12 * self.embed_dim, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 16, bias=True),
            nn.BatchNorm1d(16),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 4, bias=True),
            nn.BatchNorm1d(4),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            nn.Linear(4, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected a [batch, {len(FEATURE_NAMES)}] tensor, "
                f"received shape {tuple(features.shape)}"
            )
        batch_size = features.size(0)
        embedded = self.embedding(features.view(batch_size, len(FEATURE_NAMES), 1))
        attended, _ = self.attention(embedded, embedded, embedded)
        return self.nnECG(attended.reshape(batch_size, -1))


@dataclass(frozen=True)
class StressPrediction:
    """Result of one 12-feature stress inference."""

    probability: float
    is_stress: bool
    threshold: float
    model_id: str = MODEL_ID


class WesadAttentionPredictor:
    """Checksum-validated, CPU-friendly inference wrapper."""

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        device: str | torch.device = "cpu",
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        self.threshold = float(threshold)
        self.device = torch.device(device)
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._validate_manifest()

        checkpoint_path = self.manifest_path.parent / self.manifest["checkpoint"]["filename"]
        actual_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest().upper()
        expected_sha256 = self.manifest["checkpoint"]["sha256"].upper()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Checkpoint checksum mismatch: expected {expected_sha256}, "
                f"received {actual_sha256}"
            )

        self.model = AttentionStressClassifier().to(self.device)
        try:
            state_dict = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:  # PyTorch < 2.0 does not accept weights_only.
            state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _validate_manifest(self) -> None:
        if self.manifest.get("model_id") != MODEL_ID:
            raise ValueError(f"Manifest model_id must be {MODEL_ID!r}")
        if tuple(self.manifest.get("feature_order", ())) != FEATURE_NAMES:
            raise ValueError("Manifest feature_order does not match the model contract")

    def predict(self, features: Sequence[float]) -> StressPrediction:
        if len(features) != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(FEATURE_NAMES)} normalized features in this order: "
                f"{', '.join(FEATURE_NAMES)}"
            )
        feature_tensor = torch.tensor(
            [list(features)],
            dtype=torch.float32,
            device=self.device,
        )
        if not torch.isfinite(feature_tensor).all():
            raise ValueError("All features must be finite numbers")
        with torch.inference_mode():
            probability = float(self.model(feature_tensor).item())
        return StressPrediction(
            probability=probability,
            is_stress=probability >= self.threshold,
            threshold=self.threshold,
        )
