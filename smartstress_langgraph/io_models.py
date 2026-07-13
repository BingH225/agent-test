from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .state import SessionHandle

class UserInfo(BaseModel):
    user_id: str
    session_id: str
    traits: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional user metadata (e.g., age group, role).",
    )


class SensorData(BaseModel):
    """One model-ready feature vector or one raw ECG window plus baseline."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    timestamp: str
    normalized_features: Optional[List[float]] = Field(
        default=None,
        min_length=12,
        max_length=12,
        description="12 neutral-baseline-normalized values in the model manifest order.",
    )
    raw_ecg: Optional[List[float]] = Field(
        default=None,
        description="Raw ECG samples for a window of at least 20 seconds.",
    )
    sample_rate_hz: float = Field(default=700.0, gt=0)
    baseline_features: Optional[List[float]] = Field(
        default=None,
        min_length=12,
        max_length=12,
        description="12 unnormalized features calculated from neutral ECG.",
    )
    baseline_ecg: Optional[List[float]] = Field(
        default=None,
        description="User-specific neutral ECG samples used for normalization.",
    )
    baseline_sample_rate_hz: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_input_contract(self) -> "SensorData":
        has_features = self.normalized_features is not None
        has_ecg = self.raw_ecg is not None
        if has_features == has_ecg:
            raise ValueError("Provide exactly one of normalized_features or raw_ecg")

        has_baseline_features = self.baseline_features is not None
        has_baseline_ecg = self.baseline_ecg is not None
        if has_features:
            if has_baseline_features or has_baseline_ecg or self.baseline_sample_rate_hz:
                raise ValueError("Baseline input is only valid with raw_ecg")
            return self

        if has_baseline_features == has_baseline_ecg:
            raise ValueError(
                "raw_ecg requires exactly one of baseline_features or baseline_ecg"
            )
        minimum_samples = int(20 * self.sample_rate_hz)
        if len(self.raw_ecg or []) < minimum_samples:
            raise ValueError(
                f"raw_ecg must contain at least 20 seconds ({minimum_samples} samples)"
            )
        if has_baseline_ecg:
            baseline_rate = self.baseline_sample_rate_hz or self.sample_rate_hz
            minimum_baseline_samples = int(20 * baseline_rate)
            if len(self.baseline_ecg or []) < minimum_baseline_samples:
                raise ValueError(
                    "baseline_ecg must contain at least 20 seconds "
                    f"({minimum_baseline_samples} samples)"
                )
        elif self.baseline_sample_rate_hz is not None:
            raise ValueError("baseline_sample_rate_hz is only valid with baseline_ecg")
        return self

    def to_payload(self) -> Dict[str, Any]:
        """Serialize only fields consumed by the PhysioSense node."""
        return self.model_dump(
            exclude={"timestamp"},
            exclude_none=True,
        )


class StartSessionRequest(BaseModel):
    user: UserInfo
    initial_sensor_data: Optional[SensorData] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class SessionHandleModel(BaseModel):
    user_id: str
    session_id: str
    thread_id: str
    checkpoint_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_handle(cls, handle: SessionHandle) -> "SessionHandleModel":
        return cls(
            user_id=handle.user_id,
            session_id=handle.session_id,
            thread_id=handle.thread_id,
            checkpoint_id=handle.checkpoint_id,
            metadata=handle.metadata,
        )

    def to_handle(self) -> SessionHandle:
        return SessionHandle(
            user_id=self.user_id,
            session_id=self.session_id,
            thread_id=self.thread_id,
            checkpoint_id=self.checkpoint_id,
            metadata=self.metadata,
        )


class ContinueSessionRequest(BaseModel):
    session_handle: SessionHandleModel
    user_message: Optional[ChatMessage] = None
    sensor_data: Optional[SensorData] = None


class RagDocumentMeta(BaseModel):
    id: Optional[str] = None
    source: Optional[str] = None
    section: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class RagIngestionRequest(BaseModel):
    folder_path: str
    tags: List[str] = Field(default_factory=list)


class SmartStressStateView(BaseModel):
    """
    Serializable view of SmartStressState for frontend / API responses.
    """

    user_id: str
    session_id: str
    current_stress_prob: Optional[float] = None
    stress_detected: Optional[bool] = None
    stress_threshold: Optional[float] = None
    physio_model_id: Optional[str] = None
    physio_input_source: Optional[str] = None
    physio_feature_map: Dict[str, float] = Field(default_factory=dict)
    physio_attributions: Dict[str, float] = Field(default_factory=dict)
    physio_top_drivers: List[Dict[str, Any]] = Field(default_factory=list)
    stress_history: List[float] = Field(default_factory=list)
    stress_timestamps: List[str] = Field(default_factory=list)
    current_stressor: Optional[str] = None
    suggested_action: Optional[Dict[str, Any]] = None
    tool_output: Optional[str] = None
    tool_execution_mode: Optional[str] = None
    external_side_effects: bool = False
    awaiting_human_confirmation: bool = False
    human_confirmation_response: Optional[str] = None
    rag_context: List[str] = Field(default_factory=list)
    error_log: List[str] = Field(default_factory=list)
    audit_trail: List[Dict[str, Any]] = Field(default_factory=list)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)



