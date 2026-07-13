from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, TypedDict

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
else:
    BaseMessage = Any


class ToolCall(TypedDict):
    """A proposed tool call from TaskRelief or other agents."""

    tool_name: str
    execution_mode: Literal["dry_run"]
    tool_input: Dict[str, Any]


class SmartStressState(TypedDict, total=False):
    """
    Central shared state passed between LangGraph nodes.

    This mirrors and extends the design in Agents_develop_LangGraph.md.
    """

    # === User & session ===
    user_id: str
    session_id: str

    # === PhysioSense (L1) ===
    raw_sensor_input: Optional[Dict[str, Any]]
    current_stress_prob: float
    stress_detected: bool
    stress_threshold: float
    physio_model_id: str
    physio_input_source: str
    physio_features: List[float]
    physio_feature_map: Dict[str, float]
    physio_raw_features: List[float]
    physio_baseline_features: List[float]
    physio_attributions: Dict[str, float]
    physio_top_drivers: List[Dict[str, Any]]
    stress_history: List[float]
    stress_timestamps: List[str]  # ISO timestamps aligned with stress_history

    # === MindCare (L2) ===
    conversation_history: List[BaseMessage]
    current_stressor: Optional[str]

    # === TaskRelief (L3) ===
    suggested_action: Optional[ToolCall]
    tool_output: Optional[str]
    tool_execution_mode: Literal["dry_run"]
    external_side_effects: bool

    # === RAG context ===
    rag_context: List[str]
    use_rag: bool  # Whether to use RAG for context retrieval (default True)

    # === Preferences & meta ===
    user_preferences: Dict[str, Any]

    # === Control flow & safety (HITL) ===
    awaiting_human_confirmation: bool
    human_confirmation_response: Optional[str]
    refinement_requested: bool

    # === Meta-reflective orchestration ===
    orchestration_decision: str
    orchestration_reason: str
    orchestration_signals: Dict[str, Any]

    # === Observability & audit ===
    error_log: List[str]
    audit_trail: List[Dict[str, Any]]


def append_audit_event(
    state: SmartStressState,
    node_name: str,
    summary: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Utility to append a structured audit event to the state."""
    trail = list(state.get("audit_trail", []))
    event = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "node": node_name,
        "summary": summary,
    }
    if details:
        event["details"] = details
    trail.append(event)
    state["audit_trail"] = trail


def append_error(state: SmartStressState, message: str) -> None:
    """Utility to append an error entry to the state's error_log."""
    errors = list(state.get("error_log", []))
    errors.append(f"{datetime.utcnow().isoformat()}Z {message}")
    state["error_log"] = errors


@dataclass
class SessionHandle:
    """
    Lightweight handle that external backends can store to resume a session.

    This is not used directly by LangGraph, but by our higher-level SDK API.
    """

    user_id: str
    session_id: str
    thread_id: str
    checkpoint_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)



