"""Meta-reflective state assessment and routing for the SmartStress graph."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal

from .state import SmartStressState, append_audit_event


OrchestrationDecision = Literal[
    "wait_confirmation",
    "execute",
    "refine",
    "propose",
    "support",
    "escalate",
    "monitor",
]


@dataclass(frozen=True)
class OrchestrationReflection:
    decision: OrchestrationDecision
    reason: str
    signals: Dict[str, Any]


def reflect_on_state(state: SmartStressState) -> OrchestrationReflection:
    """Integrate physiological, semantic, grounding, and acceptance signals."""
    probability = float(state.get("current_stress_prob", 0.0))
    threshold = float(state.get("stress_threshold", 0.5))
    stress_detected = bool(state.get("stress_detected", probability >= threshold))
    has_stressor = bool(state.get("current_stressor"))
    grounding_count = len(state.get("rag_context", []))
    acceptance = state.get("human_confirmation_response")
    signals = {
        "physiological_urgency": probability,
        "stress_threshold": threshold,
        "stress_detected": stress_detected,
        "semantic_specificity": has_stressor,
        "grounding_count": grounding_count,
        "user_acceptance": acceptance or "not_requested",
        "refinement_requested": bool(state.get("refinement_requested")),
        "safety_escalation": bool(state.get("safety_escalation")),
    }

    if state.get("safety_escalation"):
        return OrchestrationReflection(
            "escalate",
            "Crisis language blocks TaskRelief and requires immediate human support guidance.",
            signals,
        )
    if state.get("awaiting_human_confirmation"):
        return OrchestrationReflection(
            "wait_confirmation",
            "A dry-run proposal requires an explicit yes, no, or cancel response.",
            signals,
        )
    if acceptance == "yes":
        return OrchestrationReflection(
            "execute",
            "The user explicitly accepted the allowlisted dry-run proposal.",
            signals,
        )
    if acceptance == "no" or state.get("refinement_requested"):
        return OrchestrationReflection(
            "refine",
            "The user declined the proposal; wait for preferences before replanning.",
            signals,
        )
    if acceptance == "cancel":
        return OrchestrationReflection(
            "monitor",
            "The user cancelled intervention; end this invocation without an action.",
            signals,
        )
    if has_stressor and not state.get("suggested_action"):
        return OrchestrationReflection(
            "propose",
            "A concrete stressor is available and no TaskRelief proposal exists.",
            signals,
        )
    if stress_detected:
        return OrchestrationReflection(
            "support",
            "Stress is elevated but more dialogue or semantic specificity is needed.",
            signals,
        )
    return OrchestrationReflection(
        "monitor",
        "No elevated stress event or pending intervention requires action.",
        signals,
    )


def meta_reflect_node(state: SmartStressState) -> Dict[str, Any]:
    """Persist the orchestrator's evidence and decision into shared state."""
    reflection = reflect_on_state(state)
    append_audit_event(
        state,
        node_name="meta_reflective_orchestrator",
        summary=f"Orchestration decision: {reflection.decision}",
        details={"reason": reflection.reason, "signals": reflection.signals},
    )
    return {
        "orchestration_decision": reflection.decision,
        "orchestration_reason": reflection.reason,
        "orchestration_signals": reflection.signals,
        "audit_trail": list(state.get("audit_trail", [])),
    }


def route_after_orchestrator(state: SmartStressState) -> str:
    """Map a persisted reflective decision to a LangGraph edge."""
    decision = state.get("orchestration_decision")
    if decision in {"wait_confirmation", "refine"}:
        return "wait_for_human_input"
    if decision == "execute":
        return "execute_tool"
    if decision == "propose":
        return "propose_relief_action"
    return "end"


def reflection_as_dict(state: SmartStressState) -> Dict[str, Any]:
    """Convenience helper for diagnostics and tests."""
    return asdict(reflect_on_state(state))
