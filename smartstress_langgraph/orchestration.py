"""Pure routing rules for the SmartStress graph."""

from __future__ import annotations

from .state import SmartStressState


def route_after_mind_care(state: SmartStressState) -> str:
    """Choose the next node without looping when no new sensor data exists."""
    if state.get("awaiting_human_confirmation"):
        return "wait_for_human_input"
    if state.get("human_confirmation_response") == "yes":
        return "execute_tool"
    if state.get("human_confirmation_response") in {"no", "cancel"}:
        return "end"
    if state.get("current_stressor") and not state.get("suggested_action"):
        return "propose_relief_action"
    return "end"
