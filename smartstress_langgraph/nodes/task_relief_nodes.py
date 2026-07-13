"""TaskRelief planning and strictly side-effect-free dry-run execution."""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.messages import AIMessage

from ..llm.prompts import TASK_RELIEF_SYSTEM_PROMPT
from ..state import SmartStressState, ToolCall, append_audit_event, append_error


DRY_RUN_TOOL_NAME = "dry_run_schedule_adjustment"
ALLOWED_DRY_RUN_TOOLS = {DRY_RUN_TOOL_NAME}


def _generate_chat(*, messages: list[dict[str, str]], system_prompt: str) -> str:
    from ..llm.client import generate_chat

    return generate_chat(messages=messages, system_prompt=system_prompt)


def _driver_context(state: SmartStressState) -> str:
    drivers = state.get("physio_top_drivers", [])[:3]
    if not drivers:
        return "No physiological driver attribution is available."
    names = [str(driver.get("feature", "unknown")) for driver in drivers]
    return (
        "The frozen model's strongest contributors were "
        + ", ".join(names)
        + ". Treat these only as personalization context, not medical findings."
    )


def _grounding_context(state: SmartStressState) -> str:
    snippets = [str(snippet).strip() for snippet in state.get("rag_context", [])[:3]]
    snippets = [snippet for snippet in snippets if snippet]
    if not snippets:
        return "No retrieved support passage is available; keep the proposal conservative."
    return (
        "Keep the proposal consistent with this retrieved support material:\n- "
        + "\n- ".join(snippets)
    )


def _with_observability(
    state: SmartStressState,
    updates: Dict[str, Any],
) -> Dict[str, Any]:
    if state.get("audit_trail"):
        updates["audit_trail"] = list(state["audit_trail"])
    if state.get("error_log"):
        updates["error_log"] = list(state["error_log"])
    return updates


def task_relief_propose_node(state: SmartStressState) -> Dict[str, Any]:
    """Propose one reversible schedule adjustment for dry-run review."""
    stressor = state.get("current_stressor")
    if not stressor:
        return {}

    preferences = state.get("user_preferences", {})
    preference_clause = ""
    if preferences:
        pref_text = ", ".join(f"{key}={value}" for key, value in preferences.items())
        preference_clause = f"User preferences: {pref_text}.\n"

    prompt = (
        f"The user's primary stressor is: {stressor}.\n"
        f"{preference_clause}"
        f"Personalization context: {_driver_context(state)}\n"
        f"Grounding context: {_grounding_context(state)}\n"
        "Propose one concrete, low-risk and reversible task or schedule adjustment. "
        "This is a dry-run proposal only: do not claim that any calendar, task, message, "
        "or external service was changed. Answer in one English sentence with the action "
        "and time window."
    )
    try:
        plan_text = _generate_chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=TASK_RELIEF_SYSTEM_PROMPT,
        ).strip()
    except Exception as exc:
        append_error(state, f"TaskRelief planning failure: {exc}")
        return _with_observability(state, {})

    if not plan_text:
        append_error(state, "TaskRelief returned an empty plan")
        return _with_observability(state, {})

    proposed_action: ToolCall = {
        "tool_name": DRY_RUN_TOOL_NAME,
        "execution_mode": "dry_run",
        "tool_input": {
            "plan": plan_text,
            "stressor": stressor,
            "physio_driver_features": [
                driver.get("feature") for driver in state.get("physio_top_drivers", [])[:3]
            ],
            "grounding_sources": list(state.get("rag_context", [])[:3]),
            "external_side_effects": False,
        },
    }
    append_audit_event(
        state,
        node_name="task_relief_propose",
        summary="Proposed side-effect-free dry-run action",
        details={
            "plan": plan_text,
            "execution_mode": "dry_run",
            "external_side_effects": False,
        },
    )
    return _with_observability(state, {"suggested_action": proposed_action})


def execute_tool_node(state: SmartStressState) -> Dict[str, Any]:
    """Simulate an allowlisted action after explicit consent; never call real tools."""
    action = state.get("suggested_action")
    if not action:
        return {}

    response = state.get("human_confirmation_response")
    if response != "yes":
        append_audit_event(
            state,
            node_name="execute_tool",
            summary="Dry-run skipped without explicit consent",
            details={"response": response, "external_side_effects": False},
        )
        return _with_observability(
            state,
            {
                "suggested_action": None,
                "human_confirmation_response": None,
                "tool_execution_mode": "dry_run",
                "external_side_effects": False,
            },
        )

    tool_name = str(action.get("tool_name", ""))
    execution_mode = action.get("execution_mode")
    if execution_mode != "dry_run" or tool_name not in ALLOWED_DRY_RUN_TOOLS:
        append_error(
            state,
            f"Blocked non-allowlisted TaskRelief action: {tool_name or 'missing tool name'}",
        )
        append_audit_event(
            state,
            node_name="execute_tool",
            summary="Blocked action outside dry-run allowlist",
            details={
                "tool_name": tool_name,
                "execution_mode": execution_mode,
                "external_side_effects": False,
            },
        )
        return _with_observability(
            state,
            {
                "suggested_action": None,
                "human_confirmation_response": None,
                "awaiting_human_confirmation": False,
                "tool_execution_mode": "dry_run",
                "external_side_effects": False,
            },
        )

    tool_input = dict(action.get("tool_input", {}))
    result_payload = {
        "status": "simulated",
        "tool_name": tool_name,
        "execution_mode": "dry_run",
        "external_side_effects": False,
        "would_apply": tool_input,
    }
    result_text = "[DRY-RUN] " + json.dumps(result_payload, ensure_ascii=False, sort_keys=True)
    history = list(state.get("conversation_history", []))
    history.append(
        AIMessage(
            content=(
                result_text
                + "\nNo calendar, task manager, message, or external system was modified."
            )
        )
    )
    append_audit_event(
        state,
        node_name="execute_tool",
        summary="Simulated allowlisted TaskRelief action",
        details={
            "tool_name": tool_name,
            "execution_mode": "dry_run",
            "external_side_effects": False,
        },
    )
    return _with_observability(
        state,
        {
            "tool_output": result_text,
            "tool_execution_mode": "dry_run",
            "external_side_effects": False,
            "suggested_action": None,
            "current_stressor": None,
            "human_confirmation_response": None,
            "awaiting_human_confirmation": False,
            "conversation_history": history,
        },
    )
