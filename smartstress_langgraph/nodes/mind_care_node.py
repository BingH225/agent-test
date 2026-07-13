"""MindCare dialogue, physiology-aware RAG, and consent handling."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage

from ..llm.prompts import MIND_CARE_SYSTEM_PROMPT
from ..state import SmartStressState, append_audit_event, append_error


Confirmation = Literal["yes", "no", "cancel"]
_YES_RESPONSES = {"yes", "y", "sure", "ok", "okay", "proceed", "confirm"}
_NO_RESPONSES = {"no", "n", "nope", "nah", "no thanks", "do not proceed"}
_CANCEL_RESPONSES = {"cancel", "stop", "never mind", "nevermind"}


def _generate_chat(*, messages: list[dict[str, str]], system_prompt: str) -> str:
    from ..llm.client import generate_chat

    return generate_chat(messages=messages, system_prompt=system_prompt)


def _normalize_confirmation(text: str) -> Confirmation | None:
    normalized = " ".join(text.strip().lower().split()).rstrip(".!?")
    if normalized in _YES_RESPONSES:
        return "yes"
    if normalized in _NO_RESPONSES:
        return "no"
    if normalized in _CANCEL_RESPONSES:
        return "cancel"
    return None


def _looks_like_confirmation(text: str) -> bool:
    return _normalize_confirmation(text) is not None


def _extract_stressor_from_text(text: str) -> Optional[str]:
    prompt = (
        "The user described their stress as follows:\n"
        f"{text}\n\n"
        "Summarize the single most likely stressor (event, task, or interaction) "
        "in <= 15 English words. If you cannot infer it, respond with 'unknown stressor'."
    )
    try:
        result = _generate_chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=(
                "You are a text classifier. Output only the stressor summary. "
                "Do not add explanations or advice."
            ),
        )
    except Exception:
        return None

    result = (result or "").strip()
    if not result or result.lower() == "unknown stressor":
        return None
    return result


def _is_stress_detected(state: SmartStressState) -> bool:
    if "stress_detected" in state:
        return bool(state.get("stress_detected"))
    probability = float(state.get("current_stress_prob", 0.0))
    threshold = float(state.get("stress_threshold", 0.5))
    return probability >= threshold


def _physio_summary(state: SmartStressState) -> str:
    probability = state.get("current_stress_prob")
    if probability is None:
        return "No current physiological inference is available."
    threshold = float(state.get("stress_threshold", 0.5))
    decision = "elevated" if _is_stress_detected(state) else "not elevated"
    drivers = state.get("physio_top_drivers", [])[:3]
    driver_parts = []
    for driver in drivers:
        feature = str(driver.get("feature", "unknown_feature")).replace("_", " ")
        direction = str(driver.get("direction", "contributes to the estimate")).replace(
            "_", " "
        )
        driver_parts.append(f"{feature} ({direction})")
    driver_text = ", ".join(driver_parts) if driver_parts else "not available"
    return (
        f"Frozen model probability={float(probability):.3f}, threshold={threshold:.3f}, "
        f"decision={decision}; strongest model contributors: {driver_text}. "
        "These are model attributions, not diagnoses or proof of causation."
    )


def _build_rag_query(user_query: str, state: SmartStressState) -> str:
    return (
        "Retrieve concise, evidence-based, non-clinical guidance for short-term "
        "stress regulation and manageable task planning.\n"
        f"User concern: {user_query}\n"
        f"PhysioSense context: {_physio_summary(state)}"
    )


def _retrieve_context(query: str, *, k: int = 3) -> list[str]:
    from ..rag.retrieval import retrieve_context

    return retrieve_context(query, k=k)


def _latest_human_message(state: SmartStressState) -> HumanMessage | None:
    for message in reversed(state.get("conversation_history", [])):
        if isinstance(message, HumanMessage):
            return message
    return None


def _chat_messages(state: SmartStressState) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for message in state.get("conversation_history", []):
        if isinstance(message, HumanMessage):
            messages.append({"role": "user", "content": str(message.content)})
        elif isinstance(message, AIMessage):
            messages.append({"role": "assistant", "content": str(message.content)})
    return messages


def _with_observability(
    state: SmartStressState,
    updates: Dict[str, Any],
) -> Dict[str, Any]:
    if state.get("audit_trail"):
        updates["audit_trail"] = list(state["audit_trail"])
    if state.get("error_log"):
        updates["error_log"] = list(state["error_log"])
    return updates


def mind_care_node(state: SmartStressState) -> Dict[str, Any]:
    """Use model decisions and drivers to guide supportive dialogue and RAG."""
    history = list(state.get("conversation_history", []))

    if state.get("awaiting_human_confirmation"):
        latest_human = _latest_human_message(state)
        confirmation = (
            _normalize_confirmation(str(latest_human.content)) if latest_human else None
        )
        if confirmation is None:
            history.append(
                AIMessage(
                    content=(
                        "I could not determine your choice. Please answer yes, no, or cancel. "
                        "No external system will be changed; this only controls the dry-run."
                    )
                )
            )
            append_audit_event(
                state,
                node_name="mind_care",
                summary="Requested explicit confirmation",
            )
            return _with_observability(
                state,
                {"conversation_history": history, "awaiting_human_confirmation": True},
            )

        updates: Dict[str, Any] = {
            "awaiting_human_confirmation": False,
            "human_confirmation_response": confirmation,
        }
        if confirmation != "yes":
            updates["suggested_action"] = None
            history.append(
                AIMessage(content="Understood. I will not run the proposed dry-run action.")
            )
            updates["conversation_history"] = history
        append_audit_event(
            state,
            node_name="mind_care",
            summary="Processed explicit human confirmation",
            details={"response": confirmation},
        )
        return _with_observability(state, updates)

    if state.get("suggested_action"):
        action = state["suggested_action"]
        plan = str(action.get("tool_input", {}).get("plan", "the proposed adjustment"))
        prompt = (
            f"Proposed dry-run plan: {plan}\n"
            "No calendar, task manager, or external service will be modified. "
            "Do you want me to simulate this action and record what would happen? "
            "Please answer yes, no, or cancel."
        )
        history.append(AIMessage(content=prompt))
        append_audit_event(
            state,
            node_name="mind_care",
            summary="Presented dry-run TaskRelief suggestion",
            details={"tool_name": action.get("tool_name", "unknown")},
        )
        return _with_observability(
            state,
            {"conversation_history": history, "awaiting_human_confirmation": True},
        )

    latest_human = _latest_human_message(state)
    stress_detected = _is_stress_detected(state)

    if (
        stress_detected
        and latest_human
        and not state.get("current_stressor")
        and len(str(latest_human.content).strip()) >= 6
        and not _looks_like_confirmation(str(latest_human.content))
    ):
        stressor = _extract_stressor_from_text(str(latest_human.content))
        if stressor:
            append_audit_event(
                state,
                node_name="mind_care",
                summary="Identified stressor from dialogue",
                details={"stressor": stressor, "physio_context": _physio_summary(state)},
            )
            return _with_observability(state, {"current_stressor": stressor})

    if (
        latest_human
        and not state.get("current_stressor")
        and len(str(latest_human.content).strip()) >= 6
        and not _looks_like_confirmation(str(latest_human.content))
    ):
        user_query = str(latest_human.content).strip()
        rag_snippets: list[str] = []
        if state.get("use_rag", True):
            try:
                rag_snippets = _retrieve_context(_build_rag_query(user_query, state), k=3)
            except Exception as exc:
                append_error(state, f"MindCare RAG retrieval failure: {exc}")

        system_prompt = (
            MIND_CARE_SYSTEM_PROMPT
            + "\n\nPhysioSense context (use cautiously):\n"
            + _physio_summary(state)
        )
        if rag_snippets:
            system_prompt += (
                "\n\nRetrieved professional guidance; paraphrase rather than copying:\n"
                + "\n---\n".join(rag_snippets)
            )
        try:
            reply = _generate_chat(
                messages=_chat_messages(state),
                system_prompt=system_prompt,
            ).strip()
        except Exception as exc:
            append_error(state, f"MindCare LLM failure: {exc}")
            reply = (
                "Thank you for sharing. What you are feeling matters. "
                "Would you like to identify the one task or situation that feels most pressing?"
            )
        if reply:
            history.append(AIMessage(content=reply))
        append_audit_event(
            state,
            node_name="mind_care",
            summary=(
                "Generated physiology-aware RAG response"
                if rag_snippets
                else "Generated physiology-aware response"
            ),
            details={"rag_snippets_count": len(rag_snippets)},
        )
        return _with_observability(
            state,
            {"conversation_history": history, "rag_context": rag_snippets},
        )

    if stress_detected and not state.get("current_stressor"):
        rag_query = _build_rag_query(
            "The user has an elevated model estimate but has not described a stressor.",
            state,
        )
        try:
            rag_snippets = (
                _retrieve_context(rag_query, k=3) if state.get("use_rag", True) else []
            )
        except Exception as exc:
            append_error(state, f"MindCare RAG retrieval failure: {exc}")
            rag_snippets = []

        system_prompt = MIND_CARE_SYSTEM_PROMPT + "\n\n" + _physio_summary(state)
        if rag_snippets:
            system_prompt += "\n\nSupporting evidence:\n" + "\n---\n".join(rag_snippets)
        try:
            reply = _generate_chat(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Write at most three sentences: cautiously acknowledge the elevated "
                            "model estimate, offer one brief evidence-grounded step, and ask one "
                            "focused question about the user's primary stressor. Do not diagnose."
                        ),
                    }
                ],
                system_prompt=system_prompt,
            ).strip()
        except Exception as exc:
            append_error(state, f"MindCare LLM failure: {exc}")
            reply = (
                "Your recent signals may indicate elevated stress, though this is not a diagnosis. "
                "If you feel comfortable, which task or situation feels most pressing right now?"
            )
        history.append(AIMessage(content=reply))
        append_audit_event(
            state,
            node_name="mind_care",
            summary="Initiated physiology-guided stressor exploration",
            details={"physio_context": _physio_summary(state)},
        )
        return _with_observability(
            state,
            {"conversation_history": history, "rag_context": rag_snippets},
        )

    append_audit_event(state, node_name="mind_care", summary="No new dialogue needed")
    return _with_observability(state, {})
