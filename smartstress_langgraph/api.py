from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .graph import build_app
from .io_models import (
    ContinueSessionRequest,
    SessionHandleModel,
    SmartStressStateView,
    StartSessionRequest,
)
from .state import SessionHandle, SmartStressState

APP = build_app()


def _blank_state(user_id: str, session_id: str) -> SmartStressState:
    return {
        "user_id": user_id,
        "session_id": session_id,
        "stress_history": [],
        "stress_timestamps": [],
        "physio_features": [],
        "physio_feature_map": {},
        "physio_raw_features": [],
        "physio_baseline_features": [],
        "physio_attributions": {},
        "physio_top_drivers": [],
        "conversation_history": [],
        "rag_context": [],
        "use_rag": True,
        "user_preferences": {},
        "external_side_effects": False,
        "awaiting_human_confirmation": False,
        "error_log": [],
        "audit_trail": [],
    }


def _build_initial_state(req: StartSessionRequest) -> SmartStressState:
    state = _blank_state(req.user.user_id, req.user.session_id)
    state["user_preferences"] = req.user.traits
    if req.initial_sensor_data:
        payload = req.initial_sensor_data.to_payload()
        payload["timestamp"] = req.initial_sensor_data.timestamp
        state["raw_sensor_input"] = payload
    return state


def _serialize_messages(messages: list[BaseMessage]) -> list[Dict[str, str]]:
    serialized: list[Dict[str, str]] = []
    for msg in messages:
        role = "assistant"
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = getattr(msg, "type", "assistant")
        serialized.append({"role": role, "content": getattr(msg, "content", "")})
    return serialized


def _state_to_view(state: SmartStressState) -> SmartStressStateView:
    return SmartStressStateView(
        user_id=state.get("user_id", ""),
        session_id=state.get("session_id", ""),
        current_stress_prob=state.get("current_stress_prob"),
        stress_detected=state.get("stress_detected"),
        stress_threshold=state.get("stress_threshold"),
        physio_model_id=state.get("physio_model_id"),
        physio_input_source=state.get("physio_input_source"),
        physio_feature_map=dict(state.get("physio_feature_map", {})),
        physio_attributions=dict(state.get("physio_attributions", {})),
        physio_top_drivers=list(state.get("physio_top_drivers", [])),
        stress_history=list(state.get("stress_history", [])),
        stress_timestamps=list(state.get("stress_timestamps", [])),
        current_stressor=state.get("current_stressor"),
        suggested_action=state.get("suggested_action"),
        tool_output=state.get("tool_output"),
        tool_execution_mode=state.get("tool_execution_mode"),
        external_side_effects=state.get("external_side_effects", False),
        awaiting_human_confirmation=state.get("awaiting_human_confirmation", False),
        human_confirmation_response=state.get("human_confirmation_response"),
        rag_context=list(state.get("rag_context", [])),
        error_log=list(state.get("error_log", [])),
        audit_trail=list(state.get("audit_trail", [])),
        conversation_history=_serialize_messages(state.get("conversation_history", [])),
    )


def _load_cached_state(handle: SessionHandle) -> SmartStressState:
    """
    Fetch state directly from the persistent graph checkpoint.
    """
    config = {"configurable": {"thread_id": handle.thread_id}}
    try:
        # APP is the compiled graph imported from .graph
        current_snapshot = APP.get_state(config)
        if current_snapshot.values:
            return current_snapshot.values
    except Exception:
        # No checkpoint exists yet, return blank
        pass
        
    return _blank_state(handle.user_id, handle.session_id)


def start_monitoring_session(
    req: StartSessionRequest,
) -> Tuple[SessionHandleModel, SmartStressStateView]:
    """
    Start a new monitoring loop for a user/session.

    External FastAPI backend can wrap this into an HTTP endpoint.
    """
    initial_state = _build_initial_state(req)
    handle = SessionHandle(
        user_id=req.user.user_id,
        session_id=req.user.session_id,
        thread_id=f"{req.user.user_id}:{req.user.session_id}",
        checkpoint_id=None,
        metadata={},
    )
    config = {"configurable": {"thread_id": handle.thread_id}}
    result = APP.invoke(deepcopy(initial_state), config=config)
    state = result if isinstance(result, dict) else result[0]
    return SessionHandleModel.from_handle(handle), _state_to_view(state)


def continue_session(
    req: ContinueSessionRequest,
) -> Tuple[SessionHandleModel, SmartStressStateView]:
    """
    Continue an existing session using a cached SessionHandle.
    """
    handle = req.session_handle.to_handle()
    state = _load_cached_state(handle)

    if req.sensor_data:
        payload = req.sensor_data.to_payload()
        payload["timestamp"] = req.sensor_data.timestamp
        state["raw_sensor_input"] = payload

    if req.user_message:
        history = list(state.get("conversation_history", []))
        if req.user_message.role.lower() == "assistant":
            history.append(AIMessage(content=req.user_message.content))
        else:
            history.append(HumanMessage(content=req.user_message.content))
        state["conversation_history"] = history

    config = {"configurable": {"thread_id": handle.thread_id}}
    result = APP.invoke(deepcopy(state), config=config)
    new_state = result if isinstance(result, dict) else result[0]
    return SessionHandleModel.from_handle(handle), _state_to_view(new_state)


def ingest_documents(folder_path: str, tags: list[str] | None = None) -> Dict[str, Any]:
    """
    High-level RAG ingestion API.
    """
    from .rag.ingestion import load_documents_from_folder, build_or_update_index

    docs = load_documents_from_folder(folder_path)
    if tags:
        for d in docs:
            d.tags.extend(tags)
    count = build_or_update_index(docs)
    return {"ingested": count}



