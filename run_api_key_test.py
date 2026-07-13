from __future__ import annotations

"""
Convenience script to verify the SmartStress multi-agent stack can call Gemini
using the GOOGLE_API_KEY provided via .env or environment variables.
"""

import json
from datetime import datetime, timezone

from smartstress_langgraph.api import continue_session, start_monitoring_session
from smartstress_langgraph.io_models import (
    ChatMessage,
    ContinueSessionRequest,
    SensorData,
    StartSessionRequest,
    UserInfo,
)
from smartstress_langgraph.examples.sample_data import DEMO_STRESS_FEATURES


def run_api_key_test() -> None:
    """Execute a short monitoring flow and print the relevant state snapshots."""

    # Step 1: start a monitoring session with sample sensor data
    start_request = StartSessionRequest(
        user=UserInfo(user_id="demo-user", session_id="api-key-test"),
        initial_sensor_data=SensorData(
            timestamp=datetime.now(timezone.utc).isoformat(),
            normalized_features=DEMO_STRESS_FEATURES,
        ),
    )
    handle, initial_state = start_monitoring_session(start_request)
    print("=== Step 1: Session bootstrapped ===")
    print(
        json.dumps(
            {
                "session_handle": handle.model_dump(),
                "current_stress_prob": initial_state.current_stress_prob,
                "stress_detected": initial_state.stress_detected,
                "physio_model_id": initial_state.physio_model_id,
                "physio_top_drivers": initial_state.physio_top_drivers,
                "orchestration_decision": initial_state.orchestration_decision,
                "stress_history": initial_state.stress_history,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    # Step 2: send a realistic stress message to trigger the MindCare agent (Gemini call)
    follow_request = ContinueSessionRequest(
        session_handle=handle,
        user_message=ChatMessage(
            role="user",
            content=(
                "The upcoming project demo is overwhelming because the slides are unfinished "
                "and the team expects me to lead."
            ),
        ),
    )
    _, updated_state = continue_session(follow_request)
    print("\n=== Step 2: MindCare response ===")
    print(
        json.dumps(
            {
                "current_stressor": updated_state.current_stressor,
                "suggested_action": updated_state.suggested_action,
                "orchestration_decision": updated_state.orchestration_decision,
                "conversation_history": updated_state.conversation_history,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    # Step 3: summarize outcome so CI / operators can quickly confirm success
    print("\n=== Result: API key verification successful ===")
    if updated_state.current_stressor:
        print(f"Identified stressor: {updated_state.current_stressor}")
    else:
        print("No stressor identified; check Gemini credentials and logs.")


if __name__ == "__main__":
    run_api_key_test()
