from __future__ import annotations

"""
Minimal demo showing how to run one monitoring step.

Usage (from project root):
    python -m smartstress_langgraph.examples.demo_session
"""

from datetime import datetime, timezone

from ..api import continue_session, start_monitoring_session
from ..io_models import (
    ChatMessage,
    ContinueSessionRequest,
    SensorData,
    StartSessionRequest,
    UserInfo,
)
from .sample_data import DEMO_STRESS_FEATURES


def main() -> None:
    req = StartSessionRequest(
        user=UserInfo(user_id="demo-user", session_id="demo-session"),
        initial_sensor_data=SensorData(
            timestamp=datetime.now(timezone.utc).isoformat(),
            normalized_features=DEMO_STRESS_FEATURES,
        ),
    )
    handle, state_view = start_monitoring_session(req)
    print("== Initial pass ==")
    print("Session handle:", handle.model_dump())
    print("State view:", state_view.model_dump())

    follow_up = ContinueSessionRequest(
        session_handle=handle,
        user_message=ChatMessage(
            role="user",
            content="The 3pm cross-team review is stressing me out and I still need to polish the slides.",
        ),
    )
    handle, updated_state = continue_session(follow_up)
    print("\n== After user message ==")
    print("State view:", updated_state.model_dump())


if __name__ == "__main__":
    main()


