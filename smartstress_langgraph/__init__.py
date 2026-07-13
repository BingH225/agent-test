"""SmartStress LangGraph backend SDK.

Public graph helpers are loaded lazily so independent components such as the
physiological model do not require the full orchestration stack at import time.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "build_app",
    "start_monitoring_session",
    "continue_session",
    "ingest_documents",
]


def __getattr__(name: str) -> Any:
    if name == "build_app":
        return getattr(import_module(".graph", __name__), name)
    if name in {"start_monitoring_session", "continue_session", "ingest_documents"}:
        return getattr(import_module(".api", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


