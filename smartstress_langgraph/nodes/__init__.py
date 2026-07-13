"""LangGraph nodes implementing PhysioSense, MindCare, and TaskRelief."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "physio_sense_node",
    "mind_care_node",
    "task_relief_propose_node",
    "execute_tool_node",
]


def __getattr__(name: str) -> Any:
    if name == "physio_sense_node":
        return getattr(import_module(".physio_sense_node", __name__), name)
    if name == "mind_care_node":
        return getattr(import_module(".mind_care_node", __name__), name)
    if name in {"task_relief_propose_node", "execute_tool_node"}:
        return getattr(import_module(".task_relief_nodes", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


