"""LLM client and prompt templates for SmartStress (Gemini-based)."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["get_chat_client", "embed_documents", "generate_chat", "prompts"]


def __getattr__(name: str) -> Any:
    if name == "prompts":
        return import_module(".prompts", __name__)
    if name in {"get_chat_client", "embed_documents", "generate_chat"}:
        return getattr(import_module(".client", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")



