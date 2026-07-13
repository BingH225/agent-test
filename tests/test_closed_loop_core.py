from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


try:
    from langchain_core.messages import AIMessage, HumanMessage
except ImportError:
    # The model-only conda environment intentionally lacks LangChain. These
    # minimal message objects let the core closed-loop logic run there while
    # the application environment continues to use the real LangChain classes.
    @dataclass
    class _Message:
        content: str

    AIMessage = type("AIMessage", (_Message,), {})
    HumanMessage = type("HumanMessage", (_Message,), {})
    langchain_core = types.ModuleType("langchain_core")
    messages_module = types.ModuleType("langchain_core.messages")
    messages_module.BaseMessage = _Message
    messages_module.AIMessage = AIMessage
    messages_module.HumanMessage = HumanMessage
    langchain_core.messages = messages_module
    sys.modules["langchain_core"] = langchain_core
    sys.modules["langchain_core.messages"] = messages_module

from smartstress_langgraph.nodes.mind_care_node import mind_care_node
from smartstress_langgraph.nodes.physio_sense_node import physio_sense_node
from smartstress_langgraph.nodes.task_relief_nodes import (
    execute_tool_node,
    task_relief_propose_node,
)
from smartstress_langgraph.orchestration import (
    meta_reflect_node,
    route_after_orchestrator,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "wesad_attention_v1_golden.json"


class ClosedLoopCoreTests(unittest.TestCase):
    @patch(
        "smartstress_langgraph.nodes.task_relief_nodes._generate_chat",
        return_value="Move the review to 15:00 and reserve a 10-minute preparation break.",
    )
    @patch(
        "smartstress_langgraph.nodes.mind_care_node._extract_stressor_from_text",
        return_value="project review deadline",
    )
    def test_feature_inference_to_consented_dry_run(
        self,
        extract_stressor,
        generate_plan,
    ) -> None:
        stress_sample = next(
            sample
            for sample in json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["samples"]
            if sample["wesad_label"] == 2
        )
        state = {
            "raw_sensor_input": {
                "timestamp": "2026-07-13T18:00:00Z",
                "normalized_features": stress_sample["features"],
            },
            "stress_history": [],
            "stress_timestamps": [],
            "conversation_history": [],
            "rag_context": [],
            "use_rag": True,
            "user_preferences": {},
            "audit_trail": [],
            "error_log": [],
        }

        state.update(physio_sense_node(state))
        self.assertTrue(state["stress_detected"])
        self.assertEqual(len(state["physio_top_drivers"]), 3)

        state["conversation_history"].append(
            HumanMessage(content="The project review deadline is overwhelming me")
        )
        state.update(mind_care_node(state))
        self.assertEqual(state["current_stressor"], "project review deadline")
        state.update(meta_reflect_node(state))
        self.assertEqual(route_after_orchestrator(state), "propose_relief_action")

        state.update(task_relief_propose_node(state))
        self.assertEqual(state["suggested_action"]["execution_mode"], "dry_run")
        state.update(mind_care_node(state))
        state.update(meta_reflect_node(state))
        self.assertEqual(route_after_orchestrator(state), "wait_for_human_input")

        state["conversation_history"].append(HumanMessage(content="yes"))
        state.update(mind_care_node(state))
        state.update(meta_reflect_node(state))
        self.assertEqual(route_after_orchestrator(state), "execute_tool")

        state.update(execute_tool_node(state))
        self.assertIn("[DRY-RUN]", state["tool_output"])
        self.assertFalse(state["external_side_effects"])
        self.assertIsNone(state["current_stressor"])
        extract_stressor.assert_called_once()
        generate_plan.assert_called_once()


if __name__ == "__main__":
    unittest.main()
