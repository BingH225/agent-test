from __future__ import annotations

import unittest
from unittest.mock import patch

from smartstress_langgraph.nodes.task_relief_nodes import (
    DRY_RUN_TOOL_NAME,
    execute_tool_node,
    task_relief_propose_node,
)


class TaskReliefTests(unittest.TestCase):
    @patch(
        "smartstress_langgraph.nodes.task_relief_nodes._generate_chat",
        return_value="Move the review block to 15:00 and reserve a 10-minute break first.",
    )
    def test_proposal_is_explicitly_side_effect_free(self, generate_chat) -> None:
        updates = task_relief_propose_node({
            "current_stressor": "project review",
            "physio_top_drivers": [{"feature": "std_hrv"}],
            "rag_context": ["Use brief workload segmentation. [source: guide-1]"],
            "audit_trail": [],
        })
        action = updates["suggested_action"]
        self.assertEqual(action["tool_name"], DRY_RUN_TOOL_NAME)
        self.assertEqual(action["execution_mode"], "dry_run")
        self.assertFalse(action["tool_input"]["external_side_effects"])
        prompt = generate_chat.call_args.kwargs["messages"][0]["content"]
        self.assertIn("std_hrv", prompt)
        self.assertIn("guide-1", prompt)
        self.assertEqual(len(action["tool_input"]["grounding_sources"]), 1)

    def test_consent_only_simulates_allowlisted_action(self) -> None:
        updates = execute_tool_node({
            "suggested_action": {
                "tool_name": DRY_RUN_TOOL_NAME,
                "execution_mode": "dry_run",
                "tool_input": {"plan": "Reserve a short break"},
            },
            "human_confirmation_response": "yes",
            "conversation_history": [],
            "audit_trail": [],
        })
        self.assertIn("[DRY-RUN]", updates["tool_output"])
        self.assertFalse(updates["external_side_effects"])
        self.assertIn("No calendar", updates["conversation_history"][-1].content)

    def test_non_allowlisted_action_is_blocked(self) -> None:
        updates = execute_tool_node({
            "suggested_action": {
                "tool_name": "real_calendar_update",
                "execution_mode": "dry_run",
                "tool_input": {},
            },
            "human_confirmation_response": "yes",
            "error_log": [],
            "audit_trail": [],
        })
        self.assertNotIn("tool_output", updates)
        self.assertFalse(updates["external_side_effects"])
        self.assertIn("Blocked non-allowlisted", updates["error_log"][-1])

    def test_missing_consent_skips_dry_run(self) -> None:
        updates = execute_tool_node({
            "suggested_action": {
                "tool_name": DRY_RUN_TOOL_NAME,
                "execution_mode": "dry_run",
                "tool_input": {},
            },
            "human_confirmation_response": "no",
            "audit_trail": [],
        })
        self.assertNotIn("tool_output", updates)
        self.assertFalse(updates["external_side_effects"])


if __name__ == "__main__":
    unittest.main()
