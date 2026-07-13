from __future__ import annotations

import unittest
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from smartstress_langgraph.nodes.mind_care_node import (
    _build_rag_query,
    _normalize_confirmation,
    mind_care_node,
)
from smartstress_langgraph.orchestration import (
    meta_reflect_node,
    reflect_on_state,
    route_after_orchestrator,
)


class MindCareTests(unittest.TestCase):
    def test_confirmation_parser_does_not_use_substrings(self) -> None:
        self.assertEqual(_normalize_confirmation("yes"), "yes")
        self.assertEqual(_normalize_confirmation("No thanks"), "no")
        self.assertIsNone(_normalize_confirmation("yesterday was difficult"))

    def test_invalid_confirmation_keeps_interrupt_active(self) -> None:
        updates = mind_care_node({
            "awaiting_human_confirmation": True,
            "conversation_history": [HumanMessage(content="maybe later")],
            "audit_trail": [],
            "error_log": [],
        })
        self.assertTrue(updates["awaiting_human_confirmation"])
        self.assertIn("yes, no, or cancel", updates["conversation_history"][-1].content)

    def test_decline_clears_proposal_and_routes_to_refinement(self) -> None:
        state = {
            "awaiting_human_confirmation": True,
            "conversation_history": [HumanMessage(content="no")],
            "suggested_action": {"tool_name": "mock", "tool_input": {}},
            "audit_trail": [],
        }
        updates = mind_care_node(state)
        state.update(updates)
        self.assertIsNone(updates["suggested_action"])
        self.assertTrue(updates["refinement_requested"])
        state.update(meta_reflect_node(state))
        self.assertEqual(state["orchestration_decision"], "refine")
        self.assertEqual(route_after_orchestrator(state), "wait_for_human_input")

    def test_refinement_feedback_is_saved_before_replanning(self) -> None:
        state = {
            "refinement_requested": True,
            "human_confirmation_response": "no",
            "current_stressor": "project review",
            "conversation_history": [
                HumanMessage(content="Make it shorter and schedule it after lunch")
            ],
            "user_preferences": {},
            "audit_trail": [],
        }
        updates = mind_care_node(state)
        state.update(updates)
        self.assertFalse(updates["refinement_requested"])
        self.assertIn("shorter", updates["user_preferences"]["intervention_feedback"])
        reflection = reflect_on_state(state)
        self.assertEqual(reflection.decision, "propose")

    def test_rag_query_contains_shap_driver_context(self) -> None:
        query = _build_rag_query(
            "deadline pressure",
            {
                "current_stress_prob": 0.8,
                "stress_threshold": 0.5,
                "stress_detected": True,
                "physio_top_drivers": [
                    {
                        "feature": "std_hrv",
                        "direction": "increases_stress_probability",
                    }
                ],
            },
        )
        self.assertIn("deadline pressure", query)
        self.assertIn("std hrv", query)
        self.assertIn("not diagnoses", query)

    @patch(
        "smartstress_langgraph.nodes.mind_care_node._generate_chat",
        return_value="Your signals may be elevated. Which task feels most pressing?",
    )
    @patch(
        "smartstress_langgraph.nodes.mind_care_node._retrieve_context",
        return_value=["Take a brief paced pause. [source: guidance]"],
    )
    def test_elevated_sensor_flow_retrieves_with_shap_context(
        self,
        retrieve_context,
        generate_chat,
    ) -> None:
        updates = mind_care_node({
            "current_stress_prob": 0.8,
            "stress_threshold": 0.5,
            "stress_detected": True,
            "physio_top_drivers": [
                {
                    "feature": "rmssd",
                    "direction": "increases_stress_probability",
                }
            ],
            "conversation_history": [],
            "use_rag": True,
            "audit_trail": [],
        })
        rag_query = retrieve_context.call_args.args[0]
        self.assertIn("rmssd", rag_query)
        self.assertEqual(len(updates["rag_context"]), 1)
        generate_chat.assert_called_once()

    @patch(
        "smartstress_langgraph.nodes.mind_care_node._extract_stressor_from_text",
        return_value="project deadline",
    )
    def test_model_decision_triggers_stressor_extraction_below_point_nine(
        self,
        extract_stressor,
    ) -> None:
        updates = mind_care_node({
            "current_stress_prob": 0.60,
            "stress_threshold": 0.5,
            "stress_detected": True,
            "conversation_history": [
                HumanMessage(content="The project deadline is overwhelming me")
            ],
            "audit_trail": [],
        })
        self.assertEqual(updates["current_stressor"], "project deadline")
        extract_stressor.assert_called_once()


class OrchestrationTests(unittest.TestCase):
    def test_routes_consent_and_stressor_states_explicitly(self) -> None:
        cases = (
            ({"awaiting_human_confirmation": True}, "wait_for_human_input"),
            ({"human_confirmation_response": "yes"}, "execute_tool"),
            ({"current_stressor": "deadline"}, "propose_relief_action"),
        )
        for initial_state, expected_route in cases:
            with self.subTest(expected_route=expected_route):
                state = dict(initial_state)
                state.update(meta_reflect_node(state))
                self.assertEqual(route_after_orchestrator(state), expected_route)

    def test_reflection_persists_all_paper_signals(self) -> None:
        state = {
            "current_stress_prob": 0.8,
            "stress_threshold": 0.5,
            "stress_detected": True,
            "current_stressor": "deadline",
            "rag_context": ["grounded passage"],
        }
        reflection = reflect_on_state(state)
        self.assertEqual(reflection.decision, "propose")
        self.assertEqual(reflection.signals["physiological_urgency"], 0.8)
        self.assertTrue(reflection.signals["semantic_specificity"])
        self.assertEqual(reflection.signals["grounding_count"], 1)


if __name__ == "__main__":
    unittest.main()
