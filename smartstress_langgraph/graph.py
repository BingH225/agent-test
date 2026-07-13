from __future__ import annotations

from langgraph.graph import StateGraph, END

from .nodes import (
    physio_sense_node,
    mind_care_node,
    task_relief_propose_node,
    execute_tool_node,
)
from .state import SmartStressState
from .orchestration import meta_reflect_node, route_after_orchestrator


def build_workflow_graph() -> StateGraph:
    workflow = StateGraph(SmartStressState)

    # Nodes
    workflow.add_node("physio_sense", physio_sense_node)
    workflow.add_node("mind_care", mind_care_node)
    workflow.add_node("task_relief_propose", task_relief_propose_node)
    workflow.add_node("execute_tool", execute_tool_node)
    workflow.add_node("meta_reflective_orchestrator", meta_reflect_node)
    workflow.add_node("wait_for_human_input", lambda state: state)

    # Entry / basic edges
    workflow.set_entry_point("physio_sense")
    workflow.add_edge("physio_sense", "mind_care")
    workflow.add_edge("mind_care", "meta_reflective_orchestrator")
    workflow.add_edge("task_relief_propose", "mind_care")
    workflow.add_edge("execute_tool", "mind_care")

    # Conditional routing from the explicit meta-reflective orchestrator.
    workflow.add_conditional_edges(
        "meta_reflective_orchestrator",
        route_after_orchestrator,
        {
            "wait_for_human_input": "wait_for_human_input",
            "execute_tool": "execute_tool",
            "propose_relief_action": "task_relief_propose",
            "end": END,
        },
    )

    return workflow


_APP = None


import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver as SqliteSaver

# ...

def build_app():
    """
    Compile (once) and return a LangGraph app with HITL interrupt configuration.
    """
    global _APP
    if _APP is None:
        workflow = build_workflow_graph()
        
        # Create DB connection (check_same_thread=False is needed for FastAPI)
        conn = sqlite3.connect("smartstress.db", check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        
        _APP = workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=["wait_for_human_input"],
        )
    return _APP



