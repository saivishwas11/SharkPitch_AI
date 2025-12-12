from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END

from graph.state_types import PitchState

# agents (top-level package)
from agents.audio_agent import audio_agent_node
from agents.voice_agent import voice_agent_node
from agents.asr_agent import asr_agent_node

# content is top-level (not nested inside agents)
from content.master_agent import master_agent_node

# sharks are top-level
from sharks.visionary_shark import visionary_shark_node
from sharks.finance_shark import finance_shark_node
from sharks.customer_shark import customer_shark_node
from sharks.skeptic_shark import skeptic_shark_node
from sharks.shark_aggregator import shark_aggregator_node


def build_graph():
    workflow = StateGraph(PitchState)

    # Register nodes
    workflow.add_node("audio_agent", audio_agent_node)
    workflow.add_node("voice_agent", voice_agent_node)
    workflow.add_node("asr_agent", asr_agent_node)

    workflow.add_node("master_agent", master_agent_node)

    workflow.add_node("visionary_shark", visionary_shark_node)
    workflow.add_node("finance_shark", finance_shark_node)
    workflow.add_node("customer_shark", customer_shark_node)
    workflow.add_node("skeptic_shark", skeptic_shark_node)
    workflow.add_node("shark_aggregator", shark_aggregator_node)

    # Flow: audio -> voice -> asr
    workflow.set_entry_point("audio_agent")
    workflow.add_edge("audio_agent", "voice_agent")
    workflow.add_edge("voice_agent", "asr_agent")

    # ASR -> Master content agent
    workflow.add_edge("asr_agent", "master_agent")

    # Sequential sharks: master -> visionary -> finance -> customer -> skeptic -> aggregator
    workflow.add_edge("master_agent", "visionary_shark")
    workflow.add_edge("visionary_shark", "finance_shark")
    workflow.add_edge("finance_shark", "customer_shark")
    workflow.add_edge("customer_shark", "skeptic_shark")
    workflow.add_edge("skeptic_shark", "shark_aggregator")

    workflow.add_edge("shark_aggregator", END)

    app = workflow.compile()
    return app
