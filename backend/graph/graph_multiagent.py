from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END

from backend.graph.state_types import PitchState

# agents (top-level package)
from backend.agents.audio_agent import audio_agent_node
from backend.agents.voice_agent import voice_agent_node
from backend.agents.asr_agent import asr_agent_node

# content is top-level (not nested inside agents)
from backend.content.master_agent import master_agent_node

# sharks are top-level
from backend.sharks.visionary_shark import visionary_shark_node
from backend.sharks.finance_shark import finance_shark_node
from backend.sharks.customer_shark import customer_shark_node
from backend.sharks.skeptic_shark import skeptic_shark_node

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

    # Flow: audio -> (voice, asr)
    workflow.set_entry_point("audio_agent")
    workflow.add_edge("audio_agent", "voice_agent")
    workflow.add_edge("audio_agent", "asr_agent")

    # ASR -> Master content agent
    workflow.add_edge("asr_agent", "master_agent")

    # Parallel sharks: master -> (visionary, finance, customer, skeptic)
    workflow.add_edge("master_agent", "visionary_shark")
    workflow.add_edge("master_agent", "finance_shark")
    workflow.add_edge("master_agent", "customer_shark")
    workflow.add_edge("master_agent", "skeptic_shark")

    # Connect voice and sharks to END
    workflow.add_edge("voice_agent", END)
    workflow.add_edge("visionary_shark", END)
    workflow.add_edge("finance_shark", END)
    workflow.add_edge("customer_shark", END)
    workflow.add_edge("skeptic_shark", END)

    app = workflow.compile()
    return app
