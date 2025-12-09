from langgraph.graph import StateGraph, END

from graph.state_types import PitchState

from agents.audio_agent import audio_agent_node
from agents.voice_agent import voice_agent_node
from agents.asr_agent import asr_agent_node


def build_graph():
    """
    Full pipeline: voice + ASR + single content agent + sharks.
    """
    from content.master_agent import content_master_node

    from sharks.visionary_shark import visionary_shark_node
    from sharks.finance_shark import finance_shark_node
    from sharks.customer_shark import customer_shark_node
    from sharks.skeptic_shark import skeptic_shark_node
    from sharks.shark_aggregator import shark_aggregator_node

    workflow = StateGraph(PitchState)

    # Register nodes (agents)
    workflow.add_node("audio_agent", audio_agent_node)
    workflow.add_node("voice_agent", voice_agent_node)
    workflow.add_node("asr_agent", asr_agent_node)

    workflow.add_node("content_master", content_master_node)

    workflow.add_node("visionary_shark", visionary_shark_node)
    workflow.add_node("finance_shark", finance_shark_node)
    workflow.add_node("customer_shark", customer_shark_node)
    workflow.add_node("skeptic_shark", skeptic_shark_node)
    workflow.add_node("shark_aggregator", shark_aggregator_node)

    # Flow: audio → voice → ASR
    workflow.set_entry_point("audio_agent")
    workflow.add_edge("audio_agent", "voice_agent")
    workflow.add_edge("voice_agent", "asr_agent")

    # After ASR: content master
    workflow.add_edge("asr_agent", "content_master")

    # After content master: parallel sharks
    workflow.add_edge("content_master", "visionary_shark")
    workflow.add_edge("content_master", "finance_shark")
    workflow.add_edge("content_master", "customer_shark")
    workflow.add_edge("content_master", "skeptic_shark")

    # All sharks converge into shark_aggregator
    workflow.add_edge("visionary_shark", "shark_aggregator")
    workflow.add_edge("finance_shark", "shark_aggregator")
    workflow.add_edge("customer_shark", "shark_aggregator")
    workflow.add_edge("skeptic_shark", "shark_aggregator")

    workflow.add_edge("shark_aggregator", END)

    app = workflow.compile()
    return app


def build_voice_graph():
    """
    Minimal pipeline: audio load/clean -> voice analysis -> ASR transcript.
    """
    workflow = StateGraph(PitchState)

    workflow.add_node("audio_agent", audio_agent_node)
    workflow.add_node("voice_agent", voice_agent_node)
    workflow.add_node("asr_agent", asr_agent_node)

    workflow.set_entry_point("audio_agent")
    workflow.add_edge("audio_agent", "voice_agent")
    workflow.add_edge("voice_agent", "asr_agent")
    workflow.add_edge("asr_agent", END)

    return workflow.compile()


def build_content_graph():
    """
    Voice + ASR + single content agent (no sharks).
    """
    from content.master_agent import content_master_node

    workflow = StateGraph(PitchState)

    workflow.add_node("audio_agent", audio_agent_node)
    workflow.add_node("voice_agent", voice_agent_node)
    workflow.add_node("asr_agent", asr_agent_node)

    workflow.add_node("content_master", content_master_node)

    workflow.set_entry_point("audio_agent")
    workflow.add_edge("audio_agent", "voice_agent")
    workflow.add_edge("voice_agent", "asr_agent")

    workflow.add_edge("asr_agent", "content_master")

    workflow.add_edge("content_master", END)

    return workflow.compile()
