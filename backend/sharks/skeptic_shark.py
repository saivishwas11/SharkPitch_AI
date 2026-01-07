from backend.graph.state_types import PitchState
from backend.utils.llm_utils import run_shark_persona


def skeptic_shark_node(state: PitchState) -> PitchState:
    """Skeptic shark analysis focusing on risks, challenges, and potential pitfalls."""
    if "shark_panel" not in state:
        state["shark_panel"] = {}
    
    result = run_shark_persona(
        name="Skeptic Shark",
        style="skeptical, focused on identifying risks, challenges, and potential pitfalls",
        transcript=state.get("transcript", ""),
        delivery_score=state.get("voice_stats", {}).get("delivery_score", 0),
        content_analysis=state.get("content_analysis", {})
    )
    
    # Store in both locations for aggregator and direct access
    state["shark_panel"]["skeptic"] = result
    state["shark_skeptic"] = result
    return state
