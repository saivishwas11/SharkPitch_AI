from graph.state_types import PitchState
from utils.llm_utils import run_shark_persona


def visionary_shark_node(state: PitchState) -> PitchState:
    """Visionary shark analysis focusing on innovation, market disruption, and growth potential."""
    if "shark_panel" not in state:
        state["shark_panel"] = {}
    
    state["shark_panel"]["visionary"] = run_shark_persona(
        name="Visionary Shark",
        style="forward-thinking, innovative, focused on market disruption and growth potential",
        transcript=state.get("transcript", ""),
        delivery_score=state.get("voice_stats", {}).get("delivery_score", 0),
        content_analysis=state.get("content_analysis", {})
    )
    return state
