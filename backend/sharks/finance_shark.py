from backend.graph.state_types import PitchState
from backend.utils.llm_utils import run_shark_persona


def finance_shark_node(state: PitchState) -> PitchState:
    """Finance shark analysis focusing on numbers, unit economics, and profitability."""
    if "shark_panel" not in state:
        state["shark_panel"] = {}
    
    result = run_shark_persona(
        name="Finance Shark",
        style="financially astute, focused on unit economics, margins, and profitability",
        transcript=state.get("transcript", ""),
        delivery_score=state.get("voice_stats", {}).get("delivery_score", 0),
        content_analysis=state.get("content_analysis", {})
    )
    
    # Store in both locations for aggregator and direct access
    state["shark_panel"]["finance"] = result
    state["shark_finance"] = result
    return state
