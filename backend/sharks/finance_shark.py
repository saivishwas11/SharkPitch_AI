from backend.graph.state_types import PitchState
from backend.utils.llm_utils import run_shark_persona

def finance_shark_node(state: PitchState) -> dict:
    """Finance shark analysis focusing on numbers, unit economics, and profitability."""
    result = run_shark_persona(
        name="Finance Shark",
        style="financially astute, focused on unit economics, margins, and profitability",
        transcript=state.get("transcript", ""),
        delivery_score=state.get("voice_stats", {}).get("delivery_score", 0),
        content_analysis=state.get("content_analysis", {})
    )
    
    return {
        "shark_panel": {"finance": result},
        "shark_finance": result
    }
