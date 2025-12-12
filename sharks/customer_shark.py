from graph.state_types import PitchState
from utils.llm_utils import run_shark_persona


def customer_shark_node(state: PitchState) -> PitchState:
    """Customer shark analysis focusing on product-market fit and customer appeal."""
    if "shark_panel" not in state:
        state["shark_panel"] = {}
    
    state["shark_panel"]["customer"] = run_shark_persona(
        name="Customer Shark",
        style="customer-focused, concerned with product-market fit, user experience, and value proposition",
        transcript=state.get("transcript", ""),
        delivery_score=state.get("voice_stats", {}).get("delivery_score", 0),
        content_analysis=state.get("content_analysis", {})
    )
    return state
