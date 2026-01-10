from backend.graph.state_types import PitchState
from backend.utils.llm_utils import run_shark_persona

def customer_shark_node(state: PitchState) -> dict:
    """Customer shark analysis focusing on product-market fit and user experience."""
    result = run_shark_persona(
        name="Customer Shark",
        style="customer-obsessed, focused on product-market fit, user experience, and adoption",
        transcript=state.get("transcript", ""),
        delivery_score=state.get("voice_stats", {}).get("delivery_score", 0),
        content_analysis=state.get("content_analysis", {})
    )
    
    return {
        "shark_panel": {"customer": result},
        "shark_customer": result
    }
