from backend.graph.state_types import PitchState
from backend.utils.llm_utils import run_shark_persona

def skeptic_shark_node(state: PitchState) -> dict:
    """Skeptic shark analysis focusing on risks, challenges, and potential pitfalls."""
    result = run_shark_persona(
        name="Skeptic Shark",
        style="skeptical, focused on identifying risks, challenges, and potential pitfalls",
        transcript=state.get("transcript", ""),
        delivery_score=state.get("voice_stats", {}).get("delivery_score", 0),
        content_analysis=state.get("content_analysis", {})
    )
    
    return {
        "shark_panel": {"skeptic": result},
        "shark_skeptic": result
    }
