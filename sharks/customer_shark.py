from graph.state_types import PitchState
from utils.llm_utils import run_shark_persona


def customer_shark_node(state: PitchState) -> PitchState:
    transcript = state.get("transcript", "")
    delivery_score = state.get("voice_stats", {}).get("delivery_score", 0)
    content_analysis = state.get("content_analysis", {})

    state["shark_customer"] = run_shark_persona(
        name="The Customer Advocate",
        style="empathetic, focused on user pain and usability",
        transcript=transcript,
        delivery_score=delivery_score,
        content_analysis=content_analysis,
    )
    return state
