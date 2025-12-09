from graph.state_types import PitchState
from utils.llm_utils import run_shark_persona


def customer_shark_node(state: PitchState) -> PitchState:
    transcript = state["transcript"]
    delivery_score = state["voice_stats"]["delivery_score"]
    content_analysis = state["content_analysis"]

    state["shark_customer"] = run_shark_persona(
        name="The Customer Advocate",
        style="empathetic, focused on user pain and usability",
        transcript=transcript,
        delivery_score=delivery_score,
        content_analysis=content_analysis,
    )
    return state
