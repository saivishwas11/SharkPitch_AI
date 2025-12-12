from graph.state_types import PitchState
from utils.llm_utils import run_shark_persona


def finance_shark_node(state: PitchState) -> PitchState:
    transcript = state.get("transcript", "")
    delivery_score = state.get("voice_stats", {}).get("delivery_score", 0)
    content_analysis = state.get("content_analysis", {})

    state["shark_finance"] = run_shark_persona(
        name="The Finance Shark",
        style="cold, analytical, focused on margins and revenue",
        transcript=transcript,
        delivery_score=delivery_score,
        content_analysis=content_analysis,
    )
    return state
