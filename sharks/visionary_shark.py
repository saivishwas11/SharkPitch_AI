from graph.state_types import PitchState
from utils.llm_utils import run_shark_persona


def visionary_shark_node(state: PitchState) -> PitchState:
    transcript = state["transcript"]
    delivery_score = state["voice_stats"]["delivery_score"]
    content_analysis = state["content_analysis"]

    state["shark_visionary"] = run_shark_persona(
        name="The Visionary",
        style="optimistic, big-picture, innovation-focused",
        transcript=transcript,
        delivery_score=delivery_score,
        content_analysis=content_analysis,
    )
    return state
