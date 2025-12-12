import logging
from graph.state_types import PitchState
from utils.voice_analysis import analyze_voice

logger = logging.getLogger(__name__)


def voice_agent_node(state: PitchState) -> PitchState:
    """
    Analyze the prepared audio for prosody / delivery features and write into state.
    Expects state["clean_audio_path"].
    """
    audio_path = state.get("clean_audio_path")
    if not audio_path:
        state["voice_error"] = "clean_audio_path missing"
        return state

    try:
        vstats = analyze_voice(audio_path)
        state["voice_stats"] = vstats
        logger.info("Voice stats extracted: delivery_score=%s", vstats.get("delivery_score"))
    except Exception as e:
        logger.exception("Voice analysis failed: %s", e)
        state["voice_error"] = str(e)

    return state
