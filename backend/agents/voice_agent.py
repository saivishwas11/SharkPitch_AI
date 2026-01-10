import logging
from backend.graph.state_types import PitchState
from backend.utils.voice_analysis import analyze_voice

logger = logging.getLogger(__name__)


def voice_agent_node(state: PitchState) -> dict:
    """
    Analyze the prepared audio for prosody / delivery features and return updates.
    Expects state["clean_audio_path"].
    """
    audio_path = state.get("clean_audio_path")
    if not audio_path:
        return {"voice_error": "clean_audio_path missing"}

    try:
        vstats = analyze_voice(audio_path)
        logger.info("Voice stats extracted: delivery_score=%s", vstats.get("delivery_score"))
        return {"voice_stats": vstats}
    except Exception as e:
        logger.exception("Voice analysis failed: %s", e)
        return {"voice_error": str(e)}
