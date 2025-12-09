# agents/asr_agent.py
import logging
from typing import Dict, Any
from utils.asr import transcribe_audio

logger = logging.getLogger(__name__)


def asr_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # State should contain a path to the prepared WAV (mono, correct sr)
    audio_path = state.get("clean_audio_path") or state.get("input_path")
    if not audio_path:
        state["asr_error"] = "no audio path provided"
        return state

    # Choose model per your config (defaults in utils/groq_asr)
    model = state.get("asr_model", None)  # optionally override per-job
    language = state.get("asr_language", None)  # e.g. "en"
    backend = state.get("asr_backend", "groq")  # "groq" or "gemini"

    try:
        result = transcribe_audio(
            audio_path=audio_path,
            model=model or None,
            language=language,
            timestamps=True,
            backend=backend,
        )
        state["transcript"] = result.get("text") or ""
        state["transcript_segments"] = result.get("segments") or []
        state["asr_text"] = result.get("text") or ""  # Keep for backward compatibility
        state["asr_segments"] = result.get("segments")
        state["asr_raw"] = result.get("raw")
        logger.info("ASR result length=%d", len(state["transcript"]))
    except Exception as e:
        logger.exception("ASR failed: %s", e)
        state["asr_error"] = str(e)
    return state
