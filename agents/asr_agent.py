import logging
from typing import Dict, Any

from utils.asr import transcribe_audio_safe  # safer wrapper that normalizes errors

logger = logging.getLogger(__name__)


def asr_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ASR agent node. Expects:
      - state["clean_audio_path"] OR state["input_path"]
      - optional overrides: asr_model, asr_language, asr_backend
    Writes:
      - state["transcript"], state["transcript_segments"], state["asr_text"], state["asr_segments"], state["asr_raw"]
      - state["asr_error"] on failure
    """
    audio_path = state.get("clean_audio_path") or state.get("input_path")
    if not audio_path:
        state["asr_error"] = "no audio path provided"
        return state

    model = state.get("asr_model", None)
    language = state.get("asr_language", None)
    backend = state.get("asr_backend", "groq")

    try:
        result = transcribe_audio_safe(
            audio_path=audio_path,
            model=model,
            language=language,
            timestamps=True,
            backend=backend,
        )
        # normalized shape from utils.asr: {"text": str, "segments": list|None, "raw": {...}}
        state["transcript"] = result.get("text", "") or ""
        state["transcript_segments"] = result.get("segments") or []
        state["asr_text"] = state["transcript"]  # backward-compat
        state["asr_segments"] = result.get("segments")
        state["asr_raw"] = result.get("raw")
        logger.info("ASR result length=%d", len(state["transcript"]))
    except Exception as e:
        logger.exception("ASR failed: %s", e)
        state["asr_error"] = str(e)

    return state
