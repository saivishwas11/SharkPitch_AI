"""
asr.py

Groq Whisper–only ASR implementation.

Features:
- Groq Whisper transcription (sync)
- Full transcript always included
- Word-level timestamps (optional)
- Retry with exponential backoff
- Concurrency-safe (semaphore)
- Clean, normalized output

NO httpx
"""

import os
import time
import random
import logging
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from dotenv import load_dotenv

import groq  # Official Groq SDK

logger = logging.getLogger(__name__)
load_dotenv()
# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

MODEL_NAME = os.getenv("GROQ_ASR_MODEL", "whisper-large-v3")
MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "5"))
ASR_CONCURRENCY = int(os.getenv("GROQ_ASR_CONCURRENCY", "1"))

_ASR_SEMAPHORE = threading.Semaphore(max(1, ASR_CONCURRENCY))


# ------------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------------

@dataclass
class TranscriptionSegment:
    text: str
    start: float
    end: float
    logprob: float
    words: Optional[List[Dict[str, Any]]] = None


@dataclass
class TranscriptionResult:
    transcript: str
    language: str
    duration: float
    segments: List[TranscriptionSegment]
    model: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transcript": self.transcript,
            "text": self.transcript,  # backward compatibility
            "language": self.language,
            "duration": self.duration,
            "model": self.model,
            "segments": [
                {
                    "text": s.text,
                    "start": s.start,
                    "end": s.end,
                    "logprob": s.logprob,
                    "words": s.words or [],
                }
                for s in self.segments
            ],
        }


# ------------------------------------------------------------------
# Errors
# ------------------------------------------------------------------

class ASRError(RuntimeError):
    pass


# ------------------------------------------------------------------
# Retry helper
# ------------------------------------------------------------------

def _with_backoff(fn: Callable[[], Any]) -> Any:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt >= MAX_RETRIES:
                break
            delay = min(2 ** attempt, 30)
            delay += random.uniform(0, 0.3)
            logger.warning(
                "Groq ASR retry %d/%d after %.2fs",
                attempt,
                MAX_RETRIES,
                delay,
            )
            time.sleep(delay)
    raise ASRError(f"ASR failed after retries: {last_exc}") from last_exc


# ------------------------------------------------------------------
# Response normalization
# ------------------------------------------------------------------

def _normalize_response(raw: Any, language: str) -> Dict[str, Any]:
    transcript = getattr(raw, "text", "") or ""
    duration = float(getattr(raw, "duration", 0.0) or 0.0)
    segments_raw = getattr(raw, "segments", None)

    segments: List[TranscriptionSegment] = []

    if isinstance(segments_raw, list):
        for seg in segments_raw:
            segments.append(
                TranscriptionSegment(
                    text=seg.get("text", ""),
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    logprob=float(seg.get("avg_logprob", 0.0)),
                    words=seg.get("words"),
                )
            )

    result = TranscriptionResult(
        transcript=transcript,
        language=language,
        duration=duration,
        segments=segments,
        model=MODEL_NAME,
    )

    return result.to_dict()


# ------------------------------------------------------------------
# Core ASR (Groq Whisper)
# ------------------------------------------------------------------

def transcribe_audio(
    audio_path: str,
    language: str = "en",
    word_timestamps: bool = True,
) -> Dict[str, Any]:

    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)

    client = groq.Groq(api_key=GROQ_API_KEY)

    def _call():
        with open(audio_path, "rb") as f:
            return client.audio.transcriptions.create(
                file=f,
                model=MODEL_NAME,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["word"] if word_timestamps else None,
            )

    with _ASR_SEMAPHORE:
        raw = _with_backoff(_call)

    return _normalize_response(raw, language)


# ------------------------------------------------------------------
# Safe wrapper
# ------------------------------------------------------------------

def transcribe_audio_safe(
    audio_path: str,
    language: str = "en",
    word_timestamps: bool = True,
    model: Optional[str] = None,  # Accept but ignore for now
    timestamps: bool = True,      # Accept but ignore (use word_timestamps)
    backend: str = "groq",        # Accept but ignore (only groq supported)
) -> Dict[str, Any]:
    """
    Safe wrapper for transcribe_audio with additional parameters for compatibility.
    
    Args:
        audio_path: Path to audio file
        language: Language code (default: "en")
        word_timestamps: Whether to include word-level timestamps
        model: Model name (ignored, uses GROQ_ASR_MODEL from env)
        timestamps: Whether to include timestamps (ignored, uses word_timestamps)
        backend: ASR backend (ignored, only groq supported)
    """
    try:
        return transcribe_audio(
            audio_path=audio_path,
            language=language,
            word_timestamps=word_timestamps or timestamps,
        )
    except Exception as e:
        raise ASRError(f"Groq ASR failed: {e}") from e
