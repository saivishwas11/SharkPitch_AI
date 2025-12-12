import os
import time
import random
import threading
import logging
from typing import Optional, Dict, Any, Callable, Tuple

import google.generativeai as genai
from .rate_limiter import groq_rate_limiter, RateLimitConfig

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Optional dependencies
try:
    import httpx
except ImportError:
    httpx = None
    logger.debug("httpx not available")

try:
    import groq
except ImportError:
    groq = None
    logger.debug("groq SDK not available")

GROQ_TRANSCRIPT_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"
DEFAULT_MODEL = os.environ.get("GROQ_ASR_MODEL", "whisper-large-v3")

DEFAULT_RATE_LIMIT = int(os.environ.get("GROQ_RATE_LIMIT_PER_MINUTE", "30"))
DEFAULT_MAX_RETRIES = int(os.environ.get("GROQ_MAX_RETRIES", "5"))
DEFAULT_TIMEOUT = int(os.environ.get("GROQ_TIMEOUT_SECONDS", "120"))

_ASR_CONCURRENCY = int(os.environ.get("GROQ_ASR_CONCURRENCY", "1"))
_asr_semaphore = threading.Semaphore(max(1, _ASR_CONCURRENCY))


class GroqASRError(RuntimeError):
    pass


def _parse_groq_response(resp_json: Dict[str, Any]) -> Dict[str, Any]:
    text = ""
    segments = None

    if not isinstance(resp_json, dict):
        return {"text": str(resp_json), "segments": None, "raw": resp_json}

    if "text" in resp_json and isinstance(resp_json["text"], str):
        text = resp_json["text"]
        segments = resp_json.get("segments")

    elif "choices" in resp_json:
        first = resp_json["choices"][0]
        msg = first.get("message") if isinstance(first, dict) else str(first)
        if isinstance(msg, dict):
            text = msg.get("content") or ""
        else:
            text = str(msg)
        segments = resp_json.get("segments")

    else:
        text = resp_json.get("text") or resp_json.get("transcript") or ""
        segments = resp_json.get("segments") or resp_json.get("timestamps")

    return {"text": text, "segments": segments, "raw": resp_json}


def _request_with_backoff(request_fn: Callable[[], Tuple[int, Any]],
                          max_attempts: int = DEFAULT_MAX_RETRIES,
                          base_delay: float = 1.0,
                          max_delay: float = 60.0,
                          jitter: float = 0.1) -> Tuple[int, Any]:

    last_exception = None

    for attempt in range(1, max_attempts + 1):
        try:
            status_code, response = request_fn()

            if status_code == 429 or (500 <= status_code < 600):
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                delay += delay * jitter * random.uniform(0, 1)
                time.sleep(delay)
                continue

            return status_code, response

        except Exception as exc:
            last_exception = exc
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            delay += delay * jitter * random.uniform(0, 1)
            time.sleep(delay)

    raise RuntimeError(f"Failed after {max_attempts} attempts") from last_exception


def _transcribe_with_gemini(audio_path: Optional[str], model: Optional[str] = None):
    if not audio_path:
        raise GroqASRError("audio_path required for Gemini ASR")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GroqASRError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)
    model_name = model or "models/gemini-1.5-flash"

    uploaded = genai.upload_file(audio_path)
    gm = genai.GenerativeModel(model_name)
    resp = gm.generate_content(["Transcribe this audio.", uploaded])

    return {"text": resp.text, "segments": None, "raw": resp}


@groq_rate_limiter
def transcribe_audio(audio_path: Optional[str] = None,
                     audio_bytes: Optional[bytes] = None,
                     model: Optional[str] = None,
                     language: Optional[str] = None,
                     timestamps: bool = True,
                     timeout_seconds: int = DEFAULT_TIMEOUT,
                     backend: str = "groq",
                     rate_limit_per_minute: Optional[int] = None) -> Dict[str, Any]:

    if backend.lower() == "gemini":
        return _transcribe_with_gemini(audio_path, model)

    if audio_path is None and audio_bytes is None:
        raise ValueError("Provide audio_path or audio_bytes")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise GroqASRError("GROQ_API_KEY not set")

    model_to_use = model or DEFAULT_MODEL

    params = {"model": model_to_use}
    if language:
        params["language"] = language
    if timestamps:
        params["timestamps"] = "true"

    with _asr_semaphore:

        # --- SDK path ---
        if groq is not None:
            try:
                client = groq.Groq(api_key=api_key)

                def _call_sdk():
                    if audio_path:
                        with open(audio_path, "rb") as f:
                            resp = client.audio.transcriptions.create(
                                file=f,
                                model=model_to_use,
                                language=language
                            )
                            return 200, resp
                    raise RuntimeError("Bytes path not supported in SDK mock")

                status_code, body = _request_with_backoff(_call_sdk)
                return _normalize_asr_result(body)

            except Exception as e:
                logger.warning("Groq SDK failed, falling back to httpx. %s", e)

        # --- httpx fallback ---
        if httpx is None:
            raise GroqASRError("httpx not installed; required for fallback")

        def _call_httpx():
            headers = {"Authorization": f"Bearer {api_key}"}
            if audio_path:
                with open(audio_path, "rb") as f:
                    files = {"file": (os.path.basename(audio_path), f, "application/octet-stream")}
                    data = params
                    with httpx.Client(timeout=timeout_seconds) as client:
                        resp = client.post(GROQ_TRANSCRIPT_ENDPOINT, headers=headers, files=files, data=data)
                        resp.raise_for_status()
                        return resp.status_code, resp.json()

        status_code, body = _request_with_backoff(_call_httpx)
        return _normalize_asr_result(body)


def transcribe_audio_safe(*args, **kwargs) -> Dict[str, Any]:
    try:
        return transcribe_audio(*args, **kwargs)
    except Exception as e:
        raise GroqASRError(f"ASR failed: {e}") from e


def _normalize_asr_result(result: Any) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {"text": str(result), "segments": None, "raw": result}

    parsed = _parse_groq_response(result)
    parsed["text"] = parsed.get("text") or ""
    parsed["segments"] = parsed.get("segments")
    parsed["raw"] = parsed.get("raw") or result
    return parsed
