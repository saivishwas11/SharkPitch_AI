# utils/groq_asr.py
import os
import time
import random
import threading
import logging
from typing import Optional, Dict, Any, Callable, Tuple

import google.generativeai as genai

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Optional dependencies
try:
    import httpx  # used if groq SDK is not available
except Exception:
    httpx = None

try:
    import groq  # optional SDK; code uses it if present
except Exception:
    groq = None

# Endpoint for OpenAI-compatible audio transcription on Groq
GROQ_TRANSCRIPT_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"

# Default model, may be overridden per-call
DEFAULT_MODEL = os.environ.get("GROQ_ASR_MODEL", "whisper-large-v3")

# Concurrency: simple semaphore to avoid flooding the API (per-process)
_ASR_CONCURRENCY = int(os.environ.get("GROQ_ASR_CONCURRENCY", "1"))
_asr_semaphore = threading.Semaphore(max(1, _ASR_CONCURRENCY))


class GroqASRError(RuntimeError):
    """Raised when ASR call fails in an expected way."""


def _parse_groq_response(resp_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize typical Groq/Whisper-like response shapes into:
      {"text": "...", "segments": [...], "raw": resp_json}
    """
    text = ""
    segments = None

    if not isinstance(resp_json, dict):
        # Unexpected shape; return stringified as text
        return {"text": str(resp_json), "segments": None, "raw": resp_json}

    # Common shapes:
    # 1) {"text": "...", "segments": [...]}
    # 2) {"choices": [{"message": {"content": "..."}}], ...}
    if "text" in resp_json and isinstance(resp_json["text"], str):
        text = resp_json["text"]
        segments = resp_json.get("segments")
    elif "choices" in resp_json and isinstance(resp_json["choices"], (list, tuple)) and resp_json["choices"]:
        first = resp_json["choices"][0]
        if isinstance(first, dict):
            # chat-style message
            msg = first.get("message") or first.get("text") or first
            if isinstance(msg, dict):
                text = msg.get("content") or ""
            else:
                text = str(msg or "")
        else:
            text = str(first)
        # segments may be present at top level
        segments = resp_json.get("segments")
    else:
        # fallback: try to read 'data' -> 'text' or other fields
        text = resp_json.get("text") or resp_json.get("transcript") or ""
        segments = resp_json.get("segments") or resp_json.get("timestamps")

    return {"text": text or "", "segments": segments, "raw": resp_json}


def _request_with_backoff(request_fn: Callable[[], Tuple[int, Any]], max_attempts: int = 6, base_delay: float = 1.0):
    """
    Retry wrapper for requests that may raise or return HTTP error.
    `request_fn` should perform the request and return (status_code, body_json_or_text).
    If the underlying client raises an httpx.HTTPStatusError or similar, the wrapper will catch
    and retry on 429 with exponential backoff + jitter.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return request_fn()
        except Exception as exc:
            # Detect httpx.HTTPStatusError or 429-like from other clients
            status = None
            retry_after = None
            # httpx
            try:
                if hasattr(exc, "response") and getattr(exc.response, "status_code", None) is not None:
                    status = exc.response.status_code
                    retry_after = exc.response.headers.get("Retry-After")
            except Exception:
                pass

            # If 429, backoff and retry (up to max_attempts)
            if status == 429 and attempt < max_attempts:
                try:
                    retry_after = float(retry_after) if retry_after else None
                except Exception:
                    retry_after = None
                delay = retry_after or min(base_delay * (2 ** (attempt - 1)), 30.0)
                jitter = random.uniform(0, 0.2 * delay)
                wait = delay + jitter
                logger.warning("Rate limited (429). Backing off for %.1f seconds (attempt %d/%d).", wait, attempt, max_attempts)
                time.sleep(wait)
                continue

            # For transport-level errors, retry a few times
            if attempt < max_attempts:
                delay = min(base_delay * (2 ** (attempt - 1)), 30.0)
                jitter = random.uniform(0, 0.2 * delay)
                wait = delay + jitter
                logger.warning("Transient error during ASR request: %s. Retrying in %.1f seconds (attempt %d/%d).", exc, wait, attempt, max_attempts)
                time.sleep(wait)
                continue

            # Exhausted retries or non-retryable error
            raise


def _transcribe_with_gemini(
    audio_path: Optional[str],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Transcribe using Gemini. Keeps it minimal: returns {"text": str, "segments": None, "raw": resp}
    """
    if not audio_path:
        raise GroqASRError("audio_path is required for Gemini ASR.")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GroqASRError("GEMINI_API_KEY environment variable is not set.")

    genai.configure(api_key=api_key)
    model_name = model or os.environ.get("GEMINI_ASR_MODEL", "models/gemini-1.5-flash")

    uploaded = genai.upload_file(audio_path)
    gm = genai.GenerativeModel(model_name)
    # Ask for plain text transcript to reduce verbosity
    resp = gm.generate_content([uploaded, "Transcribe this audio. Return only plain text."])
    text = getattr(resp, "text", "") or ""
    return {"text": text.strip(), "segments": None, "raw": resp}


def transcribe_audio(
    audio_path: Optional[str] = None,
    audio_bytes: Optional[bytes] = None,
    model: Optional[str] = None,
    language: Optional[str] = None,
    timestamps: bool = True,
    timeout_seconds: int = 120,
    backend: str = "groq",
) -> Dict[str, Any]:
    """
    Transcribe audio via Groq's OpenAI-compatible transcription API.

    Provide either audio_path or audio_bytes.

    Returns a normalized dict: {"text": str, "segments": Optional[list], "raw": dict}

    Raises GroqASRError on failure.
    """
    if backend.lower() == "gemini":
        return _transcribe_with_gemini(audio_path, model=model)
    if audio_path is None and audio_bytes is None:
        raise ValueError("Either audio_path or audio_bytes must be provided.")

    model_to_use = model or DEFAULT_MODEL
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise GroqASRError("GROQ_API_KEY environment variable is not set.")

    params = {"model": model_to_use}
    if language:
        params["language"] = language
    if timestamps:
        # Some endpoints expect timestamps True/false; include as hint
        params["timestamps"] = "true"

    with _asr_semaphore:
        # Try groq SDK path first if available
        if groq is not None:
            try:
                # Many groq SDKs expose a simple client; adapt if your SDK differs.
                # We'll try to call the OpenAI-compatible endpoint via the SDK if possible.
                client = None
                if hasattr(groq, "Groq"):  # some wrappers expose a Groq class
                    client = groq.Groq(api_key=api_key)
                elif hasattr(groq, "client"):
                    client = groq.client
                else:
                    client = groq

                def _call_groq():
                    # This block adapts to likely SDK shapes; if your SDK is different, adapt here.
                    # We'll fallback to httpx below if SDK path fails.
                    if audio_path:
                        with open(audio_path, "rb") as f:
                            # Attempt a generic create/post via SDK if it supports it
                            if hasattr(client, "create"):
                                resp = client.create(url="/openai/v1/audio/transcriptions", files={"file": f}, data=params)
                                # If resp is an object, try to extract json
                                if hasattr(resp, "json"):
                                    return resp.status_code if hasattr(resp, "status_code") else 200, resp.json()
                                return 200, resp
                            elif hasattr(client, "audio") and hasattr(client.audio, "transcriptions"):
                                # hypothetical helper
                                return 200, client.audio.transcriptions.create(file=f, model=model_to_use, language=language)
                            else:
                                # SDK not compatible; raise to fallback to httpx
                                raise RuntimeError("groq SDK present but doesn't support create/audio.transcriptions in this wrapper.")
                    else:
                        # audio_bytes path
                        if hasattr(client, "create"):
                            from io import BytesIO
                            bf = BytesIO(audio_bytes)
                            resp = client.create(url="/openai/v1/audio/transcriptions", files={"file": bf}, data=params)
                            if hasattr(resp, "json"):
                                return resp.status_code if hasattr(resp, "status_code") else 200, resp.json()
                            return 200, resp
                        raise RuntimeError("groq SDK present but does not support byte uploads in this helper.")

                # run request with backoff
                status_code, body = _request_with_backoff(_call_groq)
                if isinstance(body, dict):
                    return _parse_groq_response(body)
                # if body is not dict, try to coerce
                try:
                    return _parse_groq_response(dict(body))
                except Exception:
                    return _parse_groq_response({"text": str(body), "raw": body})
            except Exception as e:
                logger.exception("Groq SDK path failed; falling back to httpx. Error: %s", e)
                # fall through to httpx path

        # Fallback: use httpx to call the OpenAI-compatible transcription endpoint
        if httpx is None:
            raise GroqASRError("httpx is required as a fallback but is not installed. Install httpx or provide a working groq SDK.")

        def _httpx_call():
            headers = {"Authorization": f"Bearer {api_key}"}
            # build files and data in a context so file handles close correctly
            if audio_path:
                # open file and post
                with open(audio_path, "rb") as f:
                    files = {"file": (os.path.basename(audio_path), f, "application/octet-stream")}
                    data = {"model": model_to_use}
                    if language:
                        data["language"] = language
                    if timestamps:
                        data["timestamps"] = "true"
                    with httpx.Client(timeout=timeout_seconds) as client:
                        resp = client.post(GROQ_TRANSCRIPT_ENDPOINT, headers=headers, files=files, data=data)
                        resp.raise_for_status()
                        return resp.status_code, resp.json()
            else:
                # audio bytes
                files = {"file": ("audio.wav", audio_bytes, "application/octet-stream")}
                data = {"model": model_to_use}
                if language:
                    data["language"] = language
                if timestamps:
                    data["timestamps"] = "true"
                with httpx.Client(timeout=timeout_seconds) as client:
                    resp = client.post(GROQ_TRANSCRIPT_ENDPOINT, headers=headers, files=files, data=data)
                    resp.raise_for_status()
                    return resp.status_code, resp.json()

        status_code, body = _request_with_backoff(_httpx_call)
        if isinstance(body, dict):
            return _parse_groq_response(body)
        try:
            return _parse_groq_response(dict(body))
        except Exception:
            return _parse_groq_response({"text": str(body), "raw": body})


def transcribe_audio_safe(*args, **kwargs) -> Dict[str, Any]:
    """
    Wrapper that calls transcribe_audio(...) and guarantees a normalized dict response
    or raises GroqASRError with helpful message.
    """
    try:
        result = transcribe_audio(*args, **kwargs)
    except Exception as e:
        logger.exception("transcribe_audio failed: %s", e)
        raise GroqASRError(f"ASR request failed: {e}") from e

    if result is None:
        logger.error("transcribe_audio returned None.")
        raise GroqASRError("ASR returned no result (None). Check API key and network connectivity.")

    if not isinstance(result, dict):
        # try to coerce
        try:
            result = dict(result)
        except Exception:
            logger.warning("ASR returned unexpected non-dict result; coercing to string.")
            result = {"text": str(result), "raw": result}

    normalized = _parse_groq_response(result)
    # ensure fields exist
    normalized["text"] = normalized.get("text") or ""
    normalized["segments"] = normalized.get("segments")
    normalized["raw"] = normalized.get("raw") or result
    return normalized


# small test main (only runs if executed directly) - useful for quick debugging
if __name__ == "__main__":  # pragma: no cover
    import sys
    if len(sys.argv) < 2:
        print("Usage: python groq_asr.py /path/to/audio.wav")
        sys.exit(1)
    audio = sys.argv[1]
    try:
        out = transcribe_audio_safe(audio_path=audio)
        print("TRANSCRIPT:\n", out["text"])
        if out.get("segments"):
            print("SEGMENTS:")
            for s in out["segments"]:
                print(s)
    except Exception as e:
        print("Error:", e)
        raise
