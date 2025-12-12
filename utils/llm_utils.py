# utils/llm_utils.py
from pathlib import Path
import os
import threading
import time
import re
import math
import logging
import json
from typing import Dict, Any, List

# load .env if present (safe)
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv() or str(Path(__file__).resolve().parents[1] / ".env"))
except Exception:
    pass

logger = logging.getLogger("llm_utils")
logger.setLevel(logging.INFO)

# Try to import real Groq/langchain bindings; if missing, we'll fall back to DummyLLM
_HAS_GROQ = False
try:
    from langchain_groq import ChatGroq  # type: ignore
    import groq  # type: ignore
    _HAS_GROQ = True
except Exception:
    ChatGroq = None
    groq = None
    _HAS_GROQ = False
    logger.debug("langchain_groq/groq not available; falling back to DummyLLM if no key.")

# Model id from config or env
try:
    from config import GROQ_MODEL  # type: ignore
except Exception:
    GROQ_MODEL = os.getenv("GROQ_MODEL", "gpt-4o-mini")

_API_LOCK = threading.Lock()
_GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # may be None

# If real client available AND key present -> use a rate-limited wrapper
if _HAS_GROQ and _GROQ_API_KEY:

    class RateLimitedChatGroq(ChatGroq):  # type: ignore
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("api_key", _GROQ_API_KEY)
            super().__init__(*args, **kwargs)

        def invoke(self, messages: List[Any], config=None, **kwargs):
            # serialize calls and retry on rate-limit
            with _API_LOCK:
                while True:
                    try:
                        return super().invoke(messages, config=config, **kwargs)
                    except Exception as e:
                        msg = str(e)
                        # detect rate limit heuristically
                        if "429" not in msg and "rate" not in msg.lower() and "too many" not in msg.lower():
                            logger.exception("LLM invocation failed (non-rate-limit).")
                            raise
                        wait = 5
                        m = re.search(r"try again in (\d+(\.\d+)?)s", msg)
                        if m:
                            try:
                                wait = float(m.group(1))
                            except Exception:
                                pass
                        wait = math.ceil(wait) + 1
                        logger.warning("Rate limit hit. Waiting %ss before retry...", wait)
                        time.sleep(wait)

    def get_llm(temperature: float = 0.0):
        return RateLimitedChatGroq(model=GROQ_MODEL, temperature=temperature, api_key=_GROQ_API_KEY)

else:
    # Dummy LLM implementation for local development / testing
    class DummyResponse:
        def __init__(self, content: str):
            self.content = content

    class DummyLLM:
        def __init__(self, model: str = None, temperature: float = 0.0, api_key: str = None):
            self.model = model
            self.temperature = temperature
            self.api_key = api_key
            logger.info("Initialized DummyLLM (no GROQ API key or libs). Responses are placeholders.")

        def invoke(self, messages: List[Any], config=None, **kwargs):
            # build a short echo to include some context for downstream parsers
            try:
                # langchain_core.messages objects often have .content
                pieces = []
                for m in messages:
                    pieces.append(getattr(m, "content", str(m)))
                joined = "\n\n".join(pieces)[:1500]
            except Exception:
                joined = " ".join([str(m) for m in messages])[:1500]
            # return JSON string so parse_json_safely can find an object
            payload = {
                "note": "dummy response (no GROQ_API_KEY or langchain_groq)",
                "echo": joined,
            }
            return DummyResponse(json.dumps(payload))

    def get_llm(temperature: float = 0.0):
        return DummyLLM(model=GROQ_MODEL, temperature=temperature, api_key=_GROQ_API_KEY)

# --- Utility functions that other modules import ---


def parse_json_safely(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object in `text` and parse it.
    Raises ValueError if no JSON found or parsing fails.
    """
    if not text:
        raise ValueError("Empty model response")
    s = text.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model output")
    json_text = s[start : end + 1]
    return json.loads(json_text)


def invoke_llm(messages: List[Any], temperature: float = 0.0):
    """
    Convenience wrapper used by content/master_agent.py and elsewhere.
    Returns the raw response object (expected to have .content).
    """
    llm = get_llm(temperature=temperature)
    resp = llm.invoke(messages)
    return resp


# Backwards-compatible function used by sharks in some versions
def run_shark_persona(name: str, style: str, transcript: str, delivery_score: float, content_analysis: Dict[str, Any]):
    """
    Simple adapter to call the LLM for a shark persona using Gemini API.
    Returns parsed JSON if possible, otherwise returns a fallback dict.
    """
    try:
        # Configure Gemini
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set, falling back to dummy response")
            return get_dummy_response(name)
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Prepare content summary
        content_summary = ""
        if content_analysis:
            cs = content_analysis.get("content_score", 0)
            content_summary = f"Content score: {cs}\n"
            if "master_summary" in content_analysis:
                content_summary += f"Summary: {content_analysis.get('master_summary')}\n"
        
        # Create the prompt
        system_prompt = f"""You are {name}, a {style} shark on Shark Tank. 
        Analyze this pitch and provide feedback. Be concise, direct, and in character.
        Focus on business viability, market potential, and investment potential.
        
        Format your response as a JSON object with these fields:
        - feedback: Your detailed feedback
        - verdict: "Deal", "No Deal", or "Need More Info"
        - tips: List of specific suggestions (2-3 items)"""
        
        human_text = f"""Pitch Transcript:
        {transcript[:8000]}  # Truncate to stay within context window
        
        Delivery Score: {delivery_score}/100
        Content Analysis: {content_summary}"""
        
        # Generate response
        response = model.generate_content([
            {"role": "user", "parts": [system_prompt]},
            {"role": "model", "parts": ["Understood. I'll analyze the pitch as requested."]},
            {"role": "user", "parts": [human_text]}
        ])
        
        # Parse response
        try:
            result = json.loads(response.text)
            return {
                "name": name,
                "feedback": result.get("feedback", "No feedback provided"),
                "verdict": result.get("verdict", "Need More Info"),
                "tips": result.get("tips", [])
            }
        except json.JSONDecodeError:
            logger.warning("Failed to parse Gemini response as JSON")
            return {
                "name": name,
                "feedback": response.text[:1000],
                "verdict": "Need More Info",
                "tips": ["Could not parse response - please review manually"]
            }
            
    except Exception as e:
        logger.error(f"Error in Gemini shark analysis: {str(e)}")
        return get_dummy_response(name)

def get_dummy_response(name: str) -> Dict[str, Any]:
    """Fallback response if analysis fails"""
    return {
        "name": name,
        "feedback": f"{name} is currently unavailable. Please try again later.",
        "verdict": "Need More Info",
        "tips": ["Service temporarily unavailable"]
    }
