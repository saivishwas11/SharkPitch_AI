from typing import Dict, Any
import threading
import time
import re
import math

from langchain_groq import ChatGroq
import groq
from config import GROQ_MODEL

# Global lock to ensure only one request happens at a time
_API_LOCK = threading.Lock()


class RateLimitedChatGroq(ChatGroq):
    """
    A wrapper around ChatGroq that:
    1. Enforces strict serialization (one request at a time) using a global lock.
    2. Catches 429 RateLimitErrors.
    3. Parses the 'try again in X seconds' part of the error message.
    4. Waits exactly that amount + small buffer.
    5. Retries.
    """
    def invoke(self, input, config=None, **kwargs):
        # Acquire lock to ensure no other agent is using the LLM
        with _API_LOCK:
            while True:
                try:
                    return super().invoke(input, config=config, **kwargs)
                except Exception as e:
                    # Check if it's a Groq rate limit error
                    # It often comes as a groq.RateLimitError or has status_code=429
                    is_rate_limit = False
                    error_msg = str(e)
                    
                    if isinstance(e, groq.RateLimitError):
                        is_rate_limit = True
                    elif "429" in error_msg or "rate limit" in error_msg.lower():
                        is_rate_limit = True
                    
                    if not is_rate_limit:
                        raise e
                    
                    # Extract wait time
                    # Look for "Please try again in 23.13s" or similar
                    match = re.search(r"try again in (\d+(\.\d+)?)s", error_msg)
                    wait_time = 5.0 # default fallback
                    if match:
                        wait_time = float(match.group(1))
                    
                    # Add a small buffer just in case
                    wait_time = math.ceil(wait_time) + 1.0
                    
                    print(f"\n[RateLimitedChatGroq] Hit rate limit. Waiting {wait_time}s before retrying...\n")
                    time.sleep(wait_time)
                    # Retry loop continues


def get_llm(temperature: float = 0.0) -> ChatGroq:
    return RateLimitedChatGroq(model=GROQ_MODEL, temperature=temperature, max_retries=0)


def parse_json_safely(text: str) -> Dict[str, Any]:
    import json

    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start : end + 1])


SHARK_SYSTEM_TEMPLATE = """You are {name}, a Shark Tank-style investor.
Your style: {style}.

You receive:
- delivery_score: how strong the vocal delivery is (0-100)
- content_score: how strong the business content is (0-100)
- a breakdown from several analysis agents
- the full pitch transcript

Provide:
1. A short paragraph of feedback in FIRST PERSON ("I") as this shark.
2. 2 bullet-point improvement tips.
3. A final verdict: "Invest", "Not Invest", or "Need More Info".

Return JSON ONLY in this format:
{{
  "name": "...",
  "feedback": "...",
  "tips": ["...", "..."],
  "verdict": "Invest | Not Invest | Need More Info"
}}
No extra text.
"""


def _summarize_content_agents(content_analysis: Dict[str, Any]) -> str:
    agents = content_analysis.get("agents", {})
    return (
        "Content analysis agents output:\n"
        f"Problem Agent: {agents.get('problem')}\n"
        f"Market Agent: {agents.get('market')}\n"
        f"Finance Agent: {agents.get('finance')}\n"
        f"Competition Agent: {agents.get('competition')}\n"
        f"Structure Agent: {agents.get('structure')}\n"
        f"Overall content_score: {content_analysis.get('content_score')}"
    )


def run_shark_persona(
    name: str,
    style: str,
    transcript: str,
    delivery_score: float,
    content_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    llm = get_llm(temperature=0.7)
    sys_prompt = SHARK_SYSTEM_TEMPLATE.format(name=name, style=style)
    content_score = content_analysis.get("content_score")
    content_summary = _summarize_content_agents(content_analysis)

    from langchain_core.messages import SystemMessage, HumanMessage

    resp = llm.invoke(
        [
            SystemMessage(content=sys_prompt),
            HumanMessage(
                content=(
                    f"Delivery score: {delivery_score}\n"
                    f"Content score: {content_score}\n"
                    f"Detailed analysis:\n{content_summary}\n\n"
                    f"Full transcript:\n{transcript}"
                )
            ),
        ]
    )
    return parse_json_safely(resp.content)
