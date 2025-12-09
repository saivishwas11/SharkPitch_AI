import os
import google.generativeai as genai
from graph.state_types import PitchState
from utils.llm_utils import parse_json_safely


SYSTEM_PROMPT = """You are the Single Content & Viability Agent.
Given a full pitch transcript, perform these evaluations (0-10 each):
- problem_clarity
- product_differentiation
- business_model_strength
- market_opportunity_articulation
- revenue_logic
- competition_awareness

Detect pitch structure presence (true/false each):
- hook, problem, solution, ask

Compute a business_viability score (0-100) combining the above dimensions.

Return JSON ONLY in this format:
{
  "problem_clarity": int,
  "product_differentiation": int,
  "business_model_strength": int,
  "market_opportunity_articulation": int,
  "revenue_logic": int,
  "competition_awareness": int,
  "structure": {
    "hook": bool,
    "problem": bool,
    "solution": bool,
    "ask": bool,
    "structure_ok": bool
  },
  "business_viability": int,
  "comment": "concise justification"
}
Keep responses concise; no extra text."""


def content_master_node(state: PitchState) -> PitchState:
    transcript = state.get("transcript", "")
    if not transcript:
        state["content_analysis"] = {"error": "empty transcript"}
        return state

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        state["content_analysis"] = {"error": "GEMINI_API_KEY not set"}
        return state

    genai.configure(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content([SYSTEM_PROMPT, transcript])
    content = getattr(resp, "text", "") or ""
    state["content_analysis"] = parse_json_safely(content)

    # Drop large/constant fields to avoid concurrent writes downstream
    state.pop("input_path", None)
    state.pop("clean_audio_path", None)
    return state

