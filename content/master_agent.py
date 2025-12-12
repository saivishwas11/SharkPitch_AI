import logging
from graph.state_types import PitchState
from utils.llm_utils import invoke_llm, parse_json_safely
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Dict, Any

logger = logging.getLogger("master_agent")
logger.setLevel(logging.INFO)

SYSTEM_PROMPT = """
You are an expert startup investor and pitch analyst.
Given a pitch transcript, do the following and return JSON only:

1) Identify Hook, Problem, Solution, Ask (if present) and return text snippets for each (or null).
2) Rate on a 0-10 scale (integers) these criteria: problem_clarity, differentiation, business_model, market_opportunity, revenue_logic, competition_awareness.
3) Return a short master_summary (2-4 sentences).
4) List 3 strengths and 3 weaknesses.
5) Compute content_score (0-100) as average(0-10 ratings) * 10.

Return JSON in this structure:
{
  "content_score": float,
  "metrics": {
    "problem_clarity": int,
    "differentiation": int,
    "business_model": int,
    "market_opportunity": int,
    "revenue_logic": int,
    "competition_awareness": int
  },
  "structure": {
    "hook": null or {"text": "..."},
    "problem": null or {"text": "..."},
    "solution": null or {"text": "..."},
    "ask": null or {"text": "..."},
    "structure_ok": bool
  },
  "master_summary": "string",
  "strengths": ["...", "...", "..."],
  "weaknesses": ["...", "...", "..."]
}
"""

def run_master_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    transcript = state.get("transcript", "")
    if not transcript:
        logger.warning("MasterAgent: empty transcript")
        state["content_analysis"] = {
            "content_score": 0.0,
            "metrics": {},
            "structure": {},
            "master_summary": "",
            "strengths": [],
            "weaknesses": [],
        }
        return state

    sys_msg = SystemMessage(content=SYSTEM_PROMPT)
    human_msg = HumanMessage(content=transcript)
    resp = invoke_llm([sys_msg, human_msg], temperature=0.0)

    # parse JSON
    try:
        parsed = parse_json_safely(resp.content)
    except Exception as e:
        logger.exception("MasterAgent parse error: %s", e)
        state["content_analysis"] = {
            "content_score": 0.0,
            "metrics": {},
            "structure": {},
            "master_summary": getattr(resp, "content", str(resp)),
            "strengths": [],
            "weaknesses": [],
        }
        return state

    # Ensure content_score exists
    content_score = parsed.get("content_score")
    if content_score is None:
        metrics = parsed.get("metrics", {})
        if metrics:
            vals = [int(metrics.get(k, 0)) for k in metrics]
            content_score = (sum(vals) / len(vals)) * 10.0
        else:
            content_score = 0.0

    state["content_analysis"] = {
        "content_score": float(content_score),
        "metrics": parsed.get("metrics", {}),
        "structure": parsed.get("structure", {}),
        "master_summary": parsed.get("master_summary", ""),
        "strengths": parsed.get("strengths", []),
        "weaknesses": parsed.get("weaknesses", []),
    }
    return state

# LangGraph node
def master_agent_node(state: PitchState) -> PitchState:
    return run_master_agent(state)
