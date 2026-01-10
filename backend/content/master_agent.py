import logging
from typing import Dict, Any, Tuple

from backend.graph.state_types import PitchState
from backend.utils.llm_utils import invoke_llm, parse_json_safely
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger("master_agent")
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------
# SYSTEM PROMPT
# ------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert startup investor and pitch analyst.

Given a pitch transcript, analyze the business and return STRICT JSON with:

1) Business Viability Metrics (rate each 0–10):

- problem_clarity:
  How clearly and concretely the problem is defined and motivated.

- product_differentiation:
  How distinct the solution is versus existing alternatives.

- business_model_strength:
  Revenue logic, pricing power, margins, scalability.

- market_opportunity:
  Market size (TAM/SAM/SOM), growth, urgency.

- competition_awareness:
  Awareness of competitors, substitutes, and positioning.

- execution_risk:
  Operational, team, timeline, and go-to-market risk.
  (IMPORTANT: lower score = higher risk)

- financial_health:
  Burn rate, runway, cash discipline, sustainability.

- innovation_level:
  Novelty, defensibility, IP, technical advantage.

2) Pitch Structure (extract short verbatim snippets):
- hook
- problem
- solution
- ask
- structure_ok (boolean)

3) High-level Analysis:
- master_summary (2–4 sentences)
- key_strengths (top 3)
- key_risks (top 3)
- investment_recommendation:
  One of ["Strong Buy", "Buy", "Hold", "Pass"]

Return JSON ONLY in this exact format:

{
  "business_viability_score": float (0–100),
  "metrics": {
    "problem_clarity": int (0–10),
    "product_differentiation": int (0–10),
    "business_model_strength": int (0–10),
    "market_opportunity": int (0–10),
    "competition_awareness": int (0–10),
    "execution_risk": int (0–10),
    "financial_health": int (0–10),
    "innovation_level": int (0–10)
  },
  "structure": {
    "hook": {"text": "..."},
    "problem": {"text": "..."},
    "solution": {"text": "..."},
    "ask": {"text": "..."},
    "structure_ok": bool
  },
  "master_summary": "string",
  "key_strengths": ["...", "...", "..."],
  "key_risks": ["...", "...", "..."],
  "investment_recommendation": "string"
}
"""

# ------------------------------------------------------------------
# Viability Weights (explicit & interpretable)
# ------------------------------------------------------------------

BUSINESS_VIABILITY_WEIGHTS = {
    "problem_clarity": 0.10,
    "product_differentiation": 0.15,
    "business_model_strength": 0.15,
    "market_opportunity": 0.15,
    "competition_awareness": 0.10,
    "execution_risk": -0.15,     # lower risk = better
    "financial_health": 0.10,
    "innovation_level": 0.10,
}

INVESTMENT_THRESHOLDS = {
    "STRONG_BUY": 85,
    "BUY": 70,
    "HOLD": 50,
    "PASS": 0,
}

# ------------------------------------------------------------------
# Scoring Logic
# ------------------------------------------------------------------

def calculate_viability_score(metrics: Dict[str, int]) -> Tuple[float, str]:
    total = 0.0
    weight_sum = 0.0

    for metric, weight in BUSINESS_VIABILITY_WEIGHTS.items():
        val = metrics.get(metric, 0)
        if weight < 0:
            val = 10 - val  # invert risk
        total += val * abs(weight)
        weight_sum += abs(weight)

    score = (total / weight_sum) * 10 if weight_sum else 0.0

    if score >= INVESTMENT_THRESHOLDS["STRONG_BUY"]:
        rec = "Strong Buy"
    elif score >= INVESTMENT_THRESHOLDS["BUY"]:
        rec = "Buy"
    elif score >= INVESTMENT_THRESHOLDS["HOLD"]:
        rec = "Hold"
    else:
        rec = "Pass"

    return round(score, 1), rec


# ------------------------------------------------------------------
# Core Agent
# ------------------------------------------------------------------

def run_master_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    transcript = state.get("transcript", "")

    if not transcript:
        logger.warning("MasterAgent: empty transcript")
        return {
            "content_analysis": {
                "business_viability_score": 0.0,
                "content_score": 0.0,
                "metrics": {},
                "structure": {},
                "master_summary": "",
                "key_strengths": [],
                "key_risks": ["No transcript provided"],
                "investment_recommendation": "Insufficient Data",
            }
        }

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=transcript),
    ]

    resp = invoke_llm(messages, temperature=0.0)

    try:
        parsed = parse_json_safely(resp.content)
    except Exception as e:
        logger.exception("MasterAgent JSON parse error")
        return {
            "content_analysis": {
                "business_viability_score": 0.0,
                "content_score": 0.0,
                "metrics": {},
                "structure": {},
                "master_summary": "LLM output could not be parsed",
                "key_strengths": [],
                "key_risks": ["Parsing failure"],
                "investment_recommendation": "Error",
            }
        }

    metrics = parsed.get("metrics", {})

    viability_score, recommendation = calculate_viability_score(metrics)

    logger.info(
        "Business Viability Score: %s/100 | Recommendation: %s",
        viability_score,
        recommendation,
    )

    return {
        "content_analysis": {
            "business_viability_score": viability_score,
            "content_score": viability_score,  # backward compatibility
            "metrics": metrics,
            "structure": parsed.get("structure", {}),
            "master_summary": parsed.get("master_summary", ""),
            "key_strengths": parsed.get("key_strengths", []),
            "key_risks": parsed.get("key_risks", []),
            "investment_recommendation": parsed.get(
                "investment_recommendation", recommendation
            ),
        }
    }


# ------------------------------------------------------------------
# LangGraph Node
# ------------------------------------------------------------------

def master_agent_node(state: PitchState) -> dict:
    return run_master_agent(state)
