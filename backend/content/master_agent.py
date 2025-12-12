import logging
from graph.state_types import PitchState
from utils.llm_utils import invoke_llm, parse_json_safely
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Dict, Any, Tuple

logger = logging.getLogger("master_agent")
logger.setLevel(logging.INFO)

SYSTEM_PROMPT = """
You are an expert startup investor and pitch analyst.
Given a pitch transcript, analyze the business viability and return JSON with:

1) Business Viability Metrics (rate 0-10):
   - financial_strength: Revenue model, margins, unit economics
   - market_potential: TAM/SAM/SOM, growth rate, market trends
   - competitive_edge: Differentiation, IP, moat, barriers to entry
   - execution_risk: Team experience, operational plan, milestones
   - financial_health: Cash flow, burn rate, path to profitability
   - innovation_level: Novelty, technological advantage, IP protection

2) Pitch Structure (extract text snippets):
   - hook: Attention-grabbing opening
   - problem: Pain point being solved
   - solution: Product/service offered
   - ask: Funding request and terms
   - structure_ok: Boolean indicating if key elements are present

3) Analysis:
   - master_summary: 2-4 sentence overview
   - key_strengths: Top 3 business advantages
   - key_risks: Top 3 business concerns
   - investment_recommendation: "Strong Buy", "Buy", "Hold", or "Pass"

Return JSON in this structure:
{
  "business_viability_score": float (0-100),
  "metrics": {
    "financial_strength": int (0-10),
    "market_potential": int (0-10),
    "competitive_edge": int (0-10),
    "execution_risk": int (0-10, lower is better),
    "financial_health": int (0-10),
    "innovation_level": int (0-10)
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

# Weights for business viability score calculation
BUSINESS_VIABILITY_WEIGHTS = {
    'financial_strength': 0.25,     # Revenue model, margins, unit economics
    'market_potential': 0.20,       # Market size and growth potential
    'competitive_edge': 0.20,       # Differentiation and moat
    'execution_risk': -0.15,        # Negative weight - lower risk is better
    'financial_health': 0.15,       # Cash flow and sustainability
    'innovation_level': 0.10        # Novelty and IP
}

# Thresholds for investment recommendations
INVESTMENT_THRESHOLDS = {
    'STRONG_BUY': 85,
    'BUY': 70,
    'HOLD': 50,
    'PASS': 0
}

def calculate_viability_score(metrics: Dict[str, int]) -> Tuple[float, str]:
    """Calculate weighted business viability score and investment recommendation."""
    total_score = 0.0
    total_weight = 0.0
    
    for metric, weight in BUSINESS_VIABILITY_WEIGHTS.items():
        score = metrics.get(metric, 0)
        if weight < 0:  # Invert score for risk metrics (lower is better)
            score = 10 - score
        total_score += score * abs(weight)
        total_weight += abs(weight)
    
    # Normalize to 0-100 scale
    if total_weight > 0:
        viability_score = (total_score / total_weight) * 10
    else:
        viability_score = 0.0
    
    # Determine investment recommendation
    if viability_score >= INVESTMENT_THRESHOLDS['STRONG_BUY']:
        recommendation = "Strong Buy"
    elif viability_score >= INVESTMENT_THRESHOLDS['BUY']:
        recommendation = "Buy"
    elif viability_score >= INVESTMENT_THRESHOLDS['HOLD']:
        recommendation = "Hold"
    else:
        recommendation = "Pass"
    
    return round(viability_score, 1), recommendation

def run_master_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    transcript = state.get("transcript", "")
    if not transcript:
        logger.warning("MasterAgent: empty transcript")
        state["content_analysis"] = {
            "business_viability_score": 0.0,
            "content_score": 0.0,  # For backward compatibility
            "metrics": {},
            "structure": {},
            "master_summary": "",
            "key_strengths": [],
            "key_risks": [],
            "investment_recommendation": "Insufficient Data"
        }
        return state

    sys_msg = SystemMessage(content=SYSTEM_PROMPT)
    human_msg = HumanMessage(content=transcript)
    resp = invoke_llm([sys_msg, human_msg], temperature=0.0)

    # Parse JSON response
    try:
        parsed = parse_json_safely(resp.content)
    except Exception as e:
        logger.exception("MasterAgent parse error: %s", e)
        state["content_analysis"] = {
            "business_viability_score": 0.0,
            "content_score": 0.0,
            "metrics": {},
            "structure": {},
            "master_summary": f"Error parsing analysis: {str(e)[:200]}",
            "key_strengths": [],
            "key_risks": ["Failed to analyze transcript"],
            "investment_recommendation": "Error"
        }
        return state

    # Get metrics with defaults
    metrics = {
        'financial_strength': parsed.get('metrics', {}).get('financial_strength', 0),
        'market_potential': parsed.get('metrics', {}).get('market_potential', 0),
        'competitive_edge': parsed.get('metrics', {}).get('competitive_edge', 0),
        'execution_risk': parsed.get('metrics', {}).get('execution_risk', 5),  # Default to neutral
        'financial_health': parsed.get('metrics', {}).get('financial_health', 0),
        'innovation_level': parsed.get('metrics', {}).get('innovation_level', 0)
    }

    # Calculate business viability score and recommendation
    business_viability_score, recommendation = calculate_viability_score(metrics)
    
    # For backward compatibility, maintain content_score
    content_score = parsed.get('business_viability_score', business_viability_score)
    
    # Prepare response
    state["content_analysis"] = {
        "business_viability_score": business_viability_score,
        "content_score": float(content_score),  # Backward compatible
        "metrics": metrics,
        "structure": parsed.get("structure", {}),
        "master_summary": parsed.get("master_summary", ""),
        "key_strengths": parsed.get("key_strengths", parsed.get("strengths", [])),
        "key_risks": parsed.get("key_risks", parsed.get("weaknesses", [])),
        "investment_recommendation": parsed.get("investment_recommendation", recommendation),
        "analysis_timestamp": parsed.get("analysis_timestamp", "")
    }
    
    logger.info(f"Business Viability Score: {business_viability_score}/100 - {recommendation}")
    return state

# LangGraph node
def master_agent_node(state: PitchState) -> PitchState:
    return run_master_agent(state)
