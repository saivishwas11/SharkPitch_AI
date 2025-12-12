from typing import Dict, Any
from graph.state_types import PitchState


def _aggregate_shark_panel(shark_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    verdict_counts: Dict[str, int] = {}
    for _, data in shark_outputs.items():
        v = data.get("verdict", "Need More Info")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    final_verdict = max(verdict_counts.items(), key=lambda x: x[1])[0]

    return {
        "sharks": shark_outputs,
        "final_verdict": final_verdict,
        "verdict_counts": verdict_counts,
    }


def shark_aggregator_node(state: PitchState) -> PitchState:
    sharks_out = {
        "visionary": state.get("shark_visionary", {}),
        "finance": state.get("shark_finance", {}),
        "customer": state.get("shark_customer", {}),
        "skeptic": state.get("shark_skeptic", {}),
    }
    state["shark_panel"] = _aggregate_shark_panel(sharks_out)
    return state
