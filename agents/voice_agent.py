# prosody + delivery score
from graph.state_types import PitchState
from utils.voice_analysis import analyze_voice


def voice_agent_node(state: PitchState) -> PitchState:
    audio_path = state["clean_audio_path"]
    vstats = analyze_voice(audio_path)
    state["voice_stats"] = vstats
    return state
