# loads & preprocesses audio
from graph.state_types import PitchState
from utils.audio_processing import load_and_prepare_audio


def audio_agent_node(state: PitchState) -> PitchState:
    path = state["input_path"]
    # unpack the tuple: (audio_numpy, sample_rate, audio_file_path)
    _, _, clean_path = load_and_prepare_audio(path)
    state["clean_audio_path"] = clean_path
    return state
