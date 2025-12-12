from typing import TypedDict, Dict, Any, List


class PitchState(TypedDict, total=False):
    # Input & audio
    input_path: str
    clean_audio_path: str

    # Voice
    voice_stats: Dict[str, Any]

    # Transcript
    transcript: str
    transcript_segments: List[Dict[str, Any]]

    # Content agents
    content_problem: Dict[str, Any]
    content_market: Dict[str, Any]
    content_finance: Dict[str, Any]
    content_competition: Dict[str, Any]
    content_structure: Dict[str, Any]
    content_analysis: Dict[str, Any]  # aggregated content score + agents

    # Shark agents
    shark_visionary: Dict[str, Any]
    shark_finance: Dict[str, Any]
    shark_customer: Dict[str, Any]
    shark_skeptic: Dict[str, Any]
    shark_panel: Dict[str, Any]
