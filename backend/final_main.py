"""
final_main.py

Unified execution pipeline:
- Audio / Video input
- Voice analysis + ratings
- Full ASR transcript
- Business NLP (master agent)
- Shark persona feedback (optional, included)

Run:
    python final_main.py <audio_or_video_file>
"""

from __future__ import annotations

import os
import sys
import json
import logging
import warnings
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, cast

# ------------------------------------------------------------------
# Ensure project root is on sys.path so `backend` package is importable
# This allows running the script directly from the `backend` directory:
#   python final_main.py <audio_or_video_file>
# ------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ------------------------------------------------------------------
# Silence noisy warnings (environment-related, not logic issues)
# ------------------------------------------------------------------
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("final_main")

# ------------------------------------------------------------------
# Imports from your project
# ------------------------------------------------------------------
from backend.utils.voice_analysis import analyze_voice
from backend.content.master_agent import run_master_agent
from backend.graph.state_types import PitchState

from backend.sharks.customer_shark import customer_shark_node
from backend.sharks.finance_shark import finance_shark_node
from backend.sharks.skeptic_shark import skeptic_shark_node
from backend.sharks.visionary_shark import visionary_shark_node


# ------------------------------------------------------------------
# Shark sequence (aggregator intentionally skipped)
# ------------------------------------------------------------------
SHARK_SEQUENCE: List[Tuple[str, Any]] = [
    ("visionary", visionary_shark_node),
    ("finance", finance_shark_node),
    ("customer", customer_shark_node),
    ("skeptic", skeptic_shark_node),
]


# ------------------------------------------------------------------
# Rating helper
# ------------------------------------------------------------------
def rate(value: float, low: float, high: float, reverse: bool = False) -> str:
    if reverse:
        if value < low:
            return "Best"
        elif value < high:
            return "Good"
        return "Average"
    else:
        if value < low:
            return "Average"
        elif value < high:
            return "Good"
        return "Best"


# ------------------------------------------------------------------
# Audio extraction
# ------------------------------------------------------------------
def extract_audio_with_ffmpeg(video_path: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        tmp.name,
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return tmp.name


# ------------------------------------------------------------------
# Shark execution
# ------------------------------------------------------------------
def run_shark_personas(state: PitchState) -> Dict[str, Any]:
    for _, node in SHARK_SEQUENCE:
        state = node(state)
    return cast(Dict[str, Any], state.get("shark_panel", {}))


# ------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python final_main.py <audio_or_video_file>")
        sys.exit(1)

    input_path = sys.argv[1]
    audio_path = input_path

    if input_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        logger.info("Extracting audio from video...")
        audio_path = extract_audio_with_ffmpeg(input_path)

    # ---------------- Voice Analysis ----------------
    logger.info("Running voice analysis...")
    voice_results = analyze_voice(audio_path)

    vf = voice_results["vocal_features"]
    pitch = vf["pitch"]
    pitch_ml = vf["pitch_ml"]
    pauses = vf["pauses"]
    emotion = voice_results["emotion"]
    transcript = voice_results.get("transcript", "")

    # ---------------- Print Voice Results ----------------
    print("\n" + "=" * 80)
    print("VOICE ANALYSIS")
    print("=" * 80)

    print(f"Audio duration: {vf['duration_sec']:.2f} seconds")

    print("\nProsody:")
    print(f"  Pitch mean: {pitch['mean']:.2f} Hz ({rate(pitch['mean'], 120, 200)})")
    print(f"  Pitch variation: {pitch['variance']:.2f} ({rate(pitch['variance'], 2000, 8000)})")
    print(f"  Speech tempo: {vf['pace_bpm']:.2f} BPM ({rate(vf['pace_bpm'], 90, 150)})")

    print("\nPauses:")
    print(f"  Pause count: {pauses['count']}")
    print(f"  Pause density: {pauses['density']:.2f} ({rate(pauses['density'], 0.20, 0.35, reverse=True)})")

    print("\nPitch Behavior (ML):")
    print(f"  Confidence from pitch: {pitch_ml['confidence_from_pitch']:.2f} ({rate(pitch_ml['confidence_from_pitch'], 0.4, 0.7)})")
    print(f"  Expressiveness: {pitch_ml['expressiveness']:.2f} ({rate(pitch_ml['expressiveness'], 0.3, 0.6)})")
    print(f"  Monotonicity: {pitch_ml['monotonicity']:.2f} ({rate(pitch_ml['monotonicity'], 0.4, 0.7, reverse=True)})")

    print("\nEmotional Tone:")
    for k, v in emotion.items():
        print(f"  {k.capitalize()}: {v:.2f} ({rate(v, 0.4, 0.7)})")

    print(f"\nDelivery Score: {voice_results['delivery_score']}/100")

    # ---------------- Transcript ----------------
    print("\n" + "=" * 80)
    print("FULL TRANSCRIPT (excerpt)")
    print("=" * 80)
    print(transcript[:800] + ("..." if len(transcript) > 800 else ""))

    # ---------------- Master Agent ----------------
    logger.info("Running master agent analysis...")
    state: PitchState = cast(PitchState, {"transcript": transcript})
    state = run_master_agent(state)
    analysis = state["content_analysis"]

    print("\n" + "=" * 80)
    print("BUSINESS ANALYSIS")
    print("=" * 80)

    print(f"Business Viability Score: {analysis['business_viability_score']}/100")
    print(f"Investment Recommendation: {analysis['investment_recommendation']}")

    print("\nMaster Summary:")
    print(" ", analysis["master_summary"])

    print("\nMetrics (0–10):")
    for k, v in analysis["metrics"].items():
        print(f"  {k.replace('_', ' ').title()}: {v}")

    print("\nPitch Structure:")
    for k, v in analysis["structure"].items():
        if isinstance(v, dict):
            print(f"  {k.title()}: {v.get('text', '')[:120]}")

    # ---------------- Sharks ----------------
    logger.info("Running shark personas...")
    panel = run_shark_personas(state)

    print("\n" + "=" * 80)
    print("SHARK FEEDBACK")
    print("=" * 80)

    for shark, payload in panel.items():
        print(f"\n[{payload.get('name', shark.title())}]")
        print("Verdict:", payload.get("verdict"))
        print("Feedback:", payload.get("feedback"))
        for tip in payload.get("tips", []):
            print("  -", tip)

    # ---------------- Cleanup ----------------
    if audio_path != input_path:
        os.remove(audio_path)


# ------------------------------------------------------------------
# Entry
# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
