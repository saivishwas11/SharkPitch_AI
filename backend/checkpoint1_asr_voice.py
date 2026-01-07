"""
Checkpoint 1: ASR & Voice Analysis
Tests audio processing, voice analysis, and speech-to-text transcription.

This checkpoint runs:
- audio_agent: Audio quality analysis
- voice_agent: Voice characteristics analysis
- asr_agent: Automatic Speech Recognition (transcription)

Usage:
    python checkpoint1_asr_voice.py <path_to_audio_or_video_file>
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any
from langgraph.graph import StateGraph, END

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.graph.state_types import PitchState
from backend.agents.audio_agent import audio_agent_node
from backend.agents.voice_agent import voice_agent_node
from backend.agents.asr_agent import asr_agent_node

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_checkpoint1_graph():
    """Build graph for Checkpoint 1: ASR & Voice Analysis"""
    workflow = StateGraph(PitchState)
    
    # Register nodes for checkpoint 1
    workflow.add_node("audio_agent", audio_agent_node)
    workflow.add_node("voice_agent", voice_agent_node)
    workflow.add_node("asr_agent", asr_agent_node)
    
    # Flow: audio -> voice -> asr -> END
    workflow.set_entry_point("audio_agent")
    workflow.add_edge("audio_agent", "voice_agent")
    workflow.add_edge("voice_agent", "asr_agent")
    workflow.add_edge("asr_agent", END)
    
    return workflow.compile()


def run_checkpoint1(input_file: str) -> Dict[str, Any]:
    """Run Checkpoint 1 pipeline"""
    try:
        # Validate input file
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Build and run the graph
        app = build_checkpoint1_graph()
        logger.info("✓ Checkpoint 1 graph compiled successfully")
        
        # Prepare initial state
        initial_state = {
            "input_path": str(path.resolve()),
        }
        
        logger.info(f"Starting Checkpoint 1 with input: {input_file}")
        logger.info("=" * 80)
        
        # Run the graph
        result = app.invoke(initial_state)
        
        logger.info("=" * 80)
        logger.info("✓ Checkpoint 1 completed successfully!")
        
        return result
        
    except Exception as e:
        logger.exception(f"✗ Checkpoint 1 failed: {str(e)}")
        raise


def print_checkpoint1_results(result: Dict[str, Any]):
    """Pretty print Checkpoint 1 results"""
    print("\n" + "=" * 80)
    print("CHECKPOINT 1 RESULTS: ASR & VOICE ANALYSIS")
    print("=" * 80)
    
    # Audio Stats
    if "audio_stats" in result:
        print("\n📊 AUDIO STATISTICS:")
        print("-" * 80)
        audio_stats = result["audio_stats"]
        for key, value in audio_stats.items():
            print(f"  {key}: {value}")
    
    # Voice Stats (cleaned up to avoid duplicate transcript)
    if "voice_stats" in result:
        print("\n🎤 VOICE ANALYSIS:")
        print("-" * 80)
        voice_stats = result["voice_stats"]
        for key, value in voice_stats.items():
            # Skip showing the full transcript inside filler_words
            if key == "filler_words" and isinstance(value, dict):
                filler_summary = {k: v for k, v in value.items() if k != "transcript"}
                print(f"  {key}: {filler_summary}")
            else:
                print(f"  {key}: {value}")
    
    # Transcript
    if "transcript" in result:
        print("\n📝 TRANSCRIPT:")
        print("-" * 80)
        transcript = result["transcript"]
        # Show first 500 characters as preview
        if len(transcript) > 500:
            print(f"{transcript[:500]}...")
            print(f"\n  [Full transcript: {len(transcript)} characters, {len(transcript.split())} words]")
        else:
            print(transcript)
    
    # Transcript Segments
    if "transcript_segments" in result and result["transcript_segments"]:
        print("\n📋 TRANSCRIPT SEGMENTS:")
        print("-" * 80)
        for i, segment in enumerate(result["transcript_segments"][:5], 1):  # Show first 5
            print(f"  [{i}] {segment}")
        if len(result["transcript_segments"]) > 5:
            print(f"  ... and {len(result['transcript_segments']) - 5} more segments")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python checkpoint1_asr_voice.py <path_to_audio_or_video_file>")
        print("\nExample:")
        print("  python checkpoint1_asr_voice.py sample_pitch.mp4")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        result = run_checkpoint1(input_file)
        print_checkpoint1_results(result)
        
        print("\n✅ Checkpoint 1 test completed successfully!")
        print("Next: Run checkpoint2_content_analysis.py to test content analysis\n")
        
    except Exception as e:
        print(f"\n❌ Checkpoint 1 test failed: {str(e)}\n")
        sys.exit(1)
