"""
Checkpoint 2: Content & Business Analysis
Tests content analysis and business evaluation.

This checkpoint runs:
- audio_agent: Audio quality analysis
- voice_agent: Voice characteristics analysis
- asr_agent: Automatic Speech Recognition (transcription)
- master_agent: Content and business analysis

Usage:
    python checkpoint2_content_analysis.py <path_to_audio_or_video_file>
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
from backend.content.master_agent import master_agent_node

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_checkpoint2_graph():
    """Build graph for Checkpoint 2: Content & Business Analysis"""
    workflow = StateGraph(PitchState)
    
    # Register nodes for checkpoint 2
    workflow.add_node("audio_agent", audio_agent_node)
    workflow.add_node("voice_agent", voice_agent_node)
    workflow.add_node("asr_agent", asr_agent_node)
    workflow.add_node("master_agent", master_agent_node)
    
    # Flow: audio -> voice -> asr -> master -> END
    workflow.set_entry_point("audio_agent")
    workflow.add_edge("audio_agent", "voice_agent")
    workflow.add_edge("voice_agent", "asr_agent")
    workflow.add_edge("asr_agent", "master_agent")
    workflow.add_edge("master_agent", END)
    
    return workflow.compile()


def run_checkpoint2(input_file: str) -> Dict[str, Any]:
    """Run Checkpoint 2 pipeline"""
    try:
        # Validate input file
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Build and run the graph
        app = build_checkpoint2_graph()
        logger.info("✓ Checkpoint 2 graph compiled successfully")
        
        # Prepare initial state
        initial_state = {
            "input_path": str(path.resolve()),
        }
        
        logger.info(f"Starting Checkpoint 2 with input: {input_file}")
        logger.info("=" * 80)
        
        # Run the graph
        result = app.invoke(initial_state)
        
        logger.info("=" * 80)
        logger.info("✓ Checkpoint 2 completed successfully!")
        
        return result
        
    except Exception as e:
        logger.exception(f"✗ Checkpoint 2 failed: {str(e)}")
        raise


def print_checkpoint2_results(result: Dict[str, Any]):
    """Pretty print Checkpoint 2 results"""
    print("\n" + "=" * 80)
    print("CHECKPOINT 2 RESULTS: CONTENT & BUSINESS ANALYSIS")
    print("=" * 80)
    
    # Audio Stats (brief)
    if "audio_stats" in result:
        print("\n📊 AUDIO STATISTICS (Summary):")
        print("-" * 80)
        audio_stats = result["audio_stats"]
        if "duration_seconds" in audio_stats:
            print(f"  Duration: {audio_stats['duration_seconds']} seconds")
        if "sample_rate" in audio_stats:
            print(f"  Sample Rate: {audio_stats['sample_rate']} Hz")
    
    # Voice Stats (brief)
    if "voice_stats" in result:
        print("\n🎤 VOICE ANALYSIS (Summary):")
        print("-" * 80)
        voice_stats = result["voice_stats"]
        if "confidence_score" in voice_stats:
            print(f"  Confidence Score: {voice_stats['confidence_score']}")
        if "speaking_rate" in voice_stats:
            print(f"  Speaking Rate: {voice_stats['speaking_rate']}")
    
    # Transcript (brief)
    if "transcript" in result:
        print("\n📝 TRANSCRIPT (Preview):")
        print("-" * 80)
        transcript = result["transcript"]
        preview = transcript[:300] + "..." if len(transcript) > 300 else transcript
        print(f"  {preview}")
    
    # Content Analysis (MAIN FOCUS)
    if "content_analysis" in result:
        print("\n💼 CONTENT & BUSINESS ANALYSIS:")
        print("=" * 80)
        content = result["content_analysis"]
        
        if isinstance(content, dict):
            for key, value in content.items():
                print(f"\n  {key.upper().replace('_', ' ')}:")
                print(f"  {'-' * 76}")
                if isinstance(value, (dict, list)):
                    import json
                    print(f"  {json.dumps(value, indent=4)}")
                else:
                    print(f"  {value}")
        else:
            print(f"  {content}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python checkpoint2_content_analysis.py <path_to_audio_or_video_file>")
        print("\nExample:")
        print("  python checkpoint2_content_analysis.py sample_pitch.mp4")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        result = run_checkpoint2(input_file)
        print_checkpoint2_results(result)
        
        print("\n✅ Checkpoint 2 test completed successfully!")
        print("Next: Run checkpoint3_sharks.py to test shark panel analysis\n")
        
    except Exception as e:
        print(f"\n❌ Checkpoint 2 test failed: {str(e)}\n")
        sys.exit(1)
