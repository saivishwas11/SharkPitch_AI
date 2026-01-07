"""
Checkpoint 3: Shark Panel Analysis
Tests the complete pipeline including all shark evaluations.

This checkpoint runs the FULL pipeline:
- audio_agent: Audio quality analysis
- voice_agent: Voice characteristics analysis
- asr_agent: Automatic Speech Recognition (transcription)
- master_agent: Content and business analysis
- visionary_shark: Innovation and vision evaluation
- finance_shark: Financial viability evaluation
- customer_shark: Market and customer evaluation
- skeptic_shark: Risk and challenge evaluation

Usage:
    python checkpoint3_sharks.py <path_to_audio_or_video_file>
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
from backend.sharks.visionary_shark import visionary_shark_node
from backend.sharks.finance_shark import finance_shark_node
from backend.sharks.customer_shark import customer_shark_node
from backend.sharks.skeptic_shark import skeptic_shark_node

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_checkpoint3_graph():
    """Build graph for Checkpoint 3: Full Pipeline with Shark Panel"""
    workflow = StateGraph(PitchState)
    
    # Register all nodes
    workflow.add_node("audio_agent", audio_agent_node)
    workflow.add_node("voice_agent", voice_agent_node)
    workflow.add_node("asr_agent", asr_agent_node)
    workflow.add_node("master_agent", master_agent_node)
    workflow.add_node("visionary_shark", visionary_shark_node)
    workflow.add_node("finance_shark", finance_shark_node)
    workflow.add_node("customer_shark", customer_shark_node)
    workflow.add_node("skeptic_shark", skeptic_shark_node)
    
    # Flow: audio -> voice -> asr -> master -> sharks -> END
    workflow.set_entry_point("audio_agent")
    workflow.add_edge("audio_agent", "voice_agent")
    workflow.add_edge("voice_agent", "asr_agent")
    workflow.add_edge("asr_agent", "master_agent")
    
    # Sequential sharks: master -> visionary -> finance -> customer -> skeptic
    workflow.add_edge("master_agent", "visionary_shark")
    workflow.add_edge("visionary_shark", "finance_shark")
    workflow.add_edge("finance_shark", "customer_shark")
    workflow.add_edge("customer_shark", "skeptic_shark")
    workflow.add_edge("skeptic_shark", END)
    
    return workflow.compile()


def run_checkpoint3(input_file: str) -> Dict[str, Any]:
    """Run Checkpoint 3 pipeline (FULL PIPELINE)"""
    try:
        # Validate input file
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Build and run the graph
        app = build_checkpoint3_graph()
        logger.info("✓ Checkpoint 3 graph compiled successfully")
        
        # Prepare initial state
        initial_state = {
            "input_path": str(path.resolve()),
        }
        
        logger.info(f"Starting Checkpoint 3 (FULL PIPELINE) with input: {input_file}")
        logger.info("=" * 80)
        
        # Run the graph
        result = app.invoke(initial_state)
        
        logger.info("=" * 80)
        logger.info("✓ Checkpoint 3 completed successfully!")
        
        return result
        
    except Exception as e:
        logger.exception(f"✗ Checkpoint 3 failed: {str(e)}")
        raise


def print_checkpoint3_results(result: Dict[str, Any]):
    """Pretty print Checkpoint 3 results"""
    print("\n" + "=" * 80)
    print("CHECKPOINT 3 RESULTS: FULL PIPELINE WITH SHARK PANEL")
    print("=" * 80)
    
    # Brief summaries of earlier stages
    if "audio_stats" in result and "duration_seconds" in result["audio_stats"]:
        print(f"\n📊 Audio Duration: {result['audio_stats']['duration_seconds']} seconds")
    
    if "voice_stats" in result and "confidence_score" in result["voice_stats"]:
        print(f"🎤 Voice Confidence: {result['voice_stats']['confidence_score']}")
    
    if "transcript" in result:
        transcript = result["transcript"]
        preview = transcript[:200] + "..." if len(transcript) > 200 else transcript
        print(f"\n📝 Transcript Preview: {preview}")
    
    # Content Analysis (brief)
    if "content_analysis" in result:
        print("\n💼 CONTENT ANALYSIS:")
        print("-" * 80)
        content = result["content_analysis"]
        if isinstance(content, dict):
            for key in list(content.keys())[:3]:  # Show first 3 keys
                print(f"  ✓ {key.replace('_', ' ').title()}")
        else:
            preview = str(content)[:200] + "..." if len(str(content)) > 200 else str(content)
            print(f"  {preview}")
    
    # Shark Panel (MAIN FOCUS)
    if "shark_panel" in result:
        print("\n🦈 SHARK PANEL EVALUATIONS:")
        print("=" * 80)
        shark_panel = result["shark_panel"]
        
        sharks = [
            ("visionary", "🔮 VISIONARY SHARK"),
            ("finance", "💰 FINANCE SHARK"),
            ("customer", "👥 CUSTOMER SHARK"),
            ("skeptic", "⚠️  SKEPTIC SHARK")
        ]
        
        for shark_key, shark_title in sharks:
            if shark_key in shark_panel:
                print(f"\n{shark_title}:")
                print("-" * 80)
                shark_feedback = shark_panel[shark_key]
                
                if isinstance(shark_feedback, dict):
                    import json
                    print(json.dumps(shark_feedback, indent=2))
                else:
                    print(shark_feedback)
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python checkpoint3_sharks.py <path_to_audio_or_video_file>")
        print("\nExample:")
        print("  python checkpoint3_sharks.py sample_pitch.mp4")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        result = run_checkpoint3(input_file)
        print_checkpoint3_results(result)
        
        print("\n✅ Checkpoint 3 test completed successfully!")
        print("🎉 All checkpoints passed! Your SharkPitch AI pipeline is working!\n")
        
    except Exception as e:
        print(f"\n❌ Checkpoint 3 test failed: {str(e)}\n")
        sys.exit(1)
