import logging
import sys
import json
import os
from pathlib import Path
from typing import Dict, Any

from graph.graph_multiagent import build_graph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('athena_task.log')
    ]
)
logger = logging.getLogger(__name__)

def validate_input_file(file_path: str) -> bool:
    """Validate that the input file exists and is a supported format."""
    path = Path(file_path)
    if not path.exists():
        logger.error(f"Input file not found: {file_path}")
        return False
    
    # Check for supported file extensions
    supported_extensions = ['.mp4', '.avi', '.mov', '.wav', '.mp3']
    if path.suffix.lower() not in supported_extensions:
        logger.error(f"Unsupported file format: {path.suffix}. Supported formats: {', '.join(supported_extensions)}")
        return False
    
    return True

def run_pipeline(input_file: str) -> Dict[str, Any]:
    """Run the entire pipeline with the given input file."""
    try:
        # Build the graph
        app = build_graph()
        logger.info("Graph compiled successfully")

        # Prepare initial state
        initial_state = {
            "input_path": str(Path(input_file).resolve())
        }

        # Run the graph
        logger.info(f"Starting pipeline with input: {input_file}")
        
        # The compiled graph should be invoked using the invoke() method
        if hasattr(app, "invoke"):
            logger.debug("Using invoke() method to run the graph")
            result = app.invoke(initial_state)
        elif hasattr(app, "__call__"):
            logger.debug("Using __call__ method to run the graph")
            result = app(initial_state)
        elif hasattr(app, "run"):
            logger.debug("Using run() method to run the graph")
            result = app.run(initial_state)
        else:
            raise RuntimeError(
                "Graph execution failed. The compiled graph doesn't support any known execution methods. "
                f"Available methods: {[m for m in dir(app) if not m.startswith('_')]}"
            )
        
        logger.info("Pipeline completed successfully")
        return result
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {str(e)}")
        raise

def format_voice_analysis(voice_stats):
    """Format voice analysis results into a clean, detailed string."""
    if not voice_stats:
        return "No voice analysis results available."
    
    output = []
    output.append("\n" + "="*50)
    output.append("VOICE ANALYSIS REPORT".center(50))
    output.append("="*50)
    
    # 1. Overall Delivery Score
    delivery_score = voice_stats.get('delivery_score', 0)
    score_emoji = "🎯"
    if delivery_score >= 80:
        score_emoji = "🌟"
    elif delivery_score >= 60:
        score_emoji = "👍"
    output.append(f"\n{score_emoji} Overall Delivery Score: {delivery_score}/100")
    
    # 2. Vocal Features
    output.append("\n" + "-"*50)
    output.append("VOCAL FEATURES".center(50))
    output.append("-"*50)
    
    if 'prosody' in voice_stats:
        p = voice_stats['prosody']
        
        # Helper function to safely format numeric values
        def format_metric(value, format_str='.1f', suffix=''):
            if isinstance(value, (int, float)):
                return f"{value:{format_str}}{suffix}"
            return str(value)
        
        # Get and format metrics safely
        volume = p.get('volume_variation', 'N/A')
        pitch_mean = format_metric(p.get('pitch_mean'), '.1f', ' Hz')
        pitch_range = p.get('pitch_range', 'N/A')
        speaking_rate = format_metric(p.get('speaking_rate'), '.1f', ' wpm')
        pause_freq = p.get('pause_frequency', 'N/A')
        avg_pause = format_metric(p.get('avg_pause_length'), '.1f', 's')
        
        output.append(f"🔊 Volume: {volume} (variation)")
        output.append(f"🎵 Pitch: {pitch_mean} (avg), {pitch_range} (range)")
        output.append(f"⏱️  Pace: {speaking_rate}")
        output.append(f"⏸️  Pauses: {pause_freq} pauses/min, avg length: {avg_pause}")
    
    # 3. Emotional Tone Analysis
    output.append("\n" + "-"*50)
    output.append("EMOTIONAL TONE".center(50))
    output.append("-"*50)
    
    if 'tone' in voice_stats:
        tone = voice_stats['tone']
        primary_tone = tone.get('label', 'neutral').title()
        confidence = tone.get('confidence', 0) * 100
        
        # Add emoji based on tone
        tone_emoji = {
            'Happy': '😊', 'Excited': '🎉', 'Confident': '💪', 
            'Neutral': '😐', 'Nervous': '😬', 'Monotone': '😶'
        }.get(primary_tone, '🔍')
        
        output.append(f"{tone_emoji} Primary Tone: {primary_tone} ({confidence:.1f}% confidence)")
        
        # Show tone distribution
        if 'scores' in tone:
            output.append("\nTone Distribution:")
            for t, score in sorted(tone['scores'].items(), key=lambda x: x[1], reverse=True):
                bar = '█' * int(score * 20)
                output.append(f"  {t.title():<12} {bar} {score*100:.0f}%")
    
    # 4. Areas for Improvement
    output.append("\n" + "-"*50)
    output.append("AREAS FOR IMPROVEMENT".center(50))
    output.append("-"*50)
    
    if 'negative_behaviors' in voice_stats:
        neg = voice_stats['negative_behaviors']
        issues = []
        
        # Filler words
        if neg.get('filler_words'):
            issues.append(f"🚫 Filler words: {', '.join(neg['filler_words'])}")
        
        # Monotonicity
        if neg.get('monotone_score', 0) > 0.6:
            issues.append(f"😐 Monotone speech (score: {neg['monotone_score']:.2f})")
        
        # Hesitation
        if neg.get('hesitation_index', 0) > 0.6:
            issues.append(f"🤔 High hesitation (index: {neg['hesitation_index']:.2f})")
        
        if issues:
            output.append("\n".join([f"• {issue}" for issue in issues]))
        else:
            output.append("🎉 No major issues detected!")
    
    # 5. Detailed Metrics
    output.append("\n" + "-"*50)
    output.append("DETAILED METRICS".center(50))
    output.append("-"*50)
    
    if 'prosody' in voice_stats:
        p = voice_stats['prosody']
        metrics = [
            ("Energy Level", f"{p.get('energy', 0):.1f}/5.0"),
            ("Tempo", f"{p.get('tempo', 0):.1f} BPM"),
            ("Pitch Variation", f"{p.get('pitch_variance', 0):.2f}"),
            ("Pause Frequency", f"{p.get('pause_frequency', 0):.1f}/min"),
            ("Speech Clarity", f"{p.get('clarity', 0):.1f}/5.0")
        ]
        
        for label, value in metrics:
            output.append(f"• {label:<18} {value:>10}")
    
    return '\n'.join(output)

def format_shark_feedback(panel):
    """Format shark panel feedback into a clean string."""
    if not panel:
        return "No shark panel feedback available."
    
    output = ["\n=== Shark Panel Feedback ===\n"]
    
    for shark, feedback in panel.items():
        if shark in ['final_verdict', 'verdict_counts']:
            continue
            
        output.append(f"{shark.capitalize()} Shark:")
        
        # Try to parse feedback if it's a JSON string
        try:
            if isinstance(feedback, str):
                feedback = json.loads(feedback)
            
            if 'summary' in feedback:
                output.append(f"  Summary: {feedback['summary']}")
            if 'verdict' in feedback:
                output.append(f"  Verdict: {feedback['verdict']}")
            if 'tips' in feedback and feedback['tips']:
                output.append("  Tips:")
                for tip in feedback['tips']:
                    output.append(f"    - {tip}")
        except (json.JSONDecodeError, TypeError):
            # If not JSON or parsing fails, just print the raw feedback
            output.append(f"  {str(feedback)[:200]}...")
        
        output.append("\n" + "-"*50 + "\n")
    
    # Add final verdict if available
    if 'final_verdict' in panel:
        output.append(f"\nFinal Verdict: {panel['final_verdict']}")
    
    return '\n'.join(output)

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_media_file>")
        print("Supported formats: .mp4, .avi, .mov, .wav, .mp3")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Validate input file
    if not validate_input_file(input_file):
        sys.exit(1)
    
    try:
        # Run the pipeline
        result = run_pipeline(input_file)
        
        # Clear console for cleaner output
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Print a summary of the results
        print("\n" + "="*50)
        print("PIPELINE EXECUTION SUMMARY".center(50))
        print("="*50)
        print(f"\nInput file: {input_file}")
        
        # Print voice analysis results if available
        if 'voice_stats' in result:
            print(format_voice_analysis(result['voice_stats']))
        
        # Print transcript if available
        if 'transcript' in result and result['transcript']:
            print("\n=== Transcript Preview ===")
            print(result['transcript'][:500] + (result['transcript'][500:] and '...'))
        
        # Print content analysis if available
        if 'content_analysis' in result and result['content_analysis']:
            print("\n=== Content Analysis ===")
            if isinstance(result['content_analysis'], dict):
                print(json.dumps(result['content_analysis'], indent=2, ensure_ascii=False))
            else:
                print(str(result['content_analysis'])[:1000] + '...')
        
        # Print shark panel feedback if available
        if 'shark_panel' in result and result['shark_panel']:
            print(format_shark_feedback(result['shark_panel']))
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE".center(50))
        print("="*50)
        print("\nCheck athena_task.log for detailed logs.")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nCheck athena_task.log for detailed error information.")
        logger.exception("Pipeline execution failed")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        print(f"Error: {str(e)}")
        print("Check athena_task.log for more details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
