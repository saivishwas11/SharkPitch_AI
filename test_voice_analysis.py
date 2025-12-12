#!/usr/bin/env python3
"""
Enhanced Voice Analysis Test Script

This script demonstrates the voice analysis capabilities by processing an audio file
and providing detailed feedback on vocal characteristics, emotional tone, and delivery.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.absolute()))

from utils.voice_analysis import analyze_voice

def format_emotion_results(emotion_data: Dict[str, Any]) -> str:
    """Format emotion analysis results for display."""
    if not emotion_data:
        return "No emotion data available."
    
    output = []
    primary = emotion_data.get('primary_emotion', 'unknown').title()
    confidence = emotion_data.get('emotion_confidence', 0) * 100
    
    output.append(f"\n🎭 Emotional Analysis")
    output.append(f"   Primary Emotion: {primary} ({confidence:.1f}% confidence)")
    
    # Show top emotion scores
    if 'emotion_scores' in emotion_data:
        output.append("\n   Emotion Distribution:")
        for emotion, score in sorted(
            emotion_data['emotion_scores'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]:  # Show top 5 emotions
            output.append(f"   - {emotion.title():<12}: {score*100:5.1f}%")
    
    return "\n".join(output)

def format_delivery_score(score_data: Dict[str, Any]) -> str:
    """Format delivery score results for display."""
    if not score_data:
        return "No delivery score available."
    
    output = []
    overall = score_data.get('overall_score', 0)
    
    # Get emoji based on score
    if overall >= 80:
        score_emoji = "🌟"
        rating = "Excellent"
    elif overall >= 60:
        score_emoji = "👍"
        rating = "Good"
    elif overall >= 40:
        score_emoji = "😐"
        rating = "Average"
    else:
        score_emoji = "📉"
        rating = "Needs Improvement"
    
    output.append(f"\n📊 Delivery Score: {overall:.1f}/100 {score_emoji} ({rating})")
    
    # Show score components if available
    if 'score_components' in score_data:
        comp = score_data['score_components']
        output.append("\n   Score Components:")
        output.append(f"   - Base Score:      {comp.get('base_score', 0):.1f}")
        output.append(f"   - Prosody:         {comp.get('prosody_score', 0):.1f}")
        output.append(f"   - Emotion:         {comp.get('emotion_score', 0):.1f}")
        output.append(f"   - Clarity:         {comp.get('clarity_component', 0):.1f}")
        output.append(f"   - Negative Impact: -{comp.get('negative_penalty', 0):.1f}")
    
    # Show improvement areas if any
    if 'improvement_areas' in score_data and score_data['improvement_areas']:
        output.append("\n   🎯 Areas for Improvement:")
        for area in score_data['improvement_areas']:
            priority = area.get('priority', 'medium').title()
            output.append(f"   - {priority} Priority: {area.get('suggestion', '')}")
    
    return "\n".join(output)

def format_speaking_stats(stats: Dict[str, Any]) -> str:
    """Format speaking statistics for display."""
    if not stats:
        return "No speaking statistics available."
    
    output = ["\n🗣️  Speaking Statistics"]
    
    # Basic stats
    output.append(f"   - Words per minute:     {stats.get('words_per_minute', 0):.1f}")
    output.append(f"   - Articulation rate:    {stats.get('articulation_rate', 0):.1f} syllables/sec")
    output.append(f"   - Speech clarity:       {stats.get('speech_clarity_score', 0):.1f}/5.0")
    
    # Pause analysis
    if 'pauses_per_minute' in stats:
        output.append(f"   - Pauses per minute:    {stats.get('pauses_per_minute', 0):.1f}")
    if 'avg_pause_duration' in stats:
        output.append(f"   - Avg. pause duration:  {stats.get('avg_pause_duration', 0):.2f}s")
    
    # Volume analysis
    if 'volume' in stats and isinstance(stats['volume'], dict):
        vol = stats['volume']
        output.append(f"   - Volume variation:     {vol.get('volume_variation', 0):.3f}")
        output.append(f"   - Dynamic range:        {vol.get('dynamic_range_db', 0):.1f} dB")
    
    return "\n".join(output)

def format_filler_words(filler_data: Dict[str, Any]) -> str:
    """Format filler word analysis for display."""
    if not filler_data or not filler_data.get('filler_words_found'):
        return "\n✅ No significant filler words detected."
    
    output = ["\n⚠️  Filler Words Detected"]
    
    total_fillers = filler_data.get('filler_word_count', 0)
    total_words = filler_data.get('total_words', 1)
    filler_ratio = (total_fillers / total_words) * 100
    
    output.append(f"   - Total fillers: {total_fillers} ({filler_ratio:.1f}% of words)")
    
    # Show top filler words
    fillers = sorted(
        filler_data['filler_words_found'], 
        key=lambda x: x.get('count', 0), 
        reverse=True
    )
    
    output.append("   - Most used fillers:")
    for fw in fillers[:5]:  # Show top 5 filler words
        word = fw.get('word', '')
        count = fw.get('count', 0)
        ratio = fw.get('ratio', 0) * 100
        output.append(f"     • '{word}': {count} times ({ratio:.1f}%)")
    
    # Add disfluency score if available
    if 'disfluency_score' in filler_data:
        score = filler_data['disfluency_score']
        output.append(f"   - Disfluency score: {score:.2f}/1.0")
    
    return "\n".join(output)

def analyze_audio_file(audio_path: str, output_json: bool = True) -> Dict[str, Any]:
    """Analyze an audio file and return the results."""
    print(f"\n🔍 Analyzing audio file: {Path(audio_path).name}")
    
    try:
        # Run the voice analysis
        results = analyze_voice(audio_path)
        
        # Print formatted results
        print("\n" + "="*60)
        print("VOICE ANALYSIS REPORT".center(60))
        print("="*60)
        
        # Print each section
        print(format_emotion_results(results.get('emotional_tone', {})))
        print(format_delivery_score(results.get('delivery_score', {})))
        print(format_speaking_stats(results.get('speaking_stats', {})))
        print(format_filler_words(results.get('filler_words', {})))
        
        # Save full results to JSON if requested
        if output_json:
            output_file = Path(audio_path).with_suffix('.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Full results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during voice analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Voice Analysis Tool - Analyze vocal characteristics and delivery"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        help="Path to the audio file to analyze"
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Disable saving results to JSON file"
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Show version
    if args.version:
        print("Voice Analysis Tool v1.0")
        return
    
    # Check if audio file was provided
    if not args.audio_file:
        print("Error: No audio file specified.")
        parser.print_help()
        return
    
    # Check if file exists
    if not os.path.isfile(args.audio_file):
        print(f"Error: File not found: {args.audio_file}")
        return
    
    # Analyze the audio file
    analyze_audio_file(args.audio_file, not args.no_json)

if __name__ == "__main__":
    main()
