import logging
import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from backend.graph.state_types import PitchState
from backend.utils.audio_processing import load_and_prepare_audio

logger = logging.getLogger(__name__)

def validate_audio_output(audio_np: Optional[np.ndarray], 
                         sr: int, 
                         clean_path: str) -> Tuple[bool, str]:
    """Validate the output of audio processing."""
    if not os.path.exists(clean_path):
        return False, f"Output audio file not found at {clean_path}"
    
    if audio_np is not None and len(audio_np) == 0:
        return False, "Audio processing resulted in empty audio data"
    
    if sr <= 0:
        return False, f"Invalid sample rate: {sr}"
    
    return True, ""

def audio_agent_node(state: PitchState) -> PitchState:
    """
    Prepares audio: accepts video or audio path, processes accordingly.
    For audio files: Directly processes the audio file.
    For video files: Extracts audio first, then processes it.
    
    Args:
        state: The current state dictionary containing at least 'input_path'
        
    Returns:
        Updated state with 'clean_audio_path' and metadata if successful,
        or error information if processing fails.
    """
    # Initialize error tracking
    state.setdefault("audio_errors", [])
    
    # Get and validate input path
    path = state.get("input_path")
    if not path:
        error_msg = "input_path not provided in state"
        logger.error(error_msg)
        state["audio_errors"].append(error_msg)
        return state
        
    path = Path(path).resolve()
    if not path.exists():
        error_msg = f"Input file not found: {path}"
        logger.error(error_msg)
        state["audio_errors"].append(error_msg)
        return state
        
    logger.info(f"Processing input file: {path}")

    try:
        # Use shared utility that handles both audio and video paths
        logger.info("Loading and preparing audio (handles audio or video automatically)...")
        audio_np, sr, clean_path = load_and_prepare_audio(str(path), target_sr=16000)

        # Determine if original input was audio based on extension
        is_audio = path.suffix.lower() in {
            ".wav",
            ".mp3",
            ".flac",
            ".m4a",
            ".aac",
            ".ogg",
            ".wma",
            ".aiff",
            ".aif",
        }

        # Validate the output
        is_valid, error_msg = validate_audio_output(audio_np, sr, clean_path)
        if not is_valid:
            raise ValueError(f"Audio validation failed: {error_msg}")
        
        # Update state with results
        state["clean_audio_path"] = clean_path
        state["audio_sr"] = sr
        state["audio_duration"] = len(audio_np) / sr if audio_np is not None else 0
        state["audio_array_shape"] = audio_np.shape if audio_np is not None else None
        state["is_audio_file"] = is_audio  # Add flag indicating if input was audio
        
        logger.info(f"Successfully processed {'audio' if is_audio else 'video'}: {clean_path}")
        logger.debug(f"Sample rate: {sr}Hz, Duration: {state['audio_duration']:.2f}s")
        
    except Exception as e:
        error_msg = f"Audio processing failed: {str(e)}"
        logger.exception(error_msg)
        state["audio_errors"].append(error_msg)
        
        # Add more context to the error if possible
        if "No such file or directory" in str(e):
            state["audio_errors"].append(
                "This might be due to missing ffmpeg. "
                "Try installing it with: 'conda install -c conda-forge ffmpeg' or 'apt-get install ffmpeg'"
            )
    
    return state
