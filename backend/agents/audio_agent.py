import logging
import os
import sys
from pathlib import Path
from typing import Any, Tuple, Optional

import numpy as np
from graph.state_types import PitchState

# Import audio processing utilities
try:
    from utils.audio_processing import _load_audio_file_to_numpy, extract_audio_from_video
except ImportError as e:
    raise ImportError(
        "Failed to import audio_processing utils. Make sure all dependencies are installed. "
        "Run 'pip install -r requirements.txt' and ensure all audio processing libraries are available."
    ) from e

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
        # Check if input is an audio file
        is_audio = path.suffix.lower() in {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wma', '.aiff', '.aif'}
        
        if is_audio:
            logger.info("Detected audio file, processing directly...")
            # For audio files, we can process directly without video extraction
            audio_np, sr = _load_audio_file_to_numpy(str(path), sr=16000)
            clean_path = str(path)  # Use original path for audio files
        else:
            logger.info("Detected video file, extracting audio...")
            # For video files, extract audio first
            from utils.audio_processing import extract_audio_from_video, _load_audio_file_to_numpy
            audio_path = extract_audio_from_video(str(path), sr=16000)
            audio_np, sr = _load_audio_file_to_numpy(audio_path, sr=16000)
            clean_path = audio_path
        
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
