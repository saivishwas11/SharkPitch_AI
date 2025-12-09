# utils/audio_processing.py
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import logging
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma", ".aiff", ".aif"}


def _is_audio_file(path: str) -> bool:
    return Path(path).suffix.lower() in AUDIO_EXTS


def _ffmpeg_executable() -> Optional[str]:
    """Return ffmpeg executable path if found on PATH, else None."""
    return shutil.which("ffmpeg")


def extract_audio_from_video(input_path: str, sr: int = 16000, out_path: Optional[str] = None) -> str:
    """
    Extract audio from a video file into a mono WAV resampled to `sr`.
    Order of attempts:
      1) If input is already an audio file, return input_path.
      2) Try moviepy (python).
      3) Try external ffmpeg (must be on PATH).
    Returns:
      path to resulting WAV file (may be the original file if already audio).
    Raises:
      ImportError with actionable instructions if neither moviepy nor ffmpeg available.
    """
    input_path = str(input_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    # If already audio — assume acceptable and return
    if _is_audio_file(input_path):
        logger.info("Input appears to be an audio file; returning original path.")
        return input_path

    # Prepare output path
    if out_path is None:
        fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="extracted_")
        os.close(fd)
        out_path = tmp_wav
    else:
        out_path = str(out_path)

    # Attempt moviepy first (pure python)
    try:
        from moviepy.editor import VideoFileClip  # type: ignore
        logger.info("Using moviepy to extract audio...")
        clip = VideoFileClip(input_path)
        if clip.audio is None:
            clip.close()
            raise RuntimeError("No audio track found in video.")
        # write_audiofile accepts fps -> sample rate
        clip.audio.write_audiofile(out_path, fps=sr, nbytes=2, codec="pcm_s16le", verbose=False, logger=None)
        clip.close()
        logger.info(f"Audio extracted with moviepy -> {out_path}")
        return out_path
    except Exception as moviepy_exc:
        logger.warning(f"moviepy extraction failed or not available: {moviepy_exc!r}")

    # Fallback: ffmpeg via subprocess
    ffmpeg_path = _ffmpeg_executable()
    if ffmpeg_path:
        logger.info(f"Using ffmpeg ({ffmpeg_path}) to extract audio...")
        # Build command - produce mono 16-bit PCM WAV at sample rate sr
        cmd = [
            ffmpeg_path,
            "-y",               # overwrite
            "-i", input_path,   # input
            "-ac", "1",         # mono
            "-ar", str(sr),     # sample rate
            "-vn",              # no video
            "-f", "wav",
            out_path,
        ]
        try:
            # run and capture output for debugging if needed
            res = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"ffmpeg extraction succeeded -> {out_path}")
            return out_path
        except subprocess.CalledProcessError as cpe:
            # include stdout/stderr to help debugging
            stderr = cpe.stderr or ""
            stdout = cpe.stdout or ""
            raise RuntimeError(
                f"ffmpeg failed with return code {cpe.returncode}. stdout: {stdout}\nstderr: {stderr}"
            ) from cpe
        except FileNotFoundError:
            # unlikely since we checked shutil.which, but handle gracefully
            logger.warning("ffmpeg executable not found despite shutil.which.")
    else:
        logger.warning("ffmpeg not found on PATH.")

    # If we get here both methods failed
    raise ImportError(
        "Video file support requires either 'moviepy' (pip install moviepy) "
        "or 'ffmpeg' available on your PATH (https://ffmpeg.org/download.html).\n\n"
        "Recommended fixes:\n"
        "  - Install moviepy in your conda environment:\n"
        "      conda activate <env>\n"
        "      pip install moviepy\n\n"
        "  - Or install ffmpeg and add to PATH (Windows):\n"
        "      1) Download a build, extract to C:\\ffmpeg\n"
        "      2) Add C:\\ffmpeg\\bin to System PATH, restart terminal\n"
        "      3) Verify with `ffmpeg -version`\n\n"
        "If you have ffmpeg installed but the command 'ffmpeg' is not found, make sure its bin folder is on your PATH."
    )


def _load_audio_file_to_numpy(path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Load audio file into numpy array.
    Prefer soundfile (faster & no heavy resampling). Fallback to librosa if needed.
    Returns (audio_array (mono), sample_rate)
    """
    try:
        import soundfile as sf
        data, file_sr = sf.read(path, dtype="float32")
        # if multi-channel, convert to mono by averaging channels
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr is not None and file_sr != sr:
            # use librosa for resampling if needed
            try:
                import librosa
                data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
                file_sr = sr
            except Exception:
                # fallback: use resampy if installed
                try:
                    import resampy
                    data = resampy.resample(data, file_sr, sr)
                    file_sr = sr
                except Exception:
                    logger.warning("Resampling failed; returning original sample rate.")
        return data.astype(np.float32), file_sr
    except Exception as e:
        logger.warning(f"soundfile load failed: {e}; falling back to librosa.")
        import librosa
        data, file_sr = librosa.load(path, sr=sr, mono=True)
        return data.astype(np.float32), file_sr


def load_and_prepare_audio(
    path: str,
    target_sr: int = 16000,
    normalize: bool = True,
    trim_silence: bool = False,
) -> Tuple[np.ndarray, int, str]:
    """
    High-level loader used by the rest of your app.
    - Accepts audio or video path.
    - Ensures there is a WAV file extracted if input was a video.
    - Loads into numpy array and resamples to target_sr.
    - Optionally normalizes peak to -1..1 and trims leading/trailing silence.

    Returns:
      (audio_numpy, sr, audio_file_path) where audio_file_path is the WAV file used.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input path not found: {path}")

    # If it's a video, extract audio to a temp WAV
    audio_path = path
    if not _is_audio_file(path):
        logger.info("Input is not an audio file — attempting to extract audio from video...")
        audio_path = extract_audio_from_video(path, sr=target_sr)

    # Load into numpy
    audio_np, sr = _load_audio_file_to_numpy(audio_path, sr=target_sr)

    # Optional trimming of silence (simple energy-based)
    if trim_silence:
        try:
            import librosa
            intervals = librosa.effects.split(audio_np, top_db=30)
            if intervals.size:
                # keep the largest contiguous region or concat all intervals
                segments = [audio_np[start:end] for start, end in intervals]
                audio_np = np.concatenate(segments)
            else:
                logger.info("No non-silent sections detected when trimming.")
        except Exception as e:
            logger.warning(f"trim_silence requested but librosa not available or failed: {e}")

    # Optional normalize peak
    if normalize and audio_np.size:
        peak = np.max(np.abs(audio_np))
        if peak > 0:
            audio_np = audio_np / peak

    return audio_np.astype(np.float32), sr, audio_path


# Optional: make explicit exports
__all__ = [
    "extract_audio_from_video",
    "load_and_prepare_audio",
]
