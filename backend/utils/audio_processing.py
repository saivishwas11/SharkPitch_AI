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

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma", ".aiff", ".aif"}


def _is_audio_file(path: str) -> bool:
    return Path(path).suffix.lower() in AUDIO_EXTS


def _ffmpeg_executable() -> Optional[str]:
    return shutil.which("ffmpeg")


def extract_audio_from_video(
    input_path: str,
    sr: int = 16000,
    out_path: Optional[str] = None
) -> str:
    input_path = str(input_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    if _is_audio_file(input_path):
        logger.info("Input already audio; returning original")
        return input_path

    if out_path is None:
        fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        out_path = tmp_wav

    # ---- moviepy first ----
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(input_path)
        if not clip.audio:
            clip.close()
            raise RuntimeError("Video has no audio track")

        clip.audio.write_audiofile(
            out_path,
            fps=sr,
            nbytes=2,
            codec="pcm_s16le",
            verbose=False,
            logger=None,
        )
        clip.close()
        logger.info("Audio extracted using moviepy")
        return out_path
    except Exception as e:
        logger.warning("moviepy failed, falling back to ffmpeg: %s", e)

    # ---- ffmpeg fallback ----
    ffmpeg = _ffmpeg_executable()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")

    cmd = [
        ffmpeg, "-y", "-i", input_path,
        "-ac", "1",
        "-ar", str(sr),
        "-vn",
        "-f", "wav",
        out_path
    ]

    subprocess.run(cmd, check=True)
    logger.info("Audio extracted using ffmpeg")
    return out_path


def load_and_prepare_audio(
    path: str,
    target_sr: int = 16000,
    normalize: bool = True,
    trim_silence: bool = False
) -> Tuple[np.ndarray, int, str]:

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    audio_path = path
    if not _is_audio_file(path):
        audio_path = extract_audio_from_video(path, sr=target_sr)

    try:
        import soundfile as sf
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
    except Exception:
        import librosa
        data, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    if trim_silence:
        try:
            import librosa
            intervals = librosa.effects.split(data, top_db=30)
            if intervals.size:
                data = np.concatenate([data[s:e] for s, e in intervals])
        except Exception:
            pass

    if normalize and data.size:
        peak = np.max(np.abs(data))
        if peak > 0:
            data /= peak

    logger.info(
        f"Audio prepared | samples={len(data)} | "
        f"duration={len(data)/sr:.2f}s"
    )

    return data.astype(np.float32), sr, audio_path
