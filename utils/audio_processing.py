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
    return shutil.which("ffmpeg")


def extract_audio_from_video(input_path: str, sr: int = 16000, out_path: Optional[str] = None) -> str:
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

    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(input_path)
        if not clip.audio:
            clip.close()
            raise RuntimeError("Video has no audio track")
        clip.audio.write_audiofile(out_path, fps=sr, nbytes=2, codec="pcm_s16le", verbose=False)
        clip.close()
        return out_path
    except Exception as e:
        logger.warning("moviepy extract failed: %s", e)

    ffmpeg = _ffmpeg_executable()
    if ffmpeg:
        cmd = [
            ffmpeg, "-y", "-i", input_path,
            "-ac", "1",
            "-ar", str(sr),
            "-vn",
            "-f", "wav",
            out_path
        ]
        subprocess.run(cmd, check=True)
        return out_path

    raise ImportError(
        "moviepy or ffmpeg required for extracting audio from video"
    )


def _load_audio_file_to_numpy(path: str, sr: Optional[int]) -> Tuple[np.ndarray, int]:
    try:
        import soundfile as sf
        data, file_sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr and file_sr != sr:
            import librosa
            data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
            file_sr = sr
        return data.astype(np.float32), file_sr
    except Exception:
        import librosa
        data, file_sr = librosa.load(path, sr=sr, mono=True)
        return data.astype(np.float32), file_sr


def load_and_prepare_audio(path: str, target_sr: int = 16000,
                           normalize: bool = True,
                           trim_silence: bool = False) -> Tuple[np.ndarray, int, str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    audio_path = path
    if not _is_audio_file(path):
        audio_path = extract_audio_from_video(path, sr=target_sr)

    audio_np, sr = _load_audio_file_to_numpy(audio_path, sr=target_sr)

    if trim_silence:
        try:
            import librosa
            intervals = librosa.effects.split(audio_np, top_db=30)
            if intervals.size:
                audio_np = np.concatenate([audio_np[s:e] for s, e in intervals])
        except Exception:
            pass

    if normalize and audio_np.size:
        peak = np.max(np.abs(audio_np))
        if peak > 0:
            audio_np /= peak

    return audio_np.astype(np.float32), sr, audio_path
