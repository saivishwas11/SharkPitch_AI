"""
voice_analysis.py

Voice analysis pipeline:
- Vocal features (pitch, pace, volume, pauses)
- Pitch behavior (robust ML-style heuristics)
- Emotional tone (confidence, nervousness, enthusiasm)
- Negative behaviors (fillers, hesitation, monotonicity)
- Delivery score (0–100)
- Full ASR transcript included
"""

from typing import Dict, Any
import numpy as np
import librosa
import logging
from datetime import datetime

from utils.asr import transcribe_audio_safe

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_LENGTH = 2048
HOP_LENGTH = 512

PITCH_MIN = 80
PITCH_MAX = 300

FILLER_WORDS = {
    "uh", "um", "er", "ah", "like", "you", "know",
    "i", "mean", "basically", "actually", "literally",
    "sort", "kind", "so"
}


# ------------------------------------------------------------------
# Audio loading
# ------------------------------------------------------------------

def load_audio(path: str):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    y, _ = librosa.effects.trim(y, top_db=35)
    if len(y) < sr:
        raise ValueError("Audio too short for analysis")
    return y, sr


# ------------------------------------------------------------------
# Pitch behavior (robust, non-zero)
# ------------------------------------------------------------------

def pitch_behavior_ml(f0: np.ndarray) -> Dict[str, float]:
    if len(f0) < 50:
        return {
            "confidence_from_pitch": 0.1,
            "expressiveness": 0.1,
            "monotonicity": 0.9,
        }

    diffs = np.abs(np.diff(f0))
    diffs = diffs[diffs < np.percentile(diffs, 95)]

    stability = 1.0 - np.clip(np.mean(diffs) / 25.0, 0, 1)
    expressiveness = np.clip(np.std(f0) / 50.0, 0, 1)
    monotonicity = 1.0 - expressiveness

    return {
        "confidence_from_pitch": round(stability, 3),
        "expressiveness": round(expressiveness, 3),
        "monotonicity": round(monotonicity, 3),
    }


# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------

def extract_vocal_features(path: str) -> Dict[str, Any]:
    y, sr = load_audio(path)

    f0 = librosa.yin(
        y,
        fmin=PITCH_MIN,
        fmax=PITCH_MAX,
        sr=sr,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )
    f0 = f0[np.isfinite(f0)]

    if len(f0):
        p10, p90 = np.percentile(f0, [10, 90])
        f0 = f0[(f0 >= max(60, p10 - 30)) & (f0 <= p90 + 30)]

    pitch_stats = {
        "mean": float(np.mean(f0)) if len(f0) else 0.0,
        "variance": float(np.var(f0)) if len(f0) else 0.0,
        "range": float(np.ptp(f0)) if len(f0) else 0.0,
    }

    pitch_ml = pitch_behavior_ml(f0)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    rms = librosa.feature.rms(
        y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]
    rms = rms[rms > 1e-6]

    volume_variation = float(np.std(rms) / np.mean(rms)) if len(rms) else 0.0

    threshold = np.percentile(rms, 25) if len(rms) else 0.0
    is_speech = rms > threshold

    pauses = {
        "count": int(np.sum(np.diff(is_speech.astype(int)) == -1)),
        "density": float(1.0 - np.mean(is_speech)) if len(is_speech) else 0.0,
    }

    return {
        "pitch": pitch_stats,
        "pitch_ml": pitch_ml,
        "pace_bpm": float(tempo),
        "volume_variation": volume_variation,
        "pauses": pauses,
        "duration_sec": len(y) / sr,
    }


# ------------------------------------------------------------------
# Emotion detection
# ------------------------------------------------------------------

def detect_emotion(features: Dict[str, Any]) -> Dict[str, float]:
    pvar = features["pitch"]["variance"]
    prange = features["pitch"]["range"]
    tempo = features["pace_bpm"]
    pauses = features["pauses"]["density"]
    vol = features["volume_variation"]

    confidence = (
        0.4 * min(1.0, pvar / 200_000) +
        0.4 * min(1.0, vol * 2) +
        0.2 * (1.0 - pauses)
    )

    nervousness = (
        0.6 * min(1.0, pauses * 2) +
        0.4 * (1.0 - min(1.0, prange / 350))
    )

    enthusiasm = (
        0.5 * min(1.0, tempo / 160) +
        0.3 * min(1.0, prange / 500) +
        0.2 * min(1.0, vol * 2)
    )

    return {
        "confidence": round(confidence, 3),
        "nervousness": round(nervousness, 3),
        "enthusiasm": round(enthusiasm, 3),
    }


# ------------------------------------------------------------------
# Filler words (ASR-based)
# ------------------------------------------------------------------

def analyze_fillers(audio_path: str) -> Dict[str, Any]:
    asr = transcribe_audio_safe(audio_path)
    transcript = asr.get("transcript", "")
    words = transcript.lower().split()

    filler_count = sum(1 for w in words if w in FILLER_WORDS)
    ratio = filler_count / len(words) if words else 0.0

    return {
        "filler_word_count": filler_count,
        "filler_word_ratio": round(ratio, 3),
        "total_words": len(words),
        "transcript": transcript,
    }


# ------------------------------------------------------------------
# Negative behaviors
# ------------------------------------------------------------------

def detect_negative_behaviors(features: Dict[str, Any], filler_ratio: float) -> Dict[str, float]:
    prange = features["pitch"]["range"]
    pvar = features["pitch"]["variance"]
    pauses = features["pauses"]["density"]

    monotonicity = max(0.0, 1.0 - (pvar / 200_000 + prange / 400))
    hesitation = min(1.0, pauses * 2)

    return {
        "hesitation_index": round(hesitation, 3),
        "monotonicity": round(monotonicity, 3),
        "filler_word_ratio": filler_ratio,
    }


# ------------------------------------------------------------------
# Delivery score
# ------------------------------------------------------------------

def compute_delivery_score(emotion: Dict[str, float], neg: Dict[str, float]) -> float:
    score = 85.0
    score -= neg["filler_word_ratio"] * 30
    score -= neg["hesitation_index"] * 25
    score -= neg["monotonicity"] * 20
    score += emotion["confidence"] * 12
    score += emotion["enthusiasm"] * 8
    return round(max(0.0, min(100.0, score)), 1)


# ------------------------------------------------------------------
# PUBLIC API
# ------------------------------------------------------------------

def analyze_voice(audio_path: str) -> Dict[str, Any]:
    features = extract_vocal_features(audio_path)
    emotion = detect_emotion(features)
    fillers = analyze_fillers(audio_path)
    negatives = detect_negative_behaviors(features, fillers["filler_word_ratio"])
    score = compute_delivery_score(emotion, negatives)

    return {
        "vocal_features": features,
        "emotion": emotion,
        "negative_behaviors": negatives,
        "filler_words": fillers,
        "transcript": fillers["transcript"],
        "delivery_score": score,
        "timestamp": datetime.now().isoformat(),
    }
