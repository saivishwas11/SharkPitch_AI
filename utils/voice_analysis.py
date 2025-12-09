"""
Voice analysis utilities.
Extract vocal features (pitch, pace, volume variation, pauses), infer emotional tone,
and surface negative behaviors (hesitation/monotonicity). Delivery score is derived
from clarity, energy, and confidence proxies.
"""
from typing import Dict, Any, Tuple

import librosa
import numpy as np


def _load_audio(audio_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    return y, sr


def compute_basic_prosody(audio_path: str, sr: int = 16000) -> Dict[str, float]:
    """
    Core prosodic features: pitch, tempo, spectral stats, speaking ratio.
    """
    y, sr = _load_audio(audio_path, sr=sr)

    duration = len(y) / sr

    energy = float(np.mean(y ** 2))
    energy_var = float(np.var(y ** 2))

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

    pitch_var = float(np.var(spectral_centroid))
    pitch_mean = float(np.mean(spectral_centroid))

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = float(np.mean(zcr))
    zcr_var = float(np.var(zcr))

    try:
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)

        if pitch_values:
            f0_mean = float(np.mean(pitch_values))
            f0_std = float(np.std(pitch_values))
            f0_range = float(np.max(pitch_values) - np.min(pitch_values))
        else:
            f0_mean = f0_std = f0_range = 0.0
    except Exception:
        f0_mean = f0_std = f0_range = 0.0

    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
        tempo = float(tempo) if tempo > 0 else 0.0
    except Exception:
        tempo = 0.0

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = float(np.mean(mfccs))
    mfcc_var = float(np.var(mfccs))

    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    silence_threshold = np.percentile(rms, 30)
    speech_frames = np.sum(rms > silence_threshold)
    total_frames = len(rms)
    speech_ratio = speech_frames / total_frames if total_frames > 0 else 0.0
    speaking_rate = float(speech_ratio * 100.0)  # Percentage of time speaking

    return {
        "duration_sec": float(duration),
        "energy": energy,
        "energy_variance": energy_var,
        "pitch_mean": pitch_mean,
        "pitch_variance": pitch_var,
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_range": f0_range,
        "spectral_rolloff": float(np.mean(spectral_rolloff)),
        "spectral_bandwidth": float(np.mean(spectral_bandwidth)),
        "speaking_rate": speaking_rate,
        "zcr_mean": zcr_mean,
        "zcr_variance": zcr_var,
        "tempo": tempo,
        "mfcc_mean": mfcc_mean,
        "mfcc_variance": mfcc_var,
    }


def compute_pauses_and_pace(audio_path: str, sr: int = 16000) -> Dict[str, float]:
    """
    Identify pauses and estimate pace using RMS-based silence detection.
    """
    y, sr = _load_audio(audio_path, sr=sr)
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    threshold = np.percentile(rms, 25)
    is_speech = rms > threshold

    pauses = []
    in_pause = False
    start = 0.0
    for i, flag in enumerate(is_speech):
        if not flag and not in_pause:
            in_pause = True
            start = times[i]
        if flag and in_pause:
            end = times[i]
            pauses.append((start, end))
            in_pause = False
    if in_pause:
        pauses.append((start, times[-1]))

    pause_durations = [end - start for start, end in pauses]
    total_pause = float(sum(pause_durations))
    total_duration = float(len(y) / sr)
    pause_density = total_pause / total_duration if total_duration else 0.0

    # Pace proxy: syllable-ish energy bursts per minute
    speech_spans = np.sum(is_speech)
    speech_seconds = speech_spans * hop_length / sr
    pace_events = max(1, len(pauses) + 1)
    pace_events_per_min = (pace_events / max(1e-6, total_duration)) * 60.0

    return {
        "pause_count": len(pauses),
        "pause_avg_sec": float(np.mean(pause_durations)) if pause_durations else 0.0,
        "pause_density": pause_density,
        "total_pause_sec": total_pause,
        "pace_events_per_min": pace_events_per_min,
        "speech_seconds": speech_seconds,
    }


def compute_volume_stats(audio_path: str, sr: int = 16000) -> Dict[str, float]:
    """
    Volume variation metrics using RMS.
    """
    y, sr = _load_audio(audio_path, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    return {
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "volume_variation": float(np.std(rms) / (np.mean(rms) + 1e-6)),
    }


def detect_emotional_tone(prosody: Dict[str, float], pauses: Dict[str, float]) -> Dict[str, Any]:
    """
    Lightweight heuristic emotional tone based on energy, pitch variability, pace, and pauses.
    """
    energy = prosody.get("energy", 0)
    pitch_var = prosody.get("pitch_variance", 0)
    tempo = prosody.get("tempo", 0)
    pause_density = pauses.get("pause_density", 0)

    # Heuristic scoring
    enthusiasm_score = (pitch_var / 500000.0) + (energy * 10) + (tempo / 200.0)
    calm_score = max(0.0, 1.0 - pause_density * 2.0)
    nervousness_score = pause_density * 2.5 + max(0.0, 0.2 - energy) * 5

    # Pick dominant tone
    scores = {
        "enthusiastic": enthusiasm_score,
        "confident": enthusiasm_score * 0.7 + calm_score * 0.6,
        "nervous": nervousness_score,
        "neutral": calm_score,
    }
    label = max(scores, key=scores.get)
    confidence = float(min(1.0, scores[label] / (sum(scores.values()) + 1e-6)))

    rationale = {
        "energy": energy,
        "pitch_variance": pitch_var,
        "tempo": tempo,
        "pause_density": pause_density,
    }
    return {"label": label, "confidence": confidence, "scores": scores, "rationale": rationale}


def detect_negative_behaviors(prosody: Dict[str, float], pauses: Dict[str, float]) -> Dict[str, Any]:
    """
    Highlight negative patterns: monotonicity, hesitation. Filler words require transcript
    so are left empty here.
    """
    pitch_var = prosody.get("pitch_variance", 0.0)
    f0_range = prosody.get("f0_range", 0.0)
    pause_density = pauses.get("pause_density", 0.0)

    monotone_score = float(max(0.0, 1.0 - (pitch_var / 300000.0)) * (1.0 - min(1.0, f0_range / 500.0)))
    hesitation_index = float(min(1.0, pause_density * 2.5))

    return {
        "monotone_score": monotone_score,
        "hesitation_index": hesitation_index,
        "filler_words": [],  # requires transcript; left empty
    }


def compute_delivery_score(prosody: Dict[str, float], pauses: Dict[str, float], tone: Dict[str, Any]) -> float:
    """
    Delivery score based on clarity (pauses), energy (volume/pitch movement), confidence (tone).
    """
    score = 60.0

    # Clarity: fewer/shorter pauses
    pause_penalty = min(20.0, pauses.get("pause_density", 0) * 40.0)
    score -= pause_penalty

    # Energy: pitch variance + tempo
    pitch_var = prosody.get("pitch_variance", 0)
    tempo = prosody.get("tempo", 0)
    energy_bonus = min(20.0, pitch_var / 800000.0 * 20.0) + (5.0 if 90 < tempo < 150 else 0.0)
    score += energy_bonus

    # Confidence: based on tone label
    tone_label = tone.get("label", "neutral")
    tone_conf = tone.get("confidence", 0.0)
    tone_map = {
        "confident": 15.0,
        "enthusiastic": 12.0,
        "neutral": 8.0,
        "nervous": -10.0,
    }
    score += tone_map.get(tone_label, 0.0) * (0.5 + tone_conf)

    return float(max(0.0, min(100.0, score)))


def analyze_voice(audio_path: str) -> Dict[str, Any]:
    prosody = compute_basic_prosody(audio_path)
    pauses = compute_pauses_and_pace(audio_path)
    volume = compute_volume_stats(audio_path)
    emotional_tone = detect_emotional_tone(prosody, pauses)
    negative_behaviors = detect_negative_behaviors(prosody, pauses)
    delivery_score = compute_delivery_score(prosody, pauses, emotional_tone)

    return {
        "prosody": prosody,
        "pace_pauses": pauses,
        "volume": volume,
        "emotional_tone": emotional_tone,
        "negative_behaviors": negative_behaviors,
        "delivery_score": delivery_score,
    }
