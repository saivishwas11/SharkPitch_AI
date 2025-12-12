from typing import Dict, Any, Tuple

import librosa
import numpy as np


def _load_audio(path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def compute_basic_prosody(path: str, sr: int = 16000) -> Dict[str, float]:
    y, sr = _load_audio(path, sr)

    duration = len(y) / sr
    energy = float(np.mean(y ** 2))
    energy_var = float(np.var(y ** 2))

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    pitch_mean = float(np.mean(spectral_centroid))
    pitch_var = float(np.var(spectral_centroid))

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = float(np.mean(zcr))
    zcr_var = float(np.var(zcr))

    try:
        pitches, mags = librosa.piptrack(y=y, sr=sr)
        values = []
        for t in range(pitches.shape[1]):
            idx = mags[:, t].argmax()
            v = pitches[idx, t]
            if v > 0:
                values.append(v)
        if values:
            f0_mean = float(np.mean(values))
            f0_std = float(np.std(values))
            f0_range = float(np.max(values) - np.min(values))
        else:
            f0_mean = f0_std = f0_range = 0.0
    except Exception:
        f0_mean = f0_std = f0_range = 0.0

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)
    except Exception:
        tempo = 0.0

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = float(np.mean(mfccs))
    mfcc_var = float(np.var(mfccs))

    frame_len, hop = 2048, 512
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    threshold = np.percentile(rms, 30)
    is_speech = rms > threshold
    speech_ratio = np.sum(is_speech) / len(rms)

    return {
        "duration_sec": duration,
        "energy": energy,
        "energy_variance": energy_var,
        "pitch_mean": pitch_mean,
        "pitch_variance": pitch_var,
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_range": f0_range,
        "tempo": tempo,
        "zcr_mean": zcr_mean,
        "zcr_variance": zcr_var,
        "speaking_rate": speech_ratio * 100,
        "mfcc_mean": mfcc_mean,
        "mfcc_variance": mfcc_var,
    }


def compute_pauses_and_pace(path: str, sr: int = 16000) -> Dict[str, float]:
    y, sr = _load_audio(path, sr)
    frame_length, hop = 2048, 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    threshold = np.percentile(rms, 25)
    is_speech = rms > threshold
    pauses = []
    in_pause = False

    for i, flag in enumerate(is_speech):
        if not flag and not in_pause:
            in_pause = True
            start = times[i]
        if flag and in_pause:
            end = times[i]
            pauses.append((start, end))
            in_pause = False

    pause_durations = [e - s for s, e in pauses]
    total_pause = sum(pause_durations)
    duration = len(y) / sr
    pause_density = total_pause / duration if duration else 0

    pace_events = len(pauses) + 1
    pace_events_min = (pace_events / duration) * 60 if duration else 0

    return {
        "pause_count": len(pauses),
        "pause_avg_sec": float(np.mean(pause_durations)) if pause_durations else 0.0,
        "pause_density": pause_density,
        "total_pause_sec": total_pause,
        "pace_events_per_min": pace_events_min,
    }


def compute_volume_stats(path: str, sr: int = 16000) -> Dict[str, float]:
    y, sr = _load_audio(path, sr)
    rms = librosa.feature.rms(y=y)[0]
    return {
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "volume_variation": float(np.std(rms) / (np.mean(rms) + 1e-6)),
    }


def detect_emotional_tone(prosody: Dict[str, float], pauses: Dict[str, float]) -> Dict[str, Any]:
    pitch_var = prosody.get("pitch_variance", 0)
    energy = prosody.get("energy", 0)
    tempo = prosody.get("tempo", 0)
    pause_density = pauses.get("pause_density", 0)

    enthusiasm = (pitch_var / 500000.0) + (energy * 10) + (tempo / 200)
    calm = max(0.0, 1.0 - pause_density * 2)
    nervous = pause_density * 2.5 + max(0, 0.2 - energy) * 5

    scores = {
        "enthusiastic": enthusiasm,
        "confident": enthusiasm * 0.7 + calm * 0.6,
        "nervous": nervous,
        "neutral": calm,
    }
    label = max(scores, key=scores.get)
    confidence = float(scores[label] / (sum(scores.values()) + 1e-6))

    return {"label": label, "confidence": confidence, "scores": scores}


def detect_negative_behaviors(prosody: Dict[str, float], pauses: Dict[str, float]) -> Dict[str, Any]:
    pitch_var = prosody.get("pitch_variance", 0)
    f0_range = prosody.get("f0_range", 0)
    pause_density = pauses.get("pause_density", 0)

    monotone = float(max(0.0, 1.0 - pitch_var / 300000.0) * (1.0 - min(1.0, f0_range / 500)))
    hesitation = float(min(1.0, pause_density * 2.5))

    return {
        "monotone_score": monotone,
        "hesitation_index": hesitation,
        "filler_words": [],
    }


def compute_delivery_score(prosody: Dict[str, float],
                           pauses: Dict[str, float],
                           tone: Dict[str, Any]) -> float:

    score = 60.0
    pause_penalty = min(20, pauses.get("pause_density", 0) * 40)
    score -= pause_penalty

    energy_bonus = min(20, prosody.get("pitch_variance", 0) / 800000.0 * 20)
    score += energy_bonus

    tempo = prosody.get("tempo", 0)
    if 90 < tempo < 150:
        score += 5

    label = tone.get("label", "neutral")
    conf = tone.get("confidence", 0)
    tone_map = {
        "confident": 15,
        "enthusiastic": 12,
        "neutral": 8,
        "nervous": -10,
    }
    score += tone_map.get(label, 0) * (0.5 + conf)

    return float(max(0, min(100, score)))


def analyze_voice(path: str) -> Dict[str, Any]:
    prosody = compute_basic_prosody(path)
    pauses = compute_pauses_and_pace(path)
    volume = compute_volume_stats(path)
    tone = detect_emotional_tone(prosody, pauses)
    neg = detect_negative_behaviors(prosody, pauses)
    delivery = compute_delivery_score(prosody, pauses, tone)

    return {
        "prosody": prosody,
        "pace_pauses": pauses,
        "volume": volume,
        "emotional_tone": tone,
        "negative_behaviors": neg,
        "delivery_score": delivery,
    }
