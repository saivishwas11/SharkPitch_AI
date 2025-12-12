from typing import Dict, Any, Tuple, List, Optional
import re
import speech_recognition as sr
import librosa
import numpy as np
from collections import Counter

# Common filler words by language
FILLER_WORDS = {
    'en': [
        'uh', 'um', 'er', 'ah', 'like', 'you know', 'so', 'right', 'basically',
        'actually', 'literally', 'technically', 'i mean', 'sort of', 'kind of',
        'you see', 'you know what i mean', 'well', 'i guess', 'or something',
        'right', 'i think', 'i suppose', 'you know', 'i mean', 'you know what',
        'sort of', 'kind of', 'i guess', 'i suppose', 'or something', 'i think',
        'you know what i mean', 'you know what i\'m saying', 'i tell you', 'i tell ya'
    ]
}

# Emotional tone mapping with more nuanced categories
EMOTION_CATEGORIES = [
    'excited', 'happy', 'confident', 'neutral', 'calm',
    'uncertain', 'nervous', 'monotone', 'frustrated'
]


def _load_audio(path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file and return audio data and sample rate."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def _transcribe_audio(audio_path: str, language: str = 'en-US') -> Optional[str]:
    """Transcribe audio to text using Google's speech recognition."""
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language=language)
            return text.lower()
    except Exception as e:
        logger.warning(f"Speech recognition failed: {e}")
        return None


def detect_filler_words(text: str, language: str = 'en') -> Dict[str, Any]:
    """Detect and count filler words in the transcribed text."""
    if not text:
        return {
            'filler_words_found': [],
            'filler_word_count': 0,
            'total_words': 0,
            'filler_word_ratio': 0.0
        }
    
    # Clean and tokenize text
    words = re.findall(r'\b\w+\b', text)
    total_words = len(words)
    
    # Find filler words
    fillers = []
    filler_counter = Counter()
    
    for filler in FILLER_WORDS.get(language, FILLER_WORDS['en']):
        # Count single-word fillers
        if ' ' not in filler:
            count = words.count(filler)
            if count > 0:
                filler_counter[filler] += count
        # Handle multi-word fillers
        else:
            # Create a regex pattern that matches the filler phrase
            pattern = r'\b' + re.escape(filler) + r'\b'
            matches = re.findall(pattern, text)
            if matches:
                filler_counter[filler] += len(matches)
    
    filler_words_found = [{'word': k, 'count': v} for k, v in filler_counter.items()]
    filler_word_count = sum(filler_counter.values())
    
    return {
        'filler_words_found': filler_words_found,
        'filler_word_count': filler_word_count,
        'total_words': total_words,
        'filler_word_ratio': filler_word_count / max(1, total_words)
    }


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


def detect_emotional_tone(prosody: Dict[str, float], 
                         pauses: Dict[str, float],
                         volume: Dict[str, float]) -> Dict[str, Any]:
    """Detect emotional tone using a combination of audio features.
    
    Args:
        prosody: Dictionary containing pitch and tempo features
        pauses: Dictionary containing pause-related metrics
        volume: Dictionary containing volume statistics
        
    Returns:
        Dictionary with emotional tone analysis
    """
    # Extract features
    pitch_var = prosody.get("pitch_variance", 0)
    energy = prosody.get("energy", 0)
    tempo = prosody.get("tempo", 0)
    pause_density = pauses.get("pause_density", 0)
    volume_var = volume.get("volume_variation", 0)
    
    # Calculate base emotion scores
    emotion_scores = {
        'excited': (
            0.4 * min(1.0, pitch_var / 500000.0) +
            0.3 * min(1.0, energy * 8) +
            0.2 * min(1.0, tempo / 180) +
            0.1 * min(1.0, volume_var * 2)
        ),
        'happy': (
            0.3 * min(1.0, pitch_var / 400000.0) +
            0.3 * min(1.0, energy * 7) +
            0.2 * min(1.0, tempo / 160) +
            0.2 * min(1.0, volume_var * 1.5)
        ),
        'confident': (
            0.4 * min(1.0, energy * 6) +
            0.3 * min(1.0, pitch_var / 300000.0) +
            0.2 * (1.0 - min(1.0, pause_density * 1.5)) +
            0.1 * min(1.0, tempo / 150)
        ),
        'neutral': (
            0.6 * (1.0 - min(1.0, abs(energy - 0.5) * 2)) +
            0.2 * (1.0 - min(1.0, pause_density)) +
            0.2 * (1.0 - min(1.0, abs(tempo - 120) / 120))
        ),
        'calm': (
            0.5 * (1.0 - min(1.0, energy * 1.5)) +
            0.3 * (1.0 - min(1.0, pitch_var / 200000.0)) +
            0.2 * (1.0 - min(1.0, tempo / 200))
        ),
        'uncertain': (
            0.5 * min(1.0, pause_density * 2) +
            0.3 * (1.0 - min(1.0, energy * 1.2)) +
            0.2 * min(1.0, abs(tempo - 100) / 100)
        ),
        'nervous': (
            0.6 * min(1.0, pause_density * 2.5) +
            0.2 * (1.0 - min(1.0, energy * 1.5)) +
            0.2 * min(1.0, pitch_var / 100000.0)
        ),
        'monotone': (
            0.7 * (1.0 - min(1.0, pitch_var / 100000.0)) +
            0.3 * (1.0 - min(1.0, volume_var * 1.5))
        ),
        'frustrated': (
            0.4 * min(1.0, energy * 1.8) +
            0.3 * min(1.0, pitch_var / 250000.0) +
            0.2 * min(1.0, pause_density * 1.2) +
            0.1 * min(1.0, tempo / 140)
        )
    }
    
    # Normalize scores to sum to 1
    total = sum(emotion_scores.values())
    if total > 0:
        emotion_scores = {k: v/total for k, v in emotion_scores.items()}
    
    # Get primary emotion
    primary_emotion = max(emotion_scores, key=emotion_scores.get)
    confidence = emotion_scores[primary_emotion]
    
    # Group similar emotions for higher-level analysis
    emotion_groups = {
        'positive': ['excited', 'happy', 'confident'],
        'neutral': ['neutral', 'calm'],
        'negative': ['uncertain', 'nervous', 'monotone', 'frustrated']
    }
    
    # Calculate group scores
    group_scores = {}
    for group, emotions in emotion_groups.items():
        group_scores[group] = sum(emotion_scores[e] for e in emotions)
    
    # Get dominant group
    dominant_group = max(group_scores, key=group_scores.get)
    
    return {
        'primary_emotion': primary_emotion,
        'emotion_confidence': float(confidence),
        'emotion_scores': {k: float(v) for k, v in emotion_scores.items()},
        'emotion_group': dominant_group,
        'group_scores': {k: float(v) for k, v in group_scores.items()}
    }


def detect_negative_behaviors(prosody: Dict[str, float], 
                            pauses: Dict[str, float],
                            audio_path: str,
                            language: str = 'en') -> Dict[str, Any]:
    """Detect negative speaking behaviors.
    
    Args:
        prosody: Dictionary containing pitch and tempo features
        pauses: Dictionary containing pause-related metrics
        audio_path: Path to the audio file for transcription
        language: Language code for transcription and filler word detection
        
    Returns:
        Dictionary with negative behavior analysis
    """
    # Calculate monotony score
    pitch_var = prosody.get("pitch_variance", 0)
    f0_range = prosody.get("f0_range", 0)
    pause_density = pauses.get("pause_density", 0)
    speaking_rate = prosody.get("speaking_rate", 0)
    
    # Monotony: based on pitch variance and range
    monotone = float(max(0.0, 1.0 - pitch_var / 300000.0) * (1.0 - min(1.0, f0_range / 500)))
    
    # Hesitation: based on pause density and speaking rate
    hesitation = float(min(1.0, (pause_density * 2.5) * (1.0 + max(0, 100 - speaking_rate) / 100)))
    
    # Transcribe and detect filler words
    transcription = _transcribe_audio(audio_path, language=language)
    filler_analysis = detect_filler_words(transcription, language=language)
    
    # Calculate overall disfluency score (0-1)
    disfluency = min(1.0, 
        0.4 * min(1.0, filler_analysis.get('filler_word_ratio', 0) * 10) +
        0.3 * hesitation +
        0.2 * monotone +
        0.1 * (1.0 - min(1.0, speaking_rate / 150))
    )
    
    return {
        "monotone_score": monotone,
        "hesitation_index": hesitation,
        "filler_words": filler_analysis.get('filler_words_found', []),
        "filler_word_count": filler_analysis.get('filler_word_count', 0),
        "total_words": filler_analysis.get('total_words', 0),
        "filler_word_ratio": filler_analysis.get('filler_word_ratio', 0),
        "disfluency_score": disfluency,
        "transcription": transcription if transcription else ""
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


def analyze_voice(path: str, language: str = 'en') -> Dict[str, Any]:
    """Analyze voice characteristics from an audio file.
    
    Args:
        path: Path to the audio file
        language: Language code for transcription and analysis
        
    Returns:
        Dictionary containing comprehensive voice analysis
    """
    # Basic audio features
    prosody = compute_basic_prosody(path)
    pauses = compute_pauses_and_pace(path)
    volume = compute_volume_stats(path)
    
    # Advanced analysis
    tone = detect_emotional_tone(prosody, pauses, volume)
    neg = detect_negative_behaviors(prosody, pauses, path, language)
    delivery = compute_delivery_score(prosody, pauses, tone, neg)
    
    # Calculate speaking statistics
    duration = prosody.get('duration_sec', 0)
    speaking_duration = duration * (1 - pauses.get('pause_density', 0))
    words_spoken = neg.get('total_words', 0)
    
    # Calculate words per minute (WPM)
    wpm = (words_spoken / speaking_duration * 60) if speaking_duration > 0 else 0
    
    # Calculate articulation rate (syllables per second)
    # Estimate syllables as words * 1.5 (approximation)
    syllables = words_spoken * 1.5
    articulation_rate = syllables / speaking_duration if speaking_duration > 0 else 0
    
    # Calculate speech clarity score (0-100)
    clarity_score = max(0, min(100, 
        100 - (
            30 * neg.get('disfluency_score', 0) +
            20 * (1 - min(1.0, prosody.get('pitch_variance', 0) / 300000.0)) +
            10 * (1 - min(1.0, wpm / 180)) +
            10 * (1 - min(1.0, articulation_rate / 4.5)) +
            10 * (1 - min(1.0, volume.get('volume_variation', 0) * 2)) +
            20 * (1 - tone.get('emotion_confidence', 0.5) * 2)
        )
    ))
    
    return {
        "prosody": prosody,
        "pace_pauses": pauses,
        "volume": volume,
        "emotional_tone": tone,
        "negative_behaviors": neg,
        "delivery_score": delivery,
        "speaking_stats": {
            "duration_sec": duration,
            "speaking_duration_sec": speaking_duration,
            "words_spoken": words_spoken,
            "words_per_minute": wpm,
            "articulation_rate": articulation_rate,
            "speech_clarity_score": clarity_score,
            "pauses_per_minute": pauses.get('pause_count', 0) / (duration / 60) if duration > 0 else 0
        },
        "language": language,
        "timestamp": datetime.datetime.now().isoformat()
    }
