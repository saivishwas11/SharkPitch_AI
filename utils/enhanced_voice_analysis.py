import torch
import torchaudio
import numpy as np
import datetime
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VoiceAnalysisConfig:
    """Configuration for voice analysis models and parameters"""
    emotion_model_name: str = "superb/hubert-large-superb-er"  # Pre-trained emotion recognition model
    min_audio_length: float = 1.0  # Minimum audio length in seconds
    max_audio_length: float = 30.0  # Maximum audio length for emotion analysis
    target_sr: int = 16000  # Target sample rate for audio processing
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class EmotionRecognizer:
    """Advanced emotion recognition using pre-trained models"""
    
    def __init__(self, config: VoiceAnalysisConfig = None):
        self.config = config or VoiceAnalysisConfig()
        self.device = torch.device(self.config.device)
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained emotion recognition models"""
        logger.info("Loading emotion recognition model...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.config.emotion_model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.config.emotion_model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Emotion labels from the pre-trained model
        self.emotion_labels = [
            'angry', 'disgust', 'fear', 'happy', 
            'neutral', 'sad', 'surprise', 'ps', 'unknown'
        ]
    
    def preprocess_audio(self, audio_path: str) -> Tuple[Optional[torch.Tensor], int]:
        """Load and preprocess audio file"""
        try:
            # Load audio file
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if needed
            if len(waveform.shape) > 1 and waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Resample if needed
            if sample_rate != self.config.target_sr:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.config.target_sr
                )
                waveform = resampler(waveform)
                
            return waveform.squeeze(), self.config.target_sr
            
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            return None, self.config.target_sr
    
    def recognize_emotion(self, audio_path: str) -> Dict[str, Any]:
        """Recognize emotions from audio using pre-trained model"""
        try:
            # Load and preprocess audio
            waveform, sr = self.preprocess_audio(audio_path)
            if waveform is None or len(waveform) < int(sr * self.config.min_audio_length):
                return {"error": "Audio too short or invalid"}
                
            # Limit audio length
            max_samples = int(sr * self.config.max_audio_length)
            if len(waveform) > max_samples:
                waveform = waveform[:max_samples]
                
            # Extract features
            inputs = self.feature_extractor(
                waveform.numpy(),
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True
            ).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                
            # Get emotion probabilities
            emotion_probs = {emotion: float(prob) for emotion, prob in zip(self.emotion_labels, probs)}
            
            # Get dominant emotion
            dominant_emotion = self.emotion_labels[probs.argmax()]
            confidence = float(probs.max())
            
            return {
                "dominant_emotion": dominant_emotion,
                "confidence": confidence,
                "emotion_scores": emotion_probs,
                "emotion_group": self._get_emotion_group(dominant_emotion)
            }
            
        except Exception as e:
            logger.error(f"Error in emotion recognition: {e}")
            return {"error": str(e)}
    
    def _get_emotion_group(self, emotion: str) -> str:
        """Map specific emotions to broader emotion groups"""
        emotion_groups = {
            'positive': ['happy', 'surprise'],
            'neutral': ['neutral'],
            'negative': ['angry', 'sad', 'fear', 'disgust'],
            'uncertain': ['ps', 'unknown']
        }
        
        for group, emotions in emotion_groups.items():
            if emotion in emotions:
                return group
        return 'neutral'

class DeliveryScorer:
    """Advanced delivery scoring system"""
    
    def __init__(self, config: VoiceAnalysisConfig = None):
        self.config = config or VoiceAnalysisConfig()
        
    def calculate_score(self, 
                       prosody: Dict[str, float],
                       emotion_results: Dict[str, Any],
                       negative_behaviors: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive delivery score"""
        # Base score components (0-100)
        base_score = 70.0  # Start with a neutral base
        
        # 1. Prosody factors (30%)
        pitch_score = min(1.0, prosody.get("pitch_variance", 0) / 300000.0) * 100
        energy_score = min(1.0, prosody.get("energy", 0) * 10) * 100
        tempo_score = min(1.0, prosody.get("tempo", 0) / 180.0) * 100
        prosody_score = (pitch_score * 0.4 + energy_score * 0.4 + tempo_score * 0.2) * 0.3
        
        # 2. Emotion factors (30%)
        emotion_scores = emotion_results.get("emotion_scores", {})
        positive_emotions = emotion_scores.get("happy", 0) + emotion_scores.get("surprise", 0)
        negative_emotions = (emotion_scores.get("angry", 0) + 
                            emotion_scores.get("sad", 0) + 
                            emotion_scores.get("fear", 0) + 
                            emotion_scores.get("disgust", 0))
        
        emotion_score = (positive_emotions - negative_emotions * 0.5 + 1) * 50  # Scale to 0-100
        emotion_score = max(0, min(100, emotion_score)) * 0.3
        
        # 3. Negative behavior penalties (20%)
        disfluency_penalty = negative_behaviors.get("disfluency_score", 0) * 100 * 0.2
        monotone_penalty = negative_behaviors.get("monotone_score", 0) * 100 * 0.1
        hesitation_penalty = negative_behaviors.get("hesitation_index", 0) * 100 * 0.1
        filler_penalty = min(20, negative_behaviors.get("filler_word_ratio", 0) * 100)  # Max 20% penalty
        
        negative_score = (disfluency_penalty + monotone_penalty + 
                         hesitation_penalty + filler_penalty) * 0.2
        
        # 4. Speaking rate and clarity (20%)
        speaking_rate = prosody.get("speaking_rate", 0)
        rate_score = 100 * (1 - abs(speaking_rate - 150) / 150)  # Optimal around 150 WPM
        clarity_score = 100 * (1 - negative_behaviors.get("disfluency_score", 0))
        clarity_component = (rate_score * 0.5 + clarity_score * 0.5) * 0.2
        
        # Calculate final score
        final_score = (
            base_score * 0.4 +  # Base score (40%)
            prosody_score +      # Prosody (30%)
            emotion_score -      # Emotion (30%)
            negative_score +     # Negative behaviors (20%)
            clarity_component    # Clarity (20%)
        )
        
        # Ensure score is within 0-100 range
        final_score = max(0, min(100, final_score))
        
        return {
            "overall_score": final_score,
            "score_components": {
                "base_score": base_score,
                "prosody_score": prosody_score,
                "emotion_score": emotion_score,
                "negative_penalty": negative_score,
                "clarity_component": clarity_component
            },
            "improvement_areas": self._get_improvement_areas(
                prosody, emotion_results, negative_behaviors
            )
        }
    
    def _get_improvement_areas(self, 
                             prosody: Dict[str, float], 
                             emotion_results: Dict[str, Any], 
                             negative_behaviors: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate personalized improvement suggestions"""
        areas = []
        
        # Check for monotone speech
        if negative_behaviors.get("monotone_score", 0) > 0.6:
            areas.append({
                "area": "Vocal Variety",
                "score": negative_behaviors["monotone_score"],
                "suggestion": "Try varying your pitch and volume more to sound more engaging.",
                "priority": "high"
            })
            
        # Check for filler words
        if negative_behaviors.get("filler_word_ratio", 0) > 0.1:  # More than 10% filler words
            areas.append({
                "area": "Filler Words",
                "score": negative_behaviors["filler_word_ratio"],
                "suggestion": "Practice pausing instead of using filler words like 'um' and 'uh'.",
                "priority": "medium"
            })
            
        # Check for speaking rate
        speaking_rate = prosody.get("speaking_rate", 0)
        if speaking_rate < 120:
            areas.append({
                "area": "Speaking Rate",
                "score": speaking_rate / 120,
                "suggestion": "Try speaking slightly faster to maintain audience engagement.",
                "priority": "medium"
            })
        elif speaking_rate > 180:
            areas.append({
                "area": "Speaking Rate",
                "score": 180 / speaking_rate,
                "suggestion": "Try speaking slightly slower to improve clarity.",
                "priority": "medium"
            })
            
        # Check for emotional tone
        emotion_scores = emotion_results.get("emotion_scores", {})
        if emotion_scores.get("neutral", 0) > 0.7:
            areas.append({
                "area": "Emotional Expression",
                "score": emotion_scores["neutral"],
                "suggestion": "Try to express more emotion in your voice to sound more engaging.",
                "priority": "high"
            })
            
        return areas
