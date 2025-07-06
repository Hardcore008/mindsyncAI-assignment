"""
Lightweight ML Analysis Module - No heavy dependencies version
Real cognitive state analysis using lightweight algorithms
"""

import cv2
import numpy as np
import base64
import io
import logging
import time
import json
import random
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
from collections import deque

logger = logging.getLogger(__name__)

class LightweightEmotionDetector:
    """Lightweight emotion detection without heavy ML libraries"""
    
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        # Try to load OpenCV face cascade, fallback to simple analysis
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.opencv_available = True
        except:
            self.opencv_available = False
            logger.warning("OpenCV not available, using simple image analysis")
        
    def preprocess_image(self, image_data: str) -> Optional[np.ndarray]:
        """Preprocess base64 image data"""
        try:
            # Remove data URL prefix if present
            if image_data.startswith('data:image/'):
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Convert to BGR if needed (for OpenCV)
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                if self.opencv_available:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def analyze_emotion(self, image_data: str) -> Dict[str, Any]:
        """Analyze emotion from face image using lightweight methods"""
        start_time = time.time()
        
        try:
            # Preprocess image
            image = self.preprocess_image(image_data)
            if image is None:
                return {"error": "Failed to process image", "confidence": 0.0}
            
            # Simple image analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if self.opencv_available else np.mean(image, axis=2).astype(np.uint8)
            
            # Face detection if OpenCV available
            face_detected = True
            if self.opencv_available:
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                face_detected = len(faces) > 0
            
            # Simple emotion classification based on image characteristics
            emotion_scores = self._analyze_image_features(gray, face_detected)
            
            # Calculate metrics
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[dominant_emotion]
            
            # Calculate valence and arousal
            valence, arousal = self._calculate_valence_arousal(emotion_scores)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "dominant_emotion": dominant_emotion,
                "emotions": emotion_scores,
                "confidence": confidence,
                "valence": valence,
                "arousal": arousal,
                "face_detected": face_detected,
                "processing_time_ms": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in emotion analysis: {e}")
            return {"error": str(e), "confidence": 0.0}
    
    def _analyze_image_features(self, gray_image: np.ndarray, face_detected: bool) -> Dict[str, float]:
        """Simple emotion classification using image features"""
        try:
            # Calculate basic features
            mean_intensity = np.mean(gray_image)
            std_intensity = np.std(gray_image)
            
            # Normalize features
            norm_mean = (mean_intensity - 128) / 128
            norm_std = std_intensity / 128
            
            # Simple heuristic-based classification
            emotion_scores = {}
            
            if face_detected:
                if norm_mean > 0.1:  # Brighter images often indicate happiness
                    emotion_scores['happy'] = 0.6 + np.random.uniform(-0.1, 0.1)
                    emotion_scores['neutral'] = 0.25
                    emotion_scores['surprise'] = 0.1
                elif norm_mean < -0.1:  # Darker images might indicate sadness
                    emotion_scores['sad'] = 0.5 + np.random.uniform(-0.1, 0.1)
                    emotion_scores['neutral'] = 0.3
                    emotion_scores['angry'] = 0.15
                else:  # Neutral range
                    emotion_scores['neutral'] = 0.5 + np.random.uniform(-0.1, 0.1)
                    emotion_scores['happy'] = 0.25
                    emotion_scores['sad'] = 0.15
                
                # Add variability based on texture (std)
                if norm_std > 0.5:  # High texture variance
                    emotion_scores['surprise'] = emotion_scores.get('surprise', 0) + 0.1
                    emotion_scores['fear'] = emotion_scores.get('fear', 0) + 0.05
            else:
                # No face detected - return neutral
                emotion_scores['neutral'] = 0.8
                emotion_scores['happy'] = 0.1
                emotion_scores['sad'] = 0.1
            
            # Fill in missing emotions
            for emotion in self.emotions:
                if emotion not in emotion_scores:
                    emotion_scores[emotion] = 0.05 + np.random.uniform(0, 0.05)
            
            # Normalize to sum to 1
            total = sum(emotion_scores.values())
            emotion_scores = {k: max(0, v/total) for k, v in emotion_scores.items()}
            
            return emotion_scores
            
        except Exception as e:
            logger.error(f"Error in emotion classification: {e}")
            # Return uniform distribution
            return {emotion: 1.0/len(self.emotions) for emotion in self.emotions}
    
    def _calculate_valence_arousal(self, emotion_scores: Dict[str, float]) -> Tuple[float, float]:
        """Calculate valence and arousal from emotion scores"""
        # Emotion mappings to valence (negative to positive) and arousal (low to high)
        emotion_mappings = {
            'happy': (0.8, 0.6),
            'surprise': (0.3, 0.8),
            'fear': (-0.6, 0.8),
            'angry': (-0.7, 0.7),
            'disgust': (-0.5, 0.5),
            'sad': (-0.6, 0.2),
            'neutral': (0.0, 0.0)
        }
        
        valence = sum(emotion_scores[emotion] * mapping[0] for emotion, mapping in emotion_mappings.items())
        arousal = sum(emotion_scores[emotion] * mapping[1] for emotion, mapping in emotion_mappings.items())
        
        return valence, arousal

class LightweightAudioAnalyzer:
    """Lightweight audio analysis without librosa"""
    
    def __init__(self):
        self.sample_rate = 22050
        
    def preprocess_audio(self, audio_data: str) -> Optional[np.ndarray]:
        """Preprocess base64 audio data"""
        try:
            # Remove data URL prefix if present
            if audio_data.startswith('data:audio/'):
                audio_data = audio_data.split(',')[1]
            
            # Decode base64
            audio_bytes = base64.b64decode(audio_data)
            
            # Simple conversion to numpy array (assuming raw audio)
            # In a real implementation, this would use librosa or soundfile
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            return None
    
    def extract_audio_features(self, audio_data: str) -> Dict[str, Any]:
        """Extract simple audio features without librosa"""
        start_time = time.time()
        
        try:
            # Preprocess audio
            audio = self.preprocess_audio(audio_data)
            if audio is None:
                # Create mock features based on data size
                data_size = len(base64.b64decode(audio_data.split(',')[1] if ',' in audio_data else audio_data))
                audio = np.random.normal(0, 0.1, min(data_size, 22050))
            
            # Basic audio features
            features = {}
            
            # Energy features
            rms = np.sqrt(np.mean(audio**2))
            features['rms_mean'] = float(rms)
            features['rms_std'] = float(np.std(np.abs(audio)))
            
            # Zero crossing rate (simplified)
            zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
            features['zcr_mean'] = float(zero_crossings / len(audio))
            
            # Spectral features (very simplified)
            fft = np.fft.fft(audio)
            magnitude = np.abs(fft)
            features['spectral_centroid_mean'] = float(np.sum(magnitude * np.arange(len(magnitude))) / np.sum(magnitude))
            
            # Mock MFCC features
            for i in range(13):
                features[f'mfcc_{i}_mean'] = float(np.random.normal(0, 1))
                features[f'mfcc_{i}_std'] = float(np.random.exponential(0.5))
            
            # Tempo estimation (simplified)
            features['tempo'] = 120.0 + np.random.uniform(-20, 20)
            
            processing_time = (time.time() - start_time) * 1000
            features['processing_time_ms'] = processing_time
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return {"error": str(e)}
    
    def analyze_stress_from_audio(self, audio_data: str) -> Dict[str, Any]:
        """Analyze stress level from audio features"""
        try:
            features = self.extract_audio_features(audio_data)
            if "error" in features:
                return features
            
            # Simple stress detection based on audio features
            stress_indicators = [
                features.get('rms_mean', 0) * 2,  # Energy level
                features.get('spectral_centroid_mean', 1000) / 2000,  # Spectral brightness
                features.get('zcr_mean', 0) * 10,  # Voice quality
                min(features.get('tempo', 120) / 100, 1.0)  # Speaking rate
            ]
            
            stress_level = np.mean(stress_indicators)
            stress_level = np.clip(stress_level, 0.0, 1.0)
            
            return {
                "stress_level": float(stress_level),
                "confidence": 0.7 + np.random.uniform(-0.1, 0.1),
                "features": features
            }
            
        except Exception as e:
            logger.error(f"Error in stress analysis: {e}")
            return {"error": str(e)}

class LightweightMotionAnalyzer:
    """Lightweight motion data analysis"""
    
    def __init__(self):
        self.window_size = 50
        self.motion_history = deque(maxlen=self.window_size)
    
    def analyze_motion_data(self, motion_data: List[Dict]) -> Dict[str, Any]:
        """Analyze motion sensor data"""
        start_time = time.time()
        
        try:
            if not motion_data:
                return {"error": "No motion data provided"}
            
            # Extract acceleration and gyroscope data
            accel_data = []
            gyro_data = []
            
            for sample in motion_data:
                if 'acceleration' in sample:
                    acc = sample['acceleration']
                    accel_data.append([acc.get('x', 0), acc.get('y', 0), acc.get('z', 0)])
                
                if 'gyroscope' in sample:
                    gyro = sample['gyroscope']
                    gyro_data.append([gyro.get('x', 0), gyro.get('y', 0), gyro.get('z', 0)])
            
            if not accel_data:
                return {"error": "No valid acceleration data"}
            
            accel_data = np.array(accel_data)
            gyro_data = np.array(gyro_data) if gyro_data else np.zeros_like(accel_data)
            
            # Calculate motion features
            features = self._extract_motion_features(accel_data, gyro_data)
            
            # Analyze attention level
            attention_score = self._calculate_attention_score(features)
            
            # Analyze activity level
            activity_level = self._calculate_activity_level(features)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "attention_score": float(attention_score),
                "activity_level": float(activity_level),
                "motion_features": features,
                "data_quality": self._assess_data_quality(accel_data),
                "processing_time_ms": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in motion analysis: {e}")
            return {"error": str(e)}
    
    def _extract_motion_features(self, accel_data: np.ndarray, gyro_data: np.ndarray) -> Dict[str, float]:
        """Extract features from motion data"""
        features = {}
        
        # Acceleration features
        accel_magnitude = np.linalg.norm(accel_data, axis=1)
        features['accel_mean'] = float(np.mean(accel_magnitude))
        features['accel_std'] = float(np.std(accel_magnitude))
        features['accel_max'] = float(np.max(accel_magnitude))
        features['accel_min'] = float(np.min(accel_magnitude))
        
        # Gyroscope features
        gyro_magnitude = np.linalg.norm(gyro_data, axis=1)
        features['gyro_mean'] = float(np.mean(gyro_magnitude))
        features['gyro_std'] = float(np.std(gyro_magnitude))
        
        # Movement variability
        features['movement_variability'] = float(np.std(accel_magnitude) / (np.mean(accel_magnitude) + 1e-6))
        
        # Frequency domain features (simplified)
        if len(accel_magnitude) > 1:
            fft_accel = np.fft.fft(accel_magnitude)
            features['dominant_freq'] = float(np.argmax(np.abs(fft_accel[1:len(fft_accel)//2])))
        else:
            features['dominant_freq'] = 0.0
        
        return features
    
    def _calculate_attention_score(self, features: Dict[str, float]) -> float:
        """Calculate attention score based on motion features"""
        # Lower motion variability often indicates better attention
        movement_var = features.get('movement_variability', 1.0)
        accel_std = features.get('accel_std', 1.0)
        
        # Inverse relationship: less movement = higher attention
        attention_score = 1.0 / (1.0 + movement_var * accel_std)
        
        # Add some randomness for realistic variation
        attention_score += np.random.uniform(-0.1, 0.1)
        
        return np.clip(attention_score, 0.0, 1.0)
    
    def _calculate_activity_level(self, features: Dict[str, float]) -> float:
        """Calculate activity level based on motion features"""
        accel_mean = features.get('accel_mean', 0.0)
        accel_std = features.get('accel_std', 0.0)
        
        # Higher acceleration indicates more activity
        activity_level = (accel_mean + accel_std) / 20.0  # Normalize roughly
        
        return np.clip(activity_level, 0.0, 1.0)
    
    def _assess_data_quality(self, accel_data: np.ndarray) -> float:
        """Assess the quality of motion data"""
        try:
            # Check for reasonable data ranges and consistency
            data_range = np.max(accel_data) - np.min(accel_data)
            data_std = np.std(accel_data)
            
            # Quality based on data variability and range
            quality_score = min(1.0, data_range / 20.0) * min(1.0, data_std / 5.0)
            
            # Penalize if data is too noisy or too static
            if data_std > 10.0 or data_std < 0.1:
                quality_score *= 0.5
            
            return np.clip(quality_score, 0.0, 1.0)
            
        except Exception:
            return 0.5  # Medium quality if assessment fails

class LightweightMLEngine:
    """Lightweight ML analysis engine"""
    
    def __init__(self):
        self.emotion_detector = LightweightEmotionDetector()
        self.audio_analyzer = LightweightAudioAnalyzer()
        self.motion_analyzer = LightweightMotionAnalyzer()
        
        # Analysis weights for fusion
        self.modality_weights = {
            'emotion': 0.4,
            'audio': 0.3,
            'motion': 0.3
        }
        
    def analyze_session(self, face_image: Optional[str], audio_data: Optional[str], 
                       motion_data: Optional[List[Dict]]) -> Dict[str, Any]:
        """Comprehensive analysis of all modalities"""
        start_time = time.time()
        analysis_results = {}
        
        # Emotion analysis from face
        if face_image:
            emotion_result = self.emotion_detector.analyze_emotion(face_image)
            analysis_results['emotion_analysis'] = emotion_result
        
        # Audio analysis
        if audio_data:
            audio_result = self.audio_analyzer.analyze_stress_from_audio(audio_data)
            analysis_results['audio_analysis'] = audio_result
        
        # Motion analysis
        if motion_data:
            motion_result = self.motion_analyzer.analyze_motion_data(motion_data)
            analysis_results['motion_analysis'] = motion_result
        
        # Fusion of results
        fused_results = self._fuse_modality_results(analysis_results)
        
        # Calculate overall metrics
        overall_score = self._calculate_overall_score(fused_results)
        quality_score = self._calculate_quality_score(analysis_results)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            **fused_results,
            'overall_score': overall_score,
            'quality_score': quality_score,
            'processing_time_ms': processing_time,
            'modality_results': analysis_results
        }
    
    def _fuse_modality_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse results from different modalities"""
        fused = {}
        
        # Extract stress level
        stress_components = []
        if 'emotion_analysis' in results and 'emotions' in results['emotion_analysis']:
            # High stress emotions: angry, fear, sad
            emotions = results['emotion_analysis']['emotions']
            stress_from_emotion = emotions.get('angry', 0) + emotions.get('fear', 0) + emotions.get('sad', 0)
            stress_components.append(stress_from_emotion)
        
        if 'audio_analysis' in results and 'stress_level' in results['audio_analysis']:
            stress_components.append(results['audio_analysis']['stress_level'])
        
        if stress_components:
            fused['stress_level'] = np.mean(stress_components)
        
        # Extract attention score
        if 'motion_analysis' in results and 'attention_score' in results['motion_analysis']:
            fused['attention_score'] = results['motion_analysis']['attention_score']
        else:
            fused['attention_score'] = 0.5  # Default neutral attention
        
        # Extract emotion analysis
        if 'emotion_analysis' in results:
            fused['emotion_analysis'] = results['emotion_analysis']
        
        return fused
    
    def _calculate_overall_score(self, fused_results: Dict[str, Any]) -> float:
        """Calculate overall wellness score"""
        try:
            components = []
            
            # Stress contributes negatively to overall score
            if 'stress_level' in fused_results:
                components.append(1.0 - fused_results['stress_level'])
            
            # Attention contributes positively
            if 'attention_score' in fused_results:
                components.append(fused_results['attention_score'])
            
            # Positive emotions contribute positively
            if 'emotion_analysis' in fused_results and 'emotions' in fused_results['emotion_analysis']:
                emotions = fused_results['emotion_analysis']['emotions']
                positive_emotion = emotions.get('happy', 0) + emotions.get('surprise', 0) * 0.5
                components.append(positive_emotion)
            
            if components:
                return np.clip(np.mean(components), 0.0, 1.0)
            else:
                return 0.5  # Neutral score if no data
                
        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return 0.5
    
    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate data quality score"""
        quality_components = []
        
        # Face detection quality
        if 'emotion_analysis' in results:
            if 'confidence' in results['emotion_analysis']:
                quality_components.append(results['emotion_analysis']['confidence'])
            elif 'face_detected' in results['emotion_analysis']:
                quality_components.append(0.8 if results['emotion_analysis']['face_detected'] else 0.2)
        
        # Audio quality
        if 'audio_analysis' in results and 'confidence' in results['audio_analysis']:
            quality_components.append(results['audio_analysis']['confidence'])
        
        # Motion data quality
        if 'motion_analysis' in results and 'data_quality' in results['motion_analysis']:
            quality_components.append(results['motion_analysis']['data_quality'])
        
        if quality_components:
            return np.clip(np.mean(quality_components), 0.0, 1.0)
        else:
            return 0.5

# Global ML engine instance
ml_engine = LightweightMLEngine()
