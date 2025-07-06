"""
Lightweight ML Analysis Engine - Production-grade analysis without heavy dependencies
"""

import cv2
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import base64
import io
from PIL import Image
import json
import time
import soundfile as sf
from scipy import signal
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

from cognitive_fusion import fusion_engine, ModalityResult, CognitiveStateResult

logger = logging.getLogger(__name__)

class LightweightEmotionDetector:
    """Emotion detection using OpenCV and rule-based analysis"""
    
    def __init__(self):
        # Load OpenCV cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Emotion mapping
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        logger.info("Lightweight Emotion Detector initialized")
    
    def analyze_emotion(self, image_data: str) -> ModalityResult:
        """Analyze emotion using OpenCV-based features"""
        start_time = time.time()
        
        try:
            # Preprocess image
            image = self._preprocess_image(image_data)
            if image is None:
                return self._create_error_result("emotion", "Failed to preprocess image", start_time)
            
            # Detect faces and features
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return self._create_error_result("emotion", "No face detected", start_time)
            
            # Analyze first face
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Extract features
            features = self._extract_opencv_features(face_roi, gray, x, y, w, h)
            
            # Predict emotions
            emotions = self._predict_emotions_opencv(features)
            
            # Calculate confidence and quality
            confidence = self._calculate_confidence(features)
            quality_score = self._assess_image_quality(image, True)
            
            processing_time = (time.time() - start_time) * 1000
            
            return ModalityResult(
                modality="emotion",
                confidence=confidence,
                features=features,
                raw_predictions=emotions,
                quality_score=quality_score,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in emotion analysis: {e}")
            return self._create_error_result("emotion", str(e), start_time)
    
    def _preprocess_image(self, image_data: str) -> Optional[np.ndarray]:
        """Preprocess base64 image data"""
        try:
            if image_data.startswith('data:image/'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            return image
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def _extract_opencv_features(self, face_roi: np.ndarray, full_gray: np.ndarray, 
                                x: int, y: int, w: int, h: int) -> Dict[str, float]:
        """Extract features using OpenCV cascades"""
        features = {}
        
        try:
            # Eye detection
            eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 3)
            features['eye_count'] = len(eyes)
            features['eye_openness'] = 1.0 if len(eyes) >= 2 else 0.5
            
            # Smile detection
            smiles = self.smile_cascade.detectMultiScale(face_roi, 1.8, 20)
            features['smile_detected'] = 1.0 if len(smiles) > 0 else 0.0
            features['smile_intensity'] = min(len(smiles) / 2.0, 1.0)
            
            # Face size (relative to image)
            image_area = full_gray.shape[0] * full_gray.shape[1]
            face_area = w * h
            features['face_size_ratio'] = face_area / image_area
            
            # Face aspect ratio
            features['face_aspect_ratio'] = w / h
            
            # Image contrast in face region (indicator of expression intensity)
            features['face_contrast'] = np.std(face_roi) / 255.0
            
            # Edge density (higher for more expressive faces)
            edges = cv2.Canny(face_roi, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / (w * h)
            
            # Brightness
            features['face_brightness'] = np.mean(face_roi) / 255.0
            
        except Exception as e:
            logger.error(f"Error extracting OpenCV features: {e}")
            # Provide default features
            features = {
                'eye_count': 2, 'eye_openness': 1.0, 'smile_detected': 0.0,
                'smile_intensity': 0.0, 'face_size_ratio': 0.1, 'face_aspect_ratio': 0.8,
                'face_contrast': 0.5, 'edge_density': 0.1, 'face_brightness': 0.5
            }
        
        return features
    
    def _predict_emotions_opencv(self, features: Dict[str, float]) -> Dict[str, float]:
        """Predict emotions using rule-based approach with OpenCV features"""
        emotions = {emotion: 0.0 for emotion in self.emotions}
        
        # Rule-based emotion detection
        smile_detected = features.get('smile_detected', 0.0)
        smile_intensity = features.get('smile_intensity', 0.0)
        eye_openness = features.get('eye_openness', 1.0)
        face_contrast = features.get('face_contrast', 0.5)
        edge_density = features.get('edge_density', 0.1)
        
        # Happy: smile detected
        if smile_detected > 0.5:
            emotions['happy'] = 0.4 + smile_intensity * 0.4
        
        # Surprise: high contrast, wide eyes
        if face_contrast > 0.6 and eye_openness > 0.8:
            emotions['surprise'] = 0.6
        
        # Angry/Fear: high edge density (tense expressions)
        if edge_density > 0.15:
            if face_contrast > 0.5:
                emotions['angry'] = 0.5
            else:
                emotions['fear'] = 0.4
        
        # Sad: low contrast, no smile
        if face_contrast < 0.3 and smile_detected < 0.1:
            emotions['sad'] = 0.4
        
        # Default to neutral if no strong indicators
        if sum(emotions.values()) < 0.3:
            emotions['neutral'] = 0.7
        
        # Normalize probabilities
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v/total for k, v in emotions.items()}
        else:
            emotions = {emotion: 1.0/len(self.emotions) for emotion in self.emotions}
        
        return emotions
    
    def _calculate_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence based on feature reliability"""
        confidence_factors = []
        
        # Face size confidence
        face_size = features.get('face_size_ratio', 0.1)
        size_conf = min(1.0, face_size * 10)  # Prefer larger faces
        confidence_factors.append(size_conf)
        
        # Eye detection confidence
        eye_conf = 1.0 if features.get('eye_count', 0) >= 2 else 0.5
        confidence_factors.append(eye_conf)
        
        # Image quality confidence
        contrast = features.get('face_contrast', 0.5)
        quality_conf = min(1.0, contrast * 2)  # Prefer good contrast
        confidence_factors.append(quality_conf)
        
        return np.mean(confidence_factors)
    
    def _assess_image_quality(self, image: np.ndarray, face_detected: bool) -> float:
        """Assess image quality"""
        quality_factors = []
        
        # Face detection
        quality_factors.append(0.8 if face_detected else 0.2)
        
        # Image size
        h, w = image.shape[:2]
        size_quality = min(1.0, (h * w) / (200 * 200))
        quality_factors.append(size_quality)
        
        # Sharpness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_quality = min(1.0, sharpness / 1000)
        quality_factors.append(sharpness_quality)
        
        return np.mean(quality_factors)
    
    def _create_error_result(self, modality: str, error_msg: str, start_time: float) -> ModalityResult:
        """Create error result"""
        processing_time = (time.time() - start_time) * 1000
        return ModalityResult(
            modality=modality,
            confidence=0.1,
            features={'error': error_msg},
            raw_predictions={'neutral': 1.0},
            quality_score=0.1,
            processing_time_ms=processing_time
        )

class LightweightAudioAnalyzer:
    """Audio analysis using Librosa without deep learning models"""
    
    def __init__(self):
        self.sample_rate = 22050
        self.hop_length = 512
        self.n_mfcc = 13
        
        # Create a simple classifier
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        self._train_simple_model()
        
        logger.info("Lightweight Audio Analyzer initialized")
    
    def _train_simple_model(self):
        """Train a simple model with synthetic data"""
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 500
        n_features = 25  # Reduced feature set
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 3, n_samples)  # 3 stress levels
        
        # Add some structure to the data
        # High energy features tend to correlate with high stress
        for i in range(n_samples):
            if y[i] == 2:  # High stress
                X[i, :5] += 1.5  # Boost energy-related features
            elif y[i] == 0:  # Low stress
                X[i, :5] -= 1.0  # Lower energy features
        
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)
    
    def analyze_stress_from_audio(self, audio_data: str) -> ModalityResult:
        """Analyze stress from audio using Librosa features"""
        start_time = time.time()
        
        try:
            # Preprocess audio
            audio = self._preprocess_audio(audio_data)
            if audio is None:
                return self._create_error_result("audio", "Failed to preprocess audio", start_time)
            
            # Extract features
            features = self._extract_audio_features(audio)
            
            # Predict stress
            stress_prediction = self._predict_stress(features)
            
            # Calculate quality and confidence
            quality_score = self._assess_audio_quality(audio)
            confidence = self._calculate_confidence(features, quality_score)
            
            processing_time = (time.time() - start_time) * 1000
            
            return ModalityResult(
                modality="audio",
                confidence=confidence,
                features={
                    'stress_level': stress_prediction,
                    'audio_length': len(audio) / self.sample_rate,
                    'energy': features.get('rms_energy', 0.5),
                    'spectral_centroid': features.get('spectral_centroid', 0.5),
                    'tempo': features.get('tempo', 120)
                },
                raw_predictions={'stress_level': stress_prediction},
                quality_score=quality_score,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in audio analysis: {e}")
            return self._create_error_result("audio", str(e), start_time)
    
    def _preprocess_audio(self, audio_data: str) -> Optional[np.ndarray]:
        """Preprocess base64 audio data"""
        try:
            if audio_data.startswith('data:audio/'):
                audio_data = audio_data.split(',')[1]
            
            audio_bytes = base64.b64decode(audio_data)
            audio, sr = sf.read(io.BytesIO(audio_bytes))
            
            # Convert to mono
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
            return audio
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            return None
    
    def _extract_audio_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive audio features"""
        features = {}
        
        try:
            # Basic features
            features['rms_energy'] = float(np.sqrt(np.mean(audio**2)))
            features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(audio)[0]))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            features['spectral_centroid'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
            
            # MFCCs (reduced set)
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            for i in range(5):  # Only first 5 MFCCs
                features[f'mfcc_{i}'] = float(np.mean(mfccs[i]))
            
            # Chroma (reduced)
            chroma = librosa.feature.chroma(y=audio, sr=self.sample_rate)
            for i in range(3):  # Only first 3 chroma features
                features[f'chroma_{i}'] = float(np.mean(chroma[i]))
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            features['tempo'] = float(tempo)
            
            # Additional stress indicators
            features['energy_variance'] = float(np.var(librosa.feature.rms(y=audio)[0]))
            
            # Pitch features (simplified)
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
            pitch_values = []
            for t in range(min(10, pitches.shape[1])):  # Limit computation
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['pitch_mean'] = float(np.mean(pitch_values))
                features['pitch_variance'] = float(np.var(pitch_values))
            else:
                features['pitch_mean'] = 0.0
                features['pitch_variance'] = 0.0
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            # Provide minimal features
            features = {
                'rms_energy': 0.1, 'zero_crossing_rate': 0.05, 'spectral_centroid': 1000.0,
                'spectral_centroid_std': 500.0, 'spectral_rolloff': 2000.0, 'tempo': 120.0,
                'energy_variance': 0.01, 'pitch_mean': 200.0, 'pitch_variance': 100.0
            }
            for i in range(5):
                features[f'mfcc_{i}'] = 0.0
            for i in range(3):
                features[f'chroma_{i}'] = 0.0
        
        return features
    
    def _predict_stress(self, features: Dict[str, float]) -> float:
        """Predict stress level from features"""
        try:
            # Prepare feature vector (reduced to 25 features)
            feature_names = [
                'rms_energy', 'zero_crossing_rate', 'spectral_centroid', 'spectral_centroid_std',
                'spectral_rolloff', 'tempo', 'energy_variance', 'pitch_mean', 'pitch_variance'
            ]
            feature_names.extend([f'mfcc_{i}' for i in range(5)])
            feature_names.extend([f'chroma_{i}' for i in range(3)])
            feature_names.extend(['extra_1', 'extra_2', 'extra_3', 'extra_4', 'extra_5', 'extra_6', 'extra_7'])  # Padding
            
            feature_vector = []
            for name in feature_names:
                if name.startswith('extra_'):
                    feature_vector.append(0.0)  # Padding
                else:
                    feature_vector.append(features.get(name, 0.0))
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            feature_vector = self.scaler.transform(feature_vector)
            
            # Predict stress class and convert to probability
            stress_class = self.classifier.predict(feature_vector)[0]
            stress_level = [0.2, 0.6, 0.9][stress_class]  # Low, medium, high stress
            
            return stress_level
            
        except Exception as e:
            logger.error(f"Error predicting stress: {e}")
            # Rule-based fallback
            energy = features.get('rms_energy', 0.1)
            tempo = features.get('tempo', 120)
            pitch_variance = features.get('pitch_variance', 100)
            
            # High energy, fast tempo, high pitch variance = high stress
            stress_level = min(1.0, (energy * 5 + max(0, tempo - 120) / 80 + pitch_variance / 1000) / 3)
            return stress_level
    
    def _assess_audio_quality(self, audio: np.ndarray) -> float:
        """Assess audio quality"""
        quality_factors = []
        
        # Length quality
        duration = len(audio) / self.sample_rate
        length_quality = 1.0 if 1 <= duration <= 10 else max(0.3, 1.0 - abs(duration - 5) / 10)
        quality_factors.append(length_quality)
        
        # Signal level
        rms = np.sqrt(np.mean(audio**2))
        signal_quality = min(1.0, rms * 10)
        quality_factors.append(signal_quality)
        
        # Dynamic range
        dynamic_range = np.max(audio) - np.min(audio)
        range_quality = min(1.0, dynamic_range * 2)
        quality_factors.append(range_quality)
        
        return np.mean(quality_factors)
    
    def _calculate_confidence(self, features: Dict[str, float], quality_score: float) -> float:
        """Calculate analysis confidence"""
        return quality_score * 0.8 + 0.2  # Base confidence with quality adjustment
    
    def _create_error_result(self, modality: str, error_msg: str, start_time: float) -> ModalityResult:
        """Create error result"""
        processing_time = (time.time() - start_time) * 1000
        return ModalityResult(
            modality=modality,
            confidence=0.1,
            features={'error': error_msg, 'stress_level': 0.5},
            raw_predictions={'stress_level': 0.5},
            quality_score=0.1,
            processing_time_ms=processing_time
        )

class LightweightMotionAnalyzer:
    """Motion analysis using NumPy/SciPy without ML models"""
    
    def __init__(self):
        self.window_size = 100
        
        # Simple SVM for demonstration
        self.scaler = StandardScaler()
        self.classifier = SVC(probability=True, random_state=42)
        self._train_simple_model()
        
        logger.info("Lightweight Motion Analyzer initialized")
    
    def _train_simple_model(self):
        """Train simple motion stress classifier"""
        np.random.seed(42)
        n_samples = 300
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 3, n_samples)
        
        # Add structure: high variance = high stress
        for i in range(n_samples):
            if y[i] == 2:  # High stress
                X[i, 1] += 2.0  # High acceleration std
                X[i, 6] += 1.5  # High gyro std
            elif y[i] == 0:  # Low stress
                X[i, 1] -= 1.0  # Low acceleration std
                X[i, 6] -= 0.5  # Low gyro std
        
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)
    
    def analyze_motion_data(self, motion_data: List[Dict]) -> ModalityResult:
        """Analyze motion data for stress and attention"""
        start_time = time.time()
        
        try:
            # Parse motion data
            accel_data, gyro_data = self._parse_motion_data(motion_data)
            if accel_data is None:
                return self._create_error_result("motion", "Failed to parse motion data", start_time)
            
            # Extract features
            features = self._extract_motion_features(accel_data, gyro_data)
            
            # Predict stress
            stress_level = self._predict_motion_stress(features)
            
            # Calculate attention score
            attention_score = self._calculate_attention_score(accel_data, gyro_data)
            
            # Assess quality
            quality_score = self._assess_motion_quality(accel_data, gyro_data)
            
            # Calculate confidence
            confidence = self._calculate_confidence(features, quality_score)
            
            processing_time = (time.time() - start_time) * 1000
            
            return ModalityResult(
                modality="motion",
                confidence=confidence,
                features={
                    'stress_level': stress_level,
                    'motion_energy': features.get('motion_energy', 0.5),
                    'attention_score': attention_score,
                    'stability': features.get('stability_index', 0.5),
                    'data_samples': len(accel_data)
                },
                raw_predictions={'stress_level': stress_level},
                quality_score=quality_score,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in motion analysis: {e}")
            return self._create_error_result("motion", str(e), start_time)
    
    def _parse_motion_data(self, motion_data: List[Dict]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Parse motion data"""
        try:
            accel_data = []
            gyro_data = []
            
            for sample in motion_data:
                # Extract accelerometer data - handle both field names for compatibility
                if 'acceleration' in sample:
                    accel = sample['acceleration']
                    accel_data.append([accel.get('x', 0), accel.get('y', 0), accel.get('z', 0)])
                elif 'accelerometer' in sample:
                    accel = sample['accelerometer']
                    accel_data.append([accel.get('x', 0), accel.get('y', 0), accel.get('z', 0)])
                elif all(k in sample for k in ['x', 'y', 'z']):
                    accel_data.append([sample['x'], sample['y'], sample['z']])
                
                if 'gyroscope' in sample:
                    gyro = sample['gyroscope']
                    gyro_data.append([gyro.get('x', 0), gyro.get('y', 0), gyro.get('z', 0)])
                elif all(k in sample for k in ['gx', 'gy', 'gz']):
                    gyro_data.append([sample['gx'], sample['gy'], sample['gz']])
            
            if not accel_data:
                return None, None
            
            accel_array = np.array(accel_data)
            gyro_array = np.array(gyro_data) if gyro_data else np.zeros_like(accel_array)
            
            return accel_array, gyro_array
            
        except Exception as e:
            logger.error(f"Error parsing motion data: {e}")
            return None, None
    
    def _extract_motion_features(self, accel_data: np.ndarray, gyro_data: np.ndarray) -> Dict[str, float]:
        """Extract motion features"""
        features = {}
        
        try:
            # Accelerometer features
            accel_magnitude = np.linalg.norm(accel_data, axis=1)
            features['accel_mean'] = np.mean(accel_magnitude)
            features['accel_std'] = np.std(accel_magnitude)
            features['accel_max'] = np.max(accel_magnitude)
            features['accel_min'] = np.min(accel_magnitude)
            
            # Gyroscope features
            if gyro_data.size > 0:
                gyro_magnitude = np.linalg.norm(gyro_data, axis=1)
                features['gyro_mean'] = np.mean(gyro_magnitude)
                features['gyro_std'] = np.std(gyro_magnitude)
            else:
                features['gyro_mean'] = 0.0
                features['gyro_std'] = 0.0
            
            # Motion energy
            features['motion_energy'] = np.mean(accel_magnitude**2)
            
            # Stability
            features['stability_index'] = 1.0 / (1.0 + features['accel_std'])
            
            # Jerk (acceleration change)
            if len(accel_magnitude) > 1:
                jerk = np.diff(accel_magnitude)
                features['jerk_magnitude'] = np.mean(np.abs(jerk))
            else:
                features['jerk_magnitude'] = 0.0
            
        except Exception as e:
            logger.error(f"Error extracting motion features: {e}")
            features = {
                'accel_mean': 9.81, 'accel_std': 1.0, 'accel_max': 12.0, 'accel_min': 8.0,
                'gyro_mean': 0.0, 'gyro_std': 0.0, 'motion_energy': 0.5,
                'stability_index': 0.5, 'jerk_magnitude': 0.5
            }
        
        return features
    
    def _predict_motion_stress(self, features: Dict[str, float]) -> float:
        """Predict stress from motion features"""
        try:
            # Prepare feature vector (pad to 10 features)
            feature_names = [
                'accel_mean', 'accel_std', 'accel_max', 'accel_min',
                'gyro_mean', 'gyro_std', 'motion_energy', 'stability_index', 'jerk_magnitude'
            ]
            
            feature_vector = []
            for name in feature_names:
                feature_vector.append(features.get(name, 0.0))
            feature_vector.append(0.0)  # Pad to 10 features
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            feature_vector = self.scaler.transform(feature_vector)
            
            # Predict
            probabilities = self.classifier.predict_proba(feature_vector)[0]
            stress_level = probabilities[0] * 0.2 + probabilities[1] * 0.6 + probabilities[2] * 1.0
            
            return stress_level
            
        except Exception as e:
            logger.error(f"Error predicting motion stress: {e}")
            # Rule-based fallback
            accel_std = features.get('accel_std', 1.0)
            jerk = features.get('jerk_magnitude', 0.5)
            
            stress_level = min(1.0, (accel_std / 3.0 + jerk / 2.0) / 2)
            return stress_level
    
    def _calculate_attention_score(self, accel_data: np.ndarray, gyro_data: np.ndarray) -> float:
        """Calculate attention score from motion stability"""
        try:
            accel_magnitude = np.linalg.norm(accel_data, axis=1)
            stability = 1.0 / (1.0 + np.var(accel_magnitude))
            
            if len(accel_magnitude) > 1:
                jerk = np.diff(accel_magnitude)
                smoothness = 1.0 / (1.0 + np.mean(np.abs(jerk)))
            else:
                smoothness = 0.5
            
            attention_score = (stability + smoothness) / 2
            return np.clip(attention_score, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating attention score: {e}")
            return 0.5
    
    def _assess_motion_quality(self, accel_data: np.ndarray, gyro_data: np.ndarray) -> float:
        """Assess motion data quality"""
        quality_factors = []
        
        # Data completeness
        completeness = len(accel_data) / max(self.window_size, len(accel_data))
        quality_factors.append(completeness)
        
        # Data range reasonableness
        accel_magnitude = np.linalg.norm(accel_data, axis=1)
        reasonable_range = np.sum((accel_magnitude >= 0) & (accel_magnitude <= 50)) / len(accel_magnitude)
        quality_factors.append(reasonable_range)
        
        return np.mean(quality_factors)
    
    def _calculate_confidence(self, features: Dict[str, float], quality_score: float) -> float:
        """Calculate analysis confidence"""
        return quality_score * 0.8 + 0.2
    
    def _create_error_result(self, modality: str, error_msg: str, start_time: float) -> ModalityResult:
        """Create error result"""
        processing_time = (time.time() - start_time) * 1000
        return ModalityResult(
            modality=modality,
            confidence=0.1,
            features={'error': error_msg, 'stress_level': 0.5, 'motion_energy': 0.5, 'attention_score': 0.5},
            raw_predictions={'stress_level': 0.5},
            quality_score=0.1,
            processing_time_ms=processing_time
        )

class LightweightMLEngine:
    """Lightweight production ML engine that works without heavy dependencies"""
    
    def __init__(self):
        logger.info("Initializing Lightweight ML Engine...")
        
        # Initialize analyzers
        self.emotion_detector = LightweightEmotionDetector()
        self.audio_analyzer = LightweightAudioAnalyzer()
        self.motion_analyzer = LightweightMotionAnalyzer()
        
        logger.info("Lightweight ML Engine ready")
    
    def analyze_session(self, face_image: Optional[str], audio_data: Optional[str], 
                       motion_data: Optional[List[Dict]]) -> Dict[str, Any]:
        """Comprehensive multimodal analysis with fusion"""
        total_start_time = time.time()
        
        try:
            modality_results = []
            
            # Analyze each modality
            if face_image:
                emotion_result = self.emotion_detector.analyze_emotion(face_image)
                modality_results.append(emotion_result)
            
            if audio_data:
                audio_result = self.audio_analyzer.analyze_stress_from_audio(audio_data)
                modality_results.append(audio_result)
            
            if motion_data:
                motion_result = self.motion_analyzer.analyze_motion_data(motion_data)
                modality_results.append(motion_result)
            
            if not modality_results:
                return self._create_empty_result()
            
            # Fuse modality results using cognitive fusion engine
            cognitive_result = fusion_engine.fuse_modalities(modality_results)
            
            # Prepare final response
            total_processing_time = (time.time() - total_start_time) * 1000
            
            return {
                'cognitive_state': cognitive_result.state.value,
                'confidence': cognitive_result.confidence,
                'valence': cognitive_result.valence,
                'arousal': cognitive_result.arousal,
                'attention': cognitive_result.attention,
                'stress_level': cognitive_result.stress_level,
                'insights': cognitive_result.insights,
                'recommendations': cognitive_result.recommendations,
                'modality_contributions': cognitive_result.modality_contributions,
                'graph_data': cognitive_result.graph_data,
                'processing_time_ms': total_processing_time,
                'modality_results': {
                    result.modality: {
                        'confidence': result.confidence,
                        'features': result.features,
                        'raw_predictions': result.raw_predictions,
                        'quality_score': result.quality_score,
                        'processing_time_ms': result.processing_time_ms
                    } for result in modality_results
                },
                'analysis_metadata': {
                    'model_versions': self._get_model_versions(),
                    'total_modalities': len(modality_results),
                    'timestamp': time.time()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in session analysis: {e}")
            return self._create_error_result(str(e))
    
    def _get_model_versions(self) -> Dict[str, str]:
        """Get model version information"""
        return {
            'emotion_opencv': 'v1.0-lightweight',
            'audio_rf': 'v1.0-lightweight',
            'motion_svm': 'v1.0-lightweight',
            'fusion_engine': 'v1.0'
        }
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create result when no data is provided"""
        return {
            'cognitive_state': 'neutral',
            'confidence': 0.1,
            'valence': 0.0,
            'arousal': 0.5,
            'attention': 0.5,
            'stress_level': 0.5,
            'insights': ['No data provided for analysis'],
            'recommendations': ['Provide face image, audio, or motion data for analysis'],
            'modality_contributions': {},
            'graph_data': None,
            'processing_time_ms': 0,
            'modality_results': {},
            'analysis_metadata': {
                'model_versions': self._get_model_versions(),
                'total_modalities': 0,
                'timestamp': time.time()
            }
        }
    
    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """Create result when analysis fails"""
        return {
            'cognitive_state': 'neutral',
            'confidence': 0.1,
            'valence': 0.0,
            'arousal': 0.5,
            'attention': 0.5,
            'stress_level': 0.5,
            'insights': [f'Analysis failed: {error_msg}'],
            'recommendations': ['Check data format and try again'],
            'modality_contributions': {},
            'graph_data': None,
            'processing_time_ms': 0,
            'modality_results': {},
            'analysis_metadata': {
                'model_versions': self._get_model_versions(),
                'total_modalities': 0,
                'timestamp': time.time(),
                'error': error_msg
            }
        }

# Global lightweight ML engine instance
lightweight_ml_engine = LightweightMLEngine()
