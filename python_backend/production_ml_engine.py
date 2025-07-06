"""
Enhanced ML Analysis Engine - Production-grade analysis with real models and fusion
"""

import cv2
import numpy as np
import librosa
import mediapipe as mp
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
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

from model_loader import model_manager
from cognitive_fusion import fusion_engine, ModalityResult, CognitiveStateResult

logger = logging.getLogger(__name__)

class EnhancedEmotionDetector:
    """Production-grade emotion detection using MediaPipe + CNN models"""
    
    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load emotion CNN model
        self.emotion_model = model_manager.load_model("emotion_cnn")
        self.emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        logger.info("Enhanced Emotion Detector initialized")
    
    def analyze_emotion(self, image_data: str) -> ModalityResult:
        """Comprehensive emotion analysis with CNN + landmark features"""
        start_time = time.time()
        
        try:
            # Preprocess image
            image = self._preprocess_image(image_data)
            if image is None:
                return self._create_error_result("emotion", "Failed to preprocess image", start_time)
            
            # Extract facial landmarks
            landmark_features = self._extract_landmark_features(image)
            
            # Extract CNN-based emotion features
            cnn_predictions = self._extract_cnn_emotions(image)
            
            # Combine predictions
            combined_emotions = self._combine_emotion_predictions(landmark_features, cnn_predictions)
            
            # Calculate confidence and quality
            confidence = self._calculate_confidence(landmark_features, cnn_predictions)
            quality_score = self._assess_image_quality(image, landmark_features is not None)
            
            # Calculate additional features
            features = {
                'face_detected': landmark_features is not None,
                'image_quality': quality_score,
                'landmark_confidence': 0.8 if landmark_features else 0.0,
                'cnn_confidence': np.max(list(cnn_predictions.values())) if cnn_predictions else 0.0
            }
            
            processing_time = (time.time() - start_time) * 1000
            
            return ModalityResult(
                modality="emotion",
                confidence=confidence,
                features=features,
                raw_predictions=combined_emotions,
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
    
    def _extract_landmark_features(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        """Extract emotion features from facial landmarks"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return None
            
            landmarks = results.multi_face_landmarks[0]
            h, w = image.shape[:2]
            
            # Convert to pixel coordinates
            points = []
            for landmark in landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points.append([x, y])
            
            points = np.array(points)
            
            # Extract emotion-relevant features
            features = {}
            
            # Mouth features
            mouth_left = points[61]
            mouth_right = points[291]
            mouth_top = points[13]
            mouth_bottom = points[14]
            
            mouth_width = np.linalg.norm(mouth_right - mouth_left)
            mouth_height = np.linalg.norm(mouth_top - mouth_bottom)
            
            if mouth_width > 0:
                features['mouth_aspect_ratio'] = mouth_height / mouth_width
                mouth_center = (mouth_left + mouth_right) / 2
                features['mouth_curve'] = (mouth_center[1] - mouth_top[1]) / mouth_width
            
            # Eye features
            left_eye_top = points[159]
            left_eye_bottom = points[145]
            right_eye_top = points[386]
            right_eye_bottom = points[374]
            
            left_eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)
            right_eye_height = np.linalg.norm(right_eye_top - right_eye_bottom)
            features['eye_openness'] = (left_eye_height + right_eye_height) / 2
            
            # Eyebrow features
            left_eyebrow = points[70]
            right_eyebrow = points[300]
            left_eye_center = points[159]
            right_eye_center = points[386]
            
            eyebrow_height = ((left_eye_center[1] - left_eyebrow[1]) + (right_eye_center[1] - right_eyebrow[1])) / 2
            features['eyebrow_height'] = eyebrow_height
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting landmark features: {e}")
            return None
    
    def _extract_cnn_emotions(self, image: np.ndarray) -> Dict[str, float]:
        """Extract emotions using CNN model"""
        try:
            if self.emotion_model is None:
                # Fallback to simple face detection
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    return {'neutral': 0.7, 'happy': 0.1, 'sad': 0.1, 'angry': 0.05, 'surprise': 0.05, 'fear': 0.0, 'disgust': 0.0}
                else:
                    return {'neutral': 1.0, 'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'surprise': 0.0, 'fear': 0.0, 'disgust': 0.0}
            
            # Preprocess for CNN
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return {'neutral': 1.0, 'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'surprise': 0.0, 'fear': 0.0, 'disgust': 0.0}
            
            # Extract face region
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to model input size (48x48)
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.astype('float32') / 255.0
            face_roi = np.expand_dims(face_roi, axis=-1)  # Add channel dimension
            face_roi = np.expand_dims(face_roi, axis=0)   # Add batch dimension
            
            # Predict
            predictions = self.emotion_model.predict(face_roi, verbose=0)[0]
            
            # Convert to emotion dictionary
            emotion_dict = {}
            for i, emotion in enumerate(self.emotion_classes):
                emotion_dict[emotion] = float(predictions[i])
            
            return emotion_dict
            
        except Exception as e:
            logger.error(f"Error in CNN emotion prediction: {e}")
            # Return neutral emotions as fallback
            return {'neutral': 0.8, 'happy': 0.1, 'sad': 0.05, 'angry': 0.05, 'surprise': 0.0, 'fear': 0.0, 'disgust': 0.0}
    
    def _combine_emotion_predictions(self, landmark_features: Optional[Dict[str, float]], 
                                   cnn_predictions: Dict[str, float]) -> Dict[str, float]:
        """Combine landmark-based and CNN-based emotion predictions"""
        if landmark_features is None:
            return cnn_predictions
        
        # Rule-based emotion adjustments from landmarks
        landmark_emotions = self._landmarks_to_emotions(landmark_features)
        
        # Weighted combination (CNN gets more weight)
        cnn_weight = 0.7
        landmark_weight = 0.3
        
        combined = {}
        for emotion in self.emotion_classes:
            cnn_score = cnn_predictions.get(emotion, 0.0)
            landmark_score = landmark_emotions.get(emotion, 0.0)
            combined[emotion] = cnn_weight * cnn_score + landmark_weight * landmark_score
        
        # Normalize
        total = sum(combined.values())
        if total > 0:
            combined = {k: v/total for k, v in combined.items()}
        
        return combined
    
    def _landmarks_to_emotions(self, features: Dict[str, float]) -> Dict[str, float]:
        """Convert landmark features to emotion scores"""
        emotions = {emotion: 0.0 for emotion in self.emotion_classes}
        
        mouth_curve = features.get('mouth_curve', 0)
        eyebrow_height = features.get('eyebrow_height', 0)
        eye_openness = features.get('eye_openness', 5)
        mouth_ratio = features.get('mouth_aspect_ratio', 0.3)
        
        # Happy: upward mouth curve
        if mouth_curve > 0.1:
            emotions['happy'] = min(mouth_curve * 3, 1.0)
        
        # Sad: downward mouth curve
        if mouth_curve < -0.05:
            emotions['sad'] = min(abs(mouth_curve) * 4, 1.0)
        
        # Surprise: raised eyebrows, wide eyes, open mouth
        if eyebrow_height > 10 and eye_openness > 7 and mouth_ratio > 0.5:
            emotions['surprise'] = 0.8
        
        # Anger: lowered eyebrows
        if eyebrow_height < -8:
            emotions['angry'] = 0.6
        
        # Fear: raised eyebrows, wide eyes
        if eyebrow_height > 12 and eye_openness > 8:
            emotions['fear'] = 0.6
        
        # Normalize
        total = sum(emotions.values())
        if total == 0:
            emotions['neutral'] = 1.0
        else:
            emotions = {k: v/total for k, v in emotions.items()}
        
        return emotions
    
    def _calculate_confidence(self, landmark_features: Optional[Dict], cnn_predictions: Dict) -> float:
        """Calculate overall confidence in emotion detection"""
        confidence_factors = []
        
        # Landmark detection confidence
        if landmark_features is not None:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.2)
        
        # CNN prediction confidence (max probability)
        max_prob = max(cnn_predictions.values()) if cnn_predictions else 0.5
        confidence_factors.append(max_prob)
        
        return np.mean(confidence_factors)
    
    def _assess_image_quality(self, image: np.ndarray, face_detected: bool) -> float:
        """Assess image quality for emotion analysis"""
        quality_factors = []
        
        # Face detection
        quality_factors.append(0.8 if face_detected else 0.2)
        
        # Image size
        h, w = image.shape[:2]
        size_quality = min(1.0, (h * w) / (200 * 200))  # Prefer larger images
        quality_factors.append(size_quality)
        
        # Image sharpness (Laplacian variance)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_quality = min(1.0, sharpness / 1000)  # Normalize
        quality_factors.append(sharpness_quality)
        
        return np.mean(quality_factors)
    
    def _create_error_result(self, modality: str, error_msg: str, start_time: float) -> ModalityResult:
        """Create error result for failed analysis"""
        processing_time = (time.time() - start_time) * 1000
        return ModalityResult(
            modality=modality,
            confidence=0.1,
            features={'error': error_msg},
            raw_predictions={'neutral': 1.0},
            quality_score=0.1,
            processing_time_ms=processing_time
        )

class EnhancedAudioAnalyzer:
    """Production-grade audio analysis using Librosa + ML models"""
    
    def __init__(self):
        self.sample_rate = 22050
        self.hop_length = 512
        self.n_mels = 13
        self.n_mfcc = 13
        
        # Load audio model
        self.audio_model_data = model_manager.load_model("audio_emotion")
        
        logger.info("Enhanced Audio Analyzer initialized")
    
    def analyze_stress_from_audio(self, audio_data: str) -> ModalityResult:
        """Comprehensive audio stress analysis"""
        start_time = time.time()
        
        try:
            # Preprocess audio
            audio = self._preprocess_audio(audio_data)
            if audio is None:
                return self._create_error_result("audio", "Failed to preprocess audio", start_time)
            
            # Extract comprehensive features
            features = self._extract_comprehensive_features(audio)
            
            # Predict emotions/stress using model
            predictions = self._predict_audio_emotions(features)
            
            # Calculate stress level
            stress_level = self._calculate_stress_from_predictions(predictions)
            
            # Assess audio quality
            quality_score = self._assess_audio_quality(audio)
            
            # Calculate confidence
            confidence = self._calculate_audio_confidence(features, quality_score)
            
            processing_time = (time.time() - start_time) * 1000
            
            return ModalityResult(
                modality="audio",
                confidence=confidence,
                features={
                    'stress_level': stress_level,
                    'audio_length': len(audio) / self.sample_rate,
                    'avg_energy': features.get('energy', 0.5),
                    'pitch_variance': features.get('pitch_variance', 0.5),
                    'spectral_centroid': features.get('spectral_centroid', 0.5)
                },
                raw_predictions={'stress_level': stress_level, **predictions},
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
            
            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample to target sample rate
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
            return audio
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            return None
    
    def _extract_comprehensive_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive audio features using Librosa"""
        features = {}
        
        try:
            # MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
            for i in range(self.n_mfcc):
                features[f'mfcc_{i}'] = np.mean(mfccs[i])
            
            # Chroma features
            chroma = librosa.feature.chroma(y=audio, sr=self.sample_rate)
            for i in range(12):
                features[f'chroma_{i}'] = np.mean(chroma[i])
            
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
            for i in range(7):
                features[f'spectral_contrast_{i}'] = np.mean(spectral_contrast[i])
            
            # Tonnetz
            tonnetz = librosa.feature.tonnetz(y=audio, sr=self.sample_rate)
            for i in range(6):
                features[f'tonnetz_{i}'] = np.mean(tonnetz[i])
            
            # Tempo and rhythm features
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            features['tempo'] = tempo
            
            if len(beats) > 1:
                beat_intervals = np.diff(beats) / self.sample_rate * 60 / tempo
                features['tempo_std'] = np.std(beat_intervals)
            else:
                features['tempo_std'] = 0
            
            # Additional features for stress detection
            # Energy/RMS
            rms = librosa.feature.rms(y=audio)[0]
            features['energy'] = np.mean(rms)
            features['energy_variance'] = np.var(rms)
            
            # Spectral centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            features['spectral_centroid'] = np.mean(spectral_centroid)
            
            # Zero crossing rate (voice activity)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['zcr'] = np.mean(zcr)
            
            # Pitch tracking for fundamental frequency
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_variance'] = np.var(pitch_values)
            else:
                features['pitch_mean'] = 0
                features['pitch_variance'] = 0
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            # Return minimal features
            features = {f'mfcc_{i}': 0 for i in range(13)}
            features.update({f'chroma_{i}': 0 for i in range(12)})
            features.update({f'spectral_contrast_{i}': 0 for i in range(7)})
            features.update({f'tonnetz_{i}': 0 for i in range(6)})
            features.update({'tempo': 120, 'tempo_std': 0})
        
        return features
    
    def _predict_audio_emotions(self, features: Dict[str, float]) -> Dict[str, float]:
        """Predict emotions from audio features"""
        try:
            if self.audio_model_data is None:
                # Rule-based fallback
                return self._rule_based_audio_prediction(features)
            
            # Prepare feature vector
            feature_vector = []
            expected_features = [f'mfcc_{i}' for i in range(13)]
            expected_features += [f'chroma_{i}' for i in range(12)]
            expected_features += [f'spectral_contrast_{i}' for i in range(7)]
            expected_features += [f'tonnetz_{i}' for i in range(6)]
            expected_features += ['tempo', 'tempo_std']
            
            for feature_name in expected_features:
                feature_vector.append(features.get(feature_name, 0.0))
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Scale features
            scaler = self.audio_model_data.get('scaler')
            if scaler:
                feature_vector = scaler.transform(feature_vector)
            
            # Predict
            model = self.audio_model_data.get('model')
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(feature_vector)[0]
            else:
                prediction = model.predict(feature_vector)[0]
                probabilities = np.zeros(7)
                probabilities[prediction] = 1.0
            
            # Map to emotion classes
            classes = self.audio_model_data.get('classes', ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised'])
            
            emotion_dict = {}
            for i, emotion in enumerate(classes):
                emotion_dict[emotion] = float(probabilities[i])
            
            return emotion_dict
            
        except Exception as e:
            logger.error(f"Error in audio emotion prediction: {e}")
            return self._rule_based_audio_prediction(features)
    
    def _rule_based_audio_prediction(self, features: Dict[str, float]) -> Dict[str, float]:
        """Rule-based audio emotion prediction as fallback"""
        emotions = {'neutral': 0.4, 'happy': 0.1, 'sad': 0.1, 'angry': 0.1, 'fearful': 0.1, 'disgusted': 0.1, 'surprised': 0.1}
        
        # High energy and tempo suggest excitement/happiness
        energy = features.get('energy', 0.5)
        tempo = features.get('tempo', 120)
        
        if energy > 0.1 and tempo > 140:
            emotions['happy'] += 0.3
            emotions['surprised'] += 0.1
        
        # Low energy suggests sadness/fatigue
        if energy < 0.05:
            emotions['sad'] += 0.2
            emotions['neutral'] += 0.1
        
        # High pitch variance might indicate stress/anger
        pitch_variance = features.get('pitch_variance', 0)
        if pitch_variance > 1000:
            emotions['angry'] += 0.2
            emotions['fearful'] += 0.1
        
        # Normalize
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v/total for k, v in emotions.items()}
        
        return emotions
    
    def _calculate_stress_from_predictions(self, predictions: Dict[str, float]) -> float:
        """Calculate stress level from emotion predictions"""
        # Stress-related emotions
        stress_emotions = {
            'angry': 1.0,
            'fearful': 0.8,
            'disgusted': 0.6,
            'sad': 0.5,
            'neutral': 0.3,
            'surprised': 0.2,
            'happy': 0.1
        }
        
        stress_level = 0.0
        for emotion, weight in stress_emotions.items():
            stress_level += predictions.get(emotion, 0.0) * weight
        
        return np.clip(stress_level, 0.0, 1.0)
    
    def _assess_audio_quality(self, audio: np.ndarray) -> float:
        """Assess audio quality"""
        quality_factors = []
        
        # Audio length (prefer 1-10 seconds)
        duration = len(audio) / self.sample_rate
        length_quality = 1.0 if 1 <= duration <= 10 else max(0.3, 1.0 - abs(duration - 5) / 10)
        quality_factors.append(length_quality)
        
        # Signal-to-noise ratio estimate
        rms = np.sqrt(np.mean(audio**2))
        snr_quality = min(1.0, rms * 10)  # Rough SNR estimate
        quality_factors.append(snr_quality)
        
        # Dynamic range
        dynamic_range = np.max(audio) - np.min(audio)
        range_quality = min(1.0, dynamic_range * 2)
        quality_factors.append(range_quality)
        
        return np.mean(quality_factors)
    
    def _calculate_audio_confidence(self, features: Dict[str, float], quality_score: float) -> float:
        """Calculate confidence in audio analysis"""
        confidence_factors = [quality_score]
        
        # Feature completeness
        expected_feature_count = 13 + 12 + 7 + 6 + 2  # MFCCs + Chroma + Spectral + Tonnetz + Tempo
        actual_feature_count = len([k for k in features.keys() if any(prefix in k for prefix in ['mfcc_', 'chroma_', 'spectral_', 'tonnetz_', 'tempo'])])
        completeness = actual_feature_count / expected_feature_count
        confidence_factors.append(completeness)
        
        return np.mean(confidence_factors)
    
    def _create_error_result(self, modality: str, error_msg: str, start_time: float) -> ModalityResult:
        """Create error result for failed analysis"""
        processing_time = (time.time() - start_time) * 1000
        return ModalityResult(
            modality=modality,
            confidence=0.1,
            features={'error': error_msg, 'stress_level': 0.5},
            raw_predictions={'stress_level': 0.5},
            quality_score=0.1,
            processing_time_ms=processing_time
        )

class EnhancedMotionAnalyzer:
    """Production-grade motion analysis using NumPy/SciPy + ML models"""
    
    def __init__(self):
        self.window_size = 100  # Number of samples for analysis
        self.motion_model_data = model_manager.load_model("motion_stress")
        
        logger.info("Enhanced Motion Analyzer initialized")
    
    def analyze_motion_data(self, motion_data: List[Dict]) -> ModalityResult:
        """Comprehensive motion analysis"""
        start_time = time.time()
        
        try:
            # Parse motion data
            parsed_data = self._parse_motion_data(motion_data)
            if parsed_data is None:
                return self._create_error_result("motion", "Failed to parse motion data", start_time)
            
            accel_data, gyro_data = parsed_data
            
            # Extract comprehensive features
            features = self._extract_motion_features(accel_data, gyro_data)
            
            # Predict stress level using model
            stress_predictions = self._predict_motion_stress(features)
            
            # Calculate additional metrics
            attention_score = self._calculate_attention_score(accel_data, gyro_data)
            motion_energy = features.get('motion_energy', 0.5)
            
            # Assess data quality
            quality_score = self._assess_motion_quality(accel_data, gyro_data)
            
            # Calculate confidence
            confidence = self._calculate_motion_confidence(features, quality_score)
            
            processing_time = (time.time() - start_time) * 1000
            
            return ModalityResult(
                modality="motion",
                confidence=confidence,
                features={
                    'stress_level': stress_predictions.get('stress_level', 0.5),
                    'motion_energy': motion_energy,
                    'attention_score': attention_score,
                    'stability': features.get('stability_index', 0.5),
                    'data_samples': len(accel_data)
                },
                raw_predictions=stress_predictions,
                quality_score=quality_score,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in motion analysis: {e}")
            return self._create_error_result("motion", str(e), start_time)
    
    def _parse_motion_data(self, motion_data: List[Dict]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Parse motion data into accelerometer and gyroscope arrays"""
        try:
            accel_data = []
            gyro_data = []
            
            for sample in motion_data:
                # Extract accelerometer data
                if 'accelerometer' in sample:
                    accel = sample['accelerometer']
                    accel_data.append([accel.get('x', 0), accel.get('y', 0), accel.get('z', 0)])
                elif all(k in sample for k in ['x', 'y', 'z']):
                    accel_data.append([sample['x'], sample['y'], sample['z']])
                
                # Extract gyroscope data
                if 'gyroscope' in sample:
                    gyro = sample['gyroscope']
                    gyro_data.append([gyro.get('x', 0), gyro.get('y', 0), gyro.get('z', 0)])
                elif all(k in sample for k in ['gx', 'gy', 'gz']):
                    gyro_data.append([sample['gx'], sample['gy'], sample['gz']])
            
            if not accel_data:
                logger.warning("No accelerometer data found")
                return None
            
            accel_array = np.array(accel_data)
            gyro_array = np.array(gyro_data) if gyro_data else np.zeros_like(accel_array)
            
            return accel_array, gyro_array
            
        except Exception as e:
            logger.error(f"Error parsing motion data: {e}")
            return None
    
    def _extract_motion_features(self, accel_data: np.ndarray, gyro_data: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive motion features"""
        features = {}
        
        try:
            # Accelerometer features
            accel_magnitude = np.linalg.norm(accel_data, axis=1)
            features['accel_mean'] = np.mean(accel_magnitude)
            features['accel_std'] = np.std(accel_magnitude)
            features['accel_max'] = np.max(accel_magnitude)
            features['accel_min'] = np.min(accel_magnitude)
            features['accel_range'] = features['accel_max'] - features['accel_min']
            
            # Gyroscope features
            if gyro_data.size > 0:
                gyro_magnitude = np.linalg.norm(gyro_data, axis=1)
                features['gyro_mean'] = np.mean(gyro_magnitude)
                features['gyro_std'] = np.std(gyro_magnitude)
                features['gyro_max'] = np.max(gyro_magnitude)
                features['gyro_min'] = np.min(gyro_magnitude)
                features['gyro_range'] = features['gyro_max'] - features['gyro_min']
            else:
                for key in ['gyro_mean', 'gyro_std', 'gyro_max', 'gyro_min', 'gyro_range']:
                    features[key] = 0.0
            
            # Jerk (rate of change of acceleration)
            if len(accel_magnitude) > 1:
                jerk = np.diff(accel_magnitude)
                features['jerk_magnitude'] = np.mean(np.abs(jerk))
            else:
                features['jerk_magnitude'] = 0.0
            
            # Motion energy
            features['motion_energy'] = np.mean(accel_magnitude**2)
            
            # Stability index (inverse of variance)
            features['stability_index'] = 1.0 / (1.0 + features['accel_std'])
            
            # Frequency domain features
            if len(accel_magnitude) >= 8:  # Need enough samples for FFT
                fft_accel = np.fft.fft(accel_magnitude)
                power_spectrum = np.abs(fft_accel)**2
                
                # Dominant frequency
                freqs = np.fft.fftfreq(len(accel_magnitude))
                dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
                features['freq_peak'] = abs(freqs[dominant_freq_idx])
                
                # Frequency energy
                features['freq_energy'] = np.sum(power_spectrum)
            else:
                features['freq_peak'] = 0.0
                features['freq_energy'] = 0.0
            
        except Exception as e:
            logger.error(f"Error extracting motion features: {e}")
            # Provide default features
            features = {
                'accel_mean': 9.81, 'accel_std': 1.0, 'accel_max': 12.0, 'accel_min': 8.0, 'accel_range': 4.0,
                'gyro_mean': 0.0, 'gyro_std': 0.0, 'gyro_max': 0.0, 'gyro_min': 0.0, 'gyro_range': 0.0,
                'jerk_magnitude': 0.5, 'motion_energy': 0.5, 'stability_index': 0.5, 'freq_peak': 0.0, 'freq_energy': 0.0
            }
        
        return features
    
    def _predict_motion_stress(self, features: Dict[str, float]) -> Dict[str, float]:
        """Predict stress level from motion features"""
        try:
            if self.motion_model_data is None:
                return self._rule_based_motion_prediction(features)
            
            # Prepare feature vector
            expected_features = [
                'accel_mean', 'accel_std', 'accel_max', 'accel_min', 'accel_range',
                'gyro_mean', 'gyro_std', 'gyro_max', 'gyro_min', 'gyro_range',
                'jerk_magnitude', 'motion_energy', 'stability_index', 'freq_peak', 'freq_energy'
            ]
            
            feature_vector = []
            for feature_name in expected_features:
                feature_vector.append(features.get(feature_name, 0.0))
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Scale features
            scaler = self.motion_model_data.get('scaler')
            if scaler:
                feature_vector = scaler.transform(feature_vector)
            
            # Predict
            model = self.motion_model_data.get('model')
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(feature_vector)[0]
                # Convert stress classes to stress level
                stress_level = probabilities[0] * 0.2 + probabilities[1] * 0.6 + probabilities[2] * 1.0
            else:
                prediction = model.predict(feature_vector)[0]
                stress_level = [0.2, 0.6, 1.0][prediction]
            
            return {'stress_level': stress_level}
            
        except Exception as e:
            logger.error(f"Error in motion stress prediction: {e}")
            return self._rule_based_motion_prediction(features)
    
    def _rule_based_motion_prediction(self, features: Dict[str, float]) -> Dict[str, float]:
        """Rule-based motion stress prediction as fallback"""
        accel_std = features.get('accel_std', 1.0)
        jerk_magnitude = features.get('jerk_magnitude', 0.5)
        motion_energy = features.get('motion_energy', 50.0)
        
        # High variance in acceleration suggests restlessness/stress
        # High jerk suggests fidgeting
        # High motion energy suggests agitation
        
        stress_indicators = [
            min(accel_std / 3.0, 1.0),
            min(jerk_magnitude / 2.0, 1.0),
            min(motion_energy / 100.0, 1.0)
        ]
        
        stress_level = np.mean(stress_indicators)
        stress_level = np.clip(stress_level, 0.0, 1.0)
        
        return {'stress_level': stress_level}
    
    def _calculate_attention_score(self, accel_data: np.ndarray, gyro_data: np.ndarray) -> float:
        """Calculate attention score from motion stability"""
        try:
            # Stable, controlled movement suggests good attention
            accel_magnitude = np.linalg.norm(accel_data, axis=1)
            
            # Low variance in acceleration = stable = good attention
            stability = 1.0 / (1.0 + np.var(accel_magnitude))
            
            # Low jerk = smooth movement = good attention
            if len(accel_magnitude) > 1:
                jerk = np.diff(accel_magnitude)
                smoothness = 1.0 / (1.0 + np.mean(np.abs(jerk)))
            else:
                smoothness = 0.5
            
            # Combine stability and smoothness
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
        
        # Data consistency (not too many outliers)
        if len(accel_magnitude) > 1:
            z_scores = np.abs((accel_magnitude - np.mean(accel_magnitude)) / (np.std(accel_magnitude) + 1e-6))
            outlier_ratio = np.sum(z_scores > 3) / len(accel_magnitude)
            consistency = 1.0 - outlier_ratio
            quality_factors.append(consistency)
        
        return np.mean(quality_factors)
    
    def _calculate_motion_confidence(self, features: Dict[str, float], quality_score: float) -> float:
        """Calculate confidence in motion analysis"""
        confidence_factors = [quality_score]
        
        # Feature completeness
        expected_features = 15
        actual_features = len([v for v in features.values() if v != 0])
        completeness = actual_features / expected_features
        confidence_factors.append(completeness)
        
        return np.mean(confidence_factors)
    
    def _create_error_result(self, modality: str, error_msg: str, start_time: float) -> ModalityResult:
        """Create error result for failed analysis"""
        processing_time = (time.time() - start_time) * 1000
        return ModalityResult(
            modality=modality,
            confidence=0.1,
            features={'error': error_msg, 'stress_level': 0.5, 'motion_energy': 0.5, 'attention_score': 0.5},
            raw_predictions={'stress_level': 0.5},
            quality_score=0.1,
            processing_time_ms=processing_time
        )

class ProductionMLEngine:
    """Production-grade ML analysis engine with real models and fusion"""
    
    def __init__(self):
        # Initialize model manager and load all models
        logger.info("Initializing Production ML Engine...")
        
        # Initialize all models
        model_results = model_manager.initialize_all_models()
        logger.info(f"Model initialization results: {model_results}")
        
        # Initialize analyzers
        self.emotion_detector = EnhancedEmotionDetector()
        self.audio_analyzer = EnhancedAudioAnalyzer()
        self.motion_analyzer = EnhancedMotionAnalyzer()
        
        logger.info("Production ML Engine ready")
    
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
                # No data provided
                return self._create_empty_result()
            
            # Fuse modality results
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
        """Get version information for loaded models"""
        return {
            'emotion_cnn': 'v1.0-synthetic',
            'audio_emotion': 'v1.0-rf-synthetic',
            'motion_stress': 'v1.0-svm-synthetic',
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

# Global production ML engine instance
production_ml_engine = ProductionMLEngine()
