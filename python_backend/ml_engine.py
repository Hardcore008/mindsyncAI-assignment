"""
Production-grade ML Analysis Engine with real model pipelines
"""

import asyncio
import base64
import io
import logging
import time
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import cv2
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn.functional as F
from PIL import Image
import mediapipe as mp
from scipy import signal
from scipy.fft import fft
import pickle
import json
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from config import settings
from utils import InvalidMotionDataError

logger = logging.getLogger(__name__)

class FacialEmotionPipeline:
    """
    Facial emotion analysis pipeline using OpenCV + MediaPipe + PyTorch
    Processes JPEG/PNG images and outputs 7-way emotion classification
    """
    
    def __init__(self):
        self.emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.model = None
        self.face_mesh = None
        self.face_detector = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
    async def initialize(self):
        """Initialize MediaPipe and PyTorch model"""
        logger.info("Initializing facial emotion pipeline...")
        
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize OpenCV face detector
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load PyTorch emotion model
        try:
            if settings.FACE_MODEL_PATH.exists():
                self.model = torch.load(settings.FACE_MODEL_PATH, map_location='cpu')
                self.model.eval()
                logger.info("Loaded facial emotion model from checkpoint")
            else:
                # Create a simple CNN model for demonstration
                self.model = self._create_demo_model()
                logger.warning("Using demo facial emotion model")
        except Exception as e:
            logger.error(f"Error loading facial model: {e}")
            self.model = self._create_demo_model()
        
        logger.info("Facial emotion pipeline ready")
    
    def _create_demo_model(self):
        """Create a simple demo CNN model"""
        import torch.nn as nn
        
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(64 * 56 * 56, 128)
                self.fc2 = nn.Linear(128, 7)
                self.dropout = nn.Dropout(0.5)
                
            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, 64 * 56 * 56)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return F.softmax(x, dim=1)
        
        model = SimpleCNN()
        model.eval()
        return model
    
    async def analyze(self, image_b64: str) -> Tuple[str, float]:
        """
        Analyze facial emotion from base64 image
        
        Returns:
            Tuple of (emotion_label, confidence_score)
        """
        start_time = time.time()
        
        try:
            # Decode base64 image
            image = self._decode_image(image_b64)
            if image is None:
                return "neutral", 0.1
            
            # Run analysis in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._analyze_sync,
                image
            )
            
            processing_time = (time.time() - start_time) * 1000
            if processing_time > 400:
                logger.warning(f"Facial analysis took {processing_time:.1f}ms (>400ms threshold)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in facial emotion analysis: {e}")
            return "neutral", 0.1
    
    def _decode_image(self, image_b64: str) -> Optional[np.ndarray]:
        """Decode base64 image to OpenCV format"""
        try:
            if image_b64.startswith('data:image/'):
                image_b64 = image_b64.split(',')[1]
            
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            return image
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return None
    
    def _analyze_sync(self, image: np.ndarray) -> Tuple[str, float]:
        """Synchronous analysis method for thread pool execution"""
        # Detect and align face
        aligned_face = self._detect_and_align_face(image)
        if aligned_face is None:
            return "neutral", 0.1
        
        # Extract MediaPipe landmarks
        landmarks = self._extract_landmarks(image)
        
        # Preprocess for PyTorch model
        model_input = self._preprocess_for_model(aligned_face)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(model_input)
            probabilities = outputs.cpu().numpy()[0]
        
        # Get top emotion and confidence
        emotion_idx = np.argmax(probabilities)
        emotion_label = self.emotion_classes[emotion_idx]
        confidence = float(probabilities[emotion_idx])
        
        # Adjust confidence based on landmark quality
        if landmarks is not None:
            landmark_confidence = self._assess_landmark_quality(landmarks)
            confidence = (confidence + landmark_confidence) / 2
        
        return emotion_label, confidence
    
    def _detect_and_align_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect face and align to canonical bounding box"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
        
        # Use largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Extract and resize face region
        face_roi = gray[y:y+h, x:x+w]
        face_aligned = cv2.resize(face_roi, (224, 224))
        
        return face_aligned
    
    def _extract_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract 468 facial landmarks using MediaPipe"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]
        
        # Convert to normalized coordinates and flatten to 1x936 vector
        landmark_array = []
        for landmark in landmarks.landmark:
            landmark_array.extend([landmark.x, landmark.y])
        
        return np.array(landmark_array, dtype=np.float32)
    
    def _preprocess_for_model(self, face_image: np.ndarray) -> torch.Tensor:
        """Preprocess face image for PyTorch model (224x224, normalized)"""
        # Ensure 224x224
        if face_image.shape != (224, 224):
            face_image = cv2.resize(face_image, (224, 224))
        
        # Normalize to [0, 1]
        face_image = face_image.astype(np.float32) / 255.0
        
        # Mean-variance normalization (ImageNet stats)
        mean = 0.485
        std = 0.229
        face_image = (face_image - mean) / std
        
        # Convert to tensor with batch dimension
        tensor = torch.from_numpy(face_image).unsqueeze(0).unsqueeze(0)  # [1, 1, 224, 224]
        
        return tensor
    
    def _assess_landmark_quality(self, landmarks: np.ndarray) -> float:
        """Assess quality of facial landmarks for confidence adjustment"""
        # Simple quality assessment based on landmark distribution
        # In production, this would be more sophisticated
        if len(landmarks) == 936:  # 468 * 2 coordinates
            # Check if landmarks are well-distributed
            x_coords = landmarks[::2]
            y_coords = landmarks[1::2]
            
            x_range = np.max(x_coords) - np.min(x_coords)
            y_range = np.max(y_coords) - np.min(y_coords)
            
            # Good landmarks should span reasonable face region
            if x_range > 0.3 and y_range > 0.3:
                return 0.8
            else:
                return 0.5
        
        return 0.3

class SpeechEmotionPipeline:
    """
    Speech emotion analysis pipeline using Librosa + Wav2Vec2/LSTM
    Processes WAV audio and outputs emotion classification
    """
    
    def __init__(self):
        self.sample_rate = 16000  # 16 kHz as specified
        self.frame_length = 25  # 25ms frames
        self.hop_length = 10   # 10ms hop
        self.n_mfcc = 13
        self.model = None
        self.processor = None
        self.emotion_classes = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
    async def initialize(self):
        """Initialize speech emotion model"""
        logger.info("Initializing speech emotion pipeline...")
        
        try:
            if settings.SPEECH_MODEL_PATH.exists():
                # Try to load Wav2Vec2 model
                try:
                    self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
                    self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                        str(settings.SPEECH_MODEL_PATH)
                    )
                    self.model.eval()
                    logger.info("Loaded Wav2Vec2 speech emotion model")
                except Exception:
                    # Fallback to PyTorch LSTM
                    self.model = torch.load(settings.SPEECH_MODEL_PATH, map_location='cpu')
                    self.model.eval()
                    logger.info("Loaded PyTorch LSTM speech emotion model")
            else:
                # Create demo model
                self.model = self._create_demo_model()
                logger.warning("Using demo speech emotion model")
                
        except Exception as e:
            logger.error(f"Error loading speech model: {e}")
            self.model = self._create_demo_model()
        
        logger.info("Speech emotion pipeline ready")
    
    def _create_demo_model(self):
        """Create a simple demo LSTM model"""
        import torch.nn as nn
        
        class SimpleLSTM(nn.Module):
            def __init__(self, input_size=39, hidden_size=128, num_classes=7):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, num_classes)
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                # Use last output
                out = self.fc(self.dropout(lstm_out[:, -1, :]))
                return F.softmax(out, dim=1)
        
        model = SimpleLSTM()
        model.eval()
        return model
    
    async def analyze(self, audio_b64: str) -> Tuple[str, float]:
        """
        Analyze speech emotion from base64 audio
        
        Returns:
            Tuple of (emotion_label, confidence_score)
        """
        start_time = time.time()
        
        try:
            # Decode and preprocess audio
            audio = self._decode_audio(audio_b64)
            if audio is None:
                return "neutral", 0.1
            
            # Run analysis in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._analyze_sync,
                audio
            )
            
            processing_time = (time.time() - start_time) * 1000
            if processing_time > 400:
                logger.warning(f"Speech analysis took {processing_time:.1f}ms (>400ms threshold)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in speech emotion analysis: {e}")
            return "neutral", 0.1
    
    def _decode_audio(self, audio_b64: str) -> Optional[np.ndarray]:
        """Decode base64 audio to numpy array at 16kHz"""
        try:
            if audio_b64.startswith('data:audio/'):
                audio_b64 = audio_b64.split(',')[1]
            
            audio_bytes = base64.b64decode(audio_b64)
            audio, sr = sf.read(io.BytesIO(audio_bytes))
            
            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample to 16kHz
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
            return audio
        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            return None
    
    def _analyze_sync(self, audio: np.ndarray) -> Tuple[str, float]:
        """Synchronous analysis method for thread pool execution"""
        # Extract MFCC features with deltas and delta-deltas (39 features total)
        features = self._extract_mfcc_features(audio)
        
        if self.processor is not None:
            # Use Wav2Vec2 model
            return self._analyze_with_wav2vec2(audio)
        else:
            # Use LSTM model with MFCC features
            return self._analyze_with_lstm(features)
    
    def _extract_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract 39 MFCC features (13 + 13 deltas + 13 delta-deltas)"""
        # Calculate frame parameters
        frame_length_samples = int(self.frame_length * self.sample_rate / 1000)
        hop_length_samples = int(self.hop_length * self.sample_rate / 1000)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=hop_length_samples,
            n_fft=frame_length_samples * 2
        )
        
        # Calculate deltas and delta-deltas
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Stack features (39 features per frame)
        features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
        
        # Transpose to (time_frames, features)
        features = features.T
        
        return features
    
    def _analyze_with_wav2vec2(self, audio: np.ndarray) -> Tuple[str, float]:
        """Analyze using Wav2Vec2 transformer model"""
        try:
            # Preprocess audio
            inputs = self.processor(
                audio,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True
            )
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
            
            # Get top emotion and confidence
            emotion_idx = torch.argmax(probabilities, dim=-1).item()
            emotion_label = self.emotion_classes[emotion_idx]
            confidence = float(probabilities[0, emotion_idx])
            
            return emotion_label, confidence
            
        except Exception as e:
            logger.error(f"Error in Wav2Vec2 analysis: {e}")
            return "neutral", 0.1
    
    def _analyze_with_lstm(self, features: np.ndarray) -> Tuple[str, float]:
        """Analyze using LSTM model with MFCC features"""
        try:
            # Convert to tensor
            features_tensor = torch.from_numpy(features).float().unsqueeze(0)  # Add batch dim
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = outputs.cpu().numpy()[0]
            
            # Get top emotion and confidence
            emotion_idx = np.argmax(probabilities)
            emotion_label = self.emotion_classes[emotion_idx]
            confidence = float(probabilities[emotion_idx])
            
            return emotion_label, confidence
            
        except Exception as e:
            logger.error(f"Error in LSTM analysis: {e}")
            return "neutral", 0.1

class MotionAnalysisPipeline:
    """
    Motion analysis pipeline using accelerometer/gyroscope data
    Outputs shakiness level classification
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.shakiness_levels = ['low', 'medium', 'high']
        self.min_samples = 20
        
    async def initialize(self):
        """Initialize motion analysis model"""
        logger.info("Initializing motion analysis pipeline...")
        
        try:
            if settings.MOTION_MODEL_PATH.exists():
                # Load scikit-learn model
                with open(settings.MOTION_MODEL_PATH, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data.get('model')
                    self.scaler = model_data.get('scaler')
                    logger.info("Loaded motion analysis model from pickle")
            else:
                # Create demo model
                self.model = self._create_demo_model()
                logger.warning("Using demo motion analysis model")
                
        except Exception as e:
            logger.error(f"Error loading motion model: {e}")
            self.model = self._create_demo_model()
        
        logger.info("Motion analysis pipeline ready")
    
    def _create_demo_model(self):
        """Create a simple demo RandomForest model"""
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Create dummy training data to fit the model
        np.random.seed(42)
        X_dummy = np.random.randn(100, 15)  # 15 features
        y_dummy = np.random.choice([0, 1, 2], 100)  # 3 classes
        
        model.fit(X_dummy, y_dummy)
        return model
    
    async def analyze(self, motion_data: List[Dict]) -> Tuple[str, float]:
        """
        Analyze motion shakiness from accelerometer/gyroscope data
        
        Args:
            motion_data: List of motion samples with x,y,z and timestamp
            
        Returns:
            Tuple of (shakiness_level, confidence_score)
            
        Raises:
            InvalidMotionDataError: If motion data is invalid or insufficient
        """
        try:
            # Validate motion data
            if not motion_data or len(motion_data) < self.min_samples:
                raise InvalidMotionDataError(
                    f"Insufficient motion data: {len(motion_data) if motion_data else 0} samples "
                    f"(minimum {self.min_samples} required)"
                )
            
            # Parse motion data
            accel_data, gyro_data, timestamps = self._parse_motion_data(motion_data)
            
            # Extract features
            features = self._extract_motion_features(accel_data, gyro_data, timestamps)
            
            # Run inference
            shakiness_level, confidence = self._predict_shakiness(features)
            
            return shakiness_level, confidence
            
        except InvalidMotionDataError:
            raise
        except Exception as e:
            logger.error(f"Error in motion analysis: {e}")
            return "medium", 0.1
    
    def _parse_motion_data(self, motion_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse motion data into numpy arrays"""
        accel_data = []
        gyro_data = []
        timestamps = []
        
        for sample in motion_data:
            # Extract timestamp
            timestamp = sample.get('timestamp', 0)
            timestamps.append(timestamp)
            
            # Extract accelerometer data
            if 'accelerometer' in sample:
                accel = sample['accelerometer']
                accel_data.append([accel.get('x', 0), accel.get('y', 0), accel.get('z', 0)])
            elif all(k in sample for k in ['x', 'y', 'z']):
                accel_data.append([sample['x'], sample['y'], sample['z']])
            else:
                raise InvalidMotionDataError("Missing accelerometer data in motion sample")
            
            # Extract gyroscope data (optional)
            if 'gyroscope' in sample:
                gyro = sample['gyroscope']
                gyro_data.append([gyro.get('x', 0), gyro.get('y', 0), gyro.get('z', 0)])
            elif all(k in sample for k in ['gx', 'gy', 'gz']):
                gyro_data.append([sample['gx'], sample['gy'], sample['gz']])
            else:
                # Default to zeros if no gyroscope data
                gyro_data.append([0.0, 0.0, 0.0])
        
        return (
            np.array(accel_data, dtype=np.float32),
            np.array(gyro_data, dtype=np.float32),
            np.array(timestamps, dtype=np.float64)
        )
    
    def _extract_motion_features(self, accel_data: np.ndarray, gyro_data: np.ndarray, 
                                timestamps: np.ndarray) -> np.ndarray:
        """Extract comprehensive motion features for classification"""
        features = []
        
        # Accelerometer magnitude
        accel_magnitude = np.linalg.norm(accel_data, axis=1)
        
        # Statistical features for accelerometer
        features.extend([
            np.mean(accel_magnitude),
            np.std(accel_magnitude),
            np.max(accel_magnitude),
            np.min(accel_magnitude),
            np.percentile(accel_magnitude, 75) - np.percentile(accel_magnitude, 25)  # IQR
        ])
        
        # Gyroscope magnitude
        gyro_magnitude = np.linalg.norm(gyro_data, axis=1)
        
        # Statistical features for gyroscope
        features.extend([
            np.mean(gyro_magnitude),
            np.std(gyro_magnitude),
            np.max(gyro_magnitude),
            np.min(gyro_magnitude),
            np.percentile(gyro_magnitude, 75) - np.percentile(gyro_magnitude, 25)  # IQR
        ])
        
        # Spectral energy using FFT
        if len(accel_magnitude) >= 8:
            fft_accel = fft(accel_magnitude)
            spectral_energy = np.sum(np.abs(fft_accel) ** 2)
            features.append(spectral_energy)
        else:
            features.append(0.0)
        
        # Zero-crossing rate for each accelerometer axis
        for axis in range(3):
            zero_crossings = np.sum(np.diff(np.sign(accel_data[:, axis])) != 0)
            features.append(zero_crossings / len(accel_data))
        
        # Jerk (derivative of acceleration)
        jerk_magnitude = np.linalg.norm(np.diff(accel_data, axis=0), axis=1)
        features.extend([
            np.mean(jerk_magnitude),
            np.std(jerk_magnitude)
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _predict_shakiness(self, features: np.ndarray) -> Tuple[str, float]:
        """Predict shakiness level using the trained model"""
        try:
            # Reshape for sklearn
            features = features.reshape(1, -1)
            
            # Scale features if scaler available
            if self.scaler is not None:
                features = self.scaler.transform(features)
            
            # Predict
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)[0]
                prediction_idx = np.argmax(probabilities)
                confidence = float(probabilities[prediction_idx])
            else:
                prediction_idx = self.model.predict(features)[0]
                confidence = 0.7  # Default confidence for non-probabilistic models
            
            shakiness_level = self.shakiness_levels[prediction_idx]
            
            return shakiness_level, confidence
            
        except Exception as e:
            logger.error(f"Error in shakiness prediction: {e}")
            return "medium", 0.1

class CognitiveFusionEngine:
    """
    Fusion engine that combines facial, speech, and motion analysis results
    """
    
    def __init__(self):
        self.fusion_model = None
        self.suggestions_mapping = {}
        
    async def initialize(self):
        """Initialize fusion model and load suggestions mapping"""
        logger.info("Initializing cognitive fusion engine...")
        
        try:
            # Load fusion model
            if settings.FUSION_MODEL_PATH.exists():
                self.fusion_model = torch.load(settings.FUSION_MODEL_PATH, map_location='cpu')
                self.fusion_model.eval()
                logger.info("Loaded neural fusion model")
            else:
                logger.info("Using rule-based fusion (no fusion model found)")
            
            # Load suggestions mapping
            suggestions_file = Path("suggestions_by_state.json")
            if suggestions_file.exists():
                with open(suggestions_file, 'r') as f:
                    self.suggestions_mapping = json.load(f)
            else:
                self._create_default_suggestions()
                
        except Exception as e:
            logger.error(f"Error initializing fusion engine: {e}")
            self._create_default_suggestions()
        
        logger.info("Cognitive fusion engine ready")
    
    def _create_default_suggestions(self):
        """Create default suggestions mapping"""
        self.suggestions_mapping = {
            "calm": [
                "Continue with your current activities",
                "Take a moment to appreciate your calm state",
                "Consider engaging in creative activities"
            ],
            "stressed": [
                "Take deep breaths for 2-3 minutes",
                "Consider a short break or walk",
                "Try progressive muscle relaxation",
                "Limit caffeine intake"
            ],
            "anxious": [
                "Practice grounding techniques (5-4-3-2-1 method)",
                "Try breathing exercises",
                "Engage in mindfulness meditation",
                "Consider speaking with someone you trust"
            ],
            "happy": [
                "Share your positive energy with others",
                "Take note of what's making you happy",
                "Engage in activities you enjoy"
            ],
            "sad": [
                "Allow yourself to feel these emotions",
                "Consider gentle physical activity",
                "Reach out to supportive friends or family",
                "Practice self-compassion"
            ],
            "angry": [
                "Take a few deep breaths before responding",
                "Consider the root cause of your anger",
                "Try physical exercise to release tension",
                "Practice counting to ten before reacting"
            ],
            "neutral": [
                "Monitor your emotional state throughout the day",
                "Consider activities that bring you joy",
                "Stay hydrated and maintain good posture"
            ]
        }
    
    def fuse_results(self, face_emotion: str, face_confidence: float,
                    speech_emotion: str, speech_confidence: float,
                    motion_level: str, motion_confidence: float) -> Dict[str, Any]:
        """
        Fuse multimodal results into final cognitive state
        
        Returns:
            Dictionary with fused cognitive state and metadata
        """
        try:
            if self.fusion_model is not None:
                return self._neural_fusion(
                    face_emotion, face_confidence,
                    speech_emotion, speech_confidence,
                    motion_level, motion_confidence
                )
            else:
                return self._rule_based_fusion(
                    face_emotion, face_confidence,
                    speech_emotion, speech_confidence,
                    motion_level, motion_confidence
                )
        except Exception as e:
            logger.error(f"Error in fusion: {e}")
            return self._create_default_result()
    
    def _neural_fusion(self, face_emotion: str, face_confidence: float,
                      speech_emotion: str, speech_confidence: float,
                      motion_level: str, motion_confidence: float) -> Dict[str, Any]:
        """Neural network-based fusion using MLP model"""
        try:
            # Create one-hot vectors for emotions
            emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            motion_classes = ['low', 'medium', 'high']
            
            face_onehot = [1.0 if face_emotion == cls else 0.0 for cls in emotion_classes]
            speech_onehot = [1.0 if speech_emotion == cls else 0.0 for cls in emotion_classes]
            motion_onehot = [1.0 if motion_level == cls else 0.0 for cls in motion_classes]
            
            # Create input vector (one-hot + confidences)
            input_vector = face_onehot + speech_onehot + motion_onehot + [
                face_confidence, speech_confidence, motion_confidence
            ]
            
            # Run neural fusion
            input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                output = self.fusion_model(input_tensor)
                cognitive_probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
            
            # Map to cognitive states
            cognitive_states = ['calm', 'stressed', 'anxious', 'happy', 'sad', 'angry', 'neutral']
            state_idx = np.argmax(cognitive_probabilities)
            cognitive_state = cognitive_states[state_idx]
            fused_confidence = float(cognitive_probabilities[state_idx])
            
            return self._create_fusion_result(
                cognitive_state, fused_confidence,
                face_emotion, face_confidence,
                speech_emotion, speech_confidence,
                motion_level, motion_confidence
            )
            
        except Exception as e:
            logger.error(f"Error in neural fusion: {e}")
            return self._rule_based_fusion(
                face_emotion, face_confidence,
                speech_emotion, speech_confidence,
                motion_level, motion_confidence
            )
    
    def _rule_based_fusion(self, face_emotion: str, face_confidence: float,
                          speech_emotion: str, speech_confidence: float,
                          motion_level: str, motion_confidence: float) -> Dict[str, Any]:
        """Rule-based fusion logic"""
        # Weight down low-confidence modalities
        effective_confidences = []
        weighted_emotions = []
        
        if face_confidence >= 0.5:
            effective_confidences.append(face_confidence)
            weighted_emotions.append((face_emotion, face_confidence))
        
        if speech_confidence >= 0.5:
            effective_confidences.append(speech_confidence)
            weighted_emotions.append((speech_emotion, speech_confidence))
        
        # Convert motion level to emotion-like state
        motion_emotion = self._motion_to_emotion(motion_level)
        if motion_confidence >= 0.5:
            effective_confidences.append(motion_confidence)
            weighted_emotions.append((motion_emotion, motion_confidence))
        
        # Determine cognitive state
        if len(weighted_emotions) == 0:
            cognitive_state = "neutral"
            fused_confidence = 0.1
        elif len(weighted_emotions) == 1:
            cognitive_state = self._emotion_to_cognitive_state(weighted_emotions[0][0])
            fused_confidence = weighted_emotions[0][1]
        else:
            # Check for agreement
            emotion_votes = {}
            for emotion, confidence in weighted_emotions:
                cognitive = self._emotion_to_cognitive_state(emotion)
                if cognitive not in emotion_votes:
                    emotion_votes[cognitive] = []
                emotion_votes[cognitive].append(confidence)
            
            # Find consensus
            best_state = None
            best_score = 0
            
            for state, confidences in emotion_votes.items():
                if len(confidences) >= 2:  # At least 2 modalities agree
                    score = np.mean(confidences)
                    if score > best_score:
                        best_score = score
                        best_state = state
            
            if best_state is not None:
                cognitive_state = best_state
                fused_confidence = best_score
            else:
                # No agreement, use highest confidence
                best_emotion, best_conf = max(weighted_emotions, key=lambda x: x[1])
                cognitive_state = self._emotion_to_cognitive_state(best_emotion)
                fused_confidence = best_conf
        
        # Calculate harmonic mean of confidences
        if effective_confidences:
            harmonic_mean = len(effective_confidences) / sum(1/c for c in effective_confidences)
        else:
            harmonic_mean = 0.1
        
        return self._create_fusion_result(
            cognitive_state, harmonic_mean,
            face_emotion, face_confidence,
            speech_emotion, speech_confidence,
            motion_level, motion_confidence
        )
    
    def _motion_to_emotion(self, motion_level: str) -> str:
        """Convert motion shakiness level to emotion-like state"""
        mapping = {
            'low': 'calm',
            'medium': 'neutral',
            'high': 'anxious'
        }
        return mapping.get(motion_level, 'neutral')
    
    def _emotion_to_cognitive_state(self, emotion: str) -> str:
        """Convert emotion to cognitive state"""
        mapping = {
            'happy': 'happy',
            'sad': 'sad',
            'angry': 'angry',
            'fear': 'anxious',
            'fearful': 'anxious',
            'anxious': 'anxious',
            'surprise': 'neutral',
            'surprised': 'neutral',
            'disgust': 'angry',
            'disgusted': 'angry',
            'neutral': 'neutral',
            'calm': 'calm'
        }
        return mapping.get(emotion, 'neutral')
    
    def _create_fusion_result(self, cognitive_state: str, fused_confidence: float,
                             face_emotion: str, face_confidence: float,
                             speech_emotion: str, speech_confidence: float,
                             motion_level: str, motion_confidence: float) -> Dict[str, Any]:
        """Create structured fusion result"""
        # Generate suggestions
        suggestions = self.suggestions_mapping.get(cognitive_state, [
            "Monitor your emotional state",
            "Practice mindfulness",
            "Take care of your physical and mental health"
        ])[:5]  # Limit to 5 suggestions
        
        # Create graph data for visualization
        graph_data = {
            'timestamp': time.time(),
            'modalities': {
                'face': {'emotion': face_emotion, 'confidence': face_confidence},
                'speech': {'emotion': speech_emotion, 'confidence': speech_confidence},
                'motion': {'level': motion_level, 'confidence': motion_confidence}
            },
            'cognitive_state': cognitive_state,
            'confidence_timeline': [
                {'timestamp': time.time(), 'face': face_confidence, 
                 'speech': speech_confidence, 'motion': motion_confidence}
            ]
        }
        
        # Calculate valence and arousal
        valence, arousal = self._calculate_valence_arousal(cognitive_state)
        
        return {
            'cognitive_state': cognitive_state,
            'confidence': fused_confidence,
            'valence': valence,
            'arousal': arousal,
            'attention': self._calculate_attention(motion_level, motion_confidence),
            'stress_level': self._calculate_stress_level(cognitive_state),
            'face_emotion': face_emotion,
            'face_confidence': face_confidence,
            'speech_emotion': speech_emotion,
            'speech_confidence': speech_confidence,
            'motion_level': motion_level,
            'motion_confidence': motion_confidence,
            'suggestions': suggestions,
            'graph_data': graph_data
        }
    
    def _calculate_valence_arousal(self, cognitive_state: str) -> Tuple[float, float]:
        """Calculate valence (positive/negative) and arousal (energy) scores"""
        mapping = {
            'happy': (0.8, 0.7),
            'calm': (0.6, 0.3),
            'neutral': (0.5, 0.5),
            'sad': (0.2, 0.3),
            'angry': (0.2, 0.8),
            'anxious': (0.3, 0.7),
            'stressed': (0.3, 0.8)
        }
        return mapping.get(cognitive_state, (0.5, 0.5))
    
    def _calculate_attention(self, motion_level: str, motion_confidence: float) -> float:
        """Calculate attention score based on motion stability"""
        if motion_level == 'low':
            return 0.8 * motion_confidence + 0.2 * 0.7
        elif motion_level == 'medium':
            return 0.5 * motion_confidence + 0.5 * 0.5
        else:  # high
            return 0.3 * motion_confidence + 0.7 * 0.3
    
    def _calculate_stress_level(self, cognitive_state: str) -> float:
        """Calculate stress level from cognitive state"""
        mapping = {
            'calm': 0.1,
            'happy': 0.2,
            'neutral': 0.4,
            'sad': 0.6,
            'anxious': 0.8,
            'angry': 0.9,
            'stressed': 1.0
        }
        return mapping.get(cognitive_state, 0.5)
    
    def _create_default_result(self) -> Dict[str, Any]:
        """Create default result for error cases"""
        return {
            'cognitive_state': 'neutral',
            'confidence': 0.1,
            'valence': 0.5,
            'arousal': 0.5,
            'attention': 0.5,
            'stress_level': 0.5,
            'face_emotion': 'neutral',
            'face_confidence': 0.1,
            'speech_emotion': 'neutral',
            'speech_confidence': 0.1,
            'motion_level': 'medium',
            'motion_confidence': 0.1,
            'suggestions': ["System error - please try again"],
            'graph_data': None
        }

class MLAnalysisEngine:
    """
    Main ML Analysis Engine coordinating all pipelines
    """
    
    def __init__(self):
        self.facial_pipeline = FacialEmotionPipeline()
        self.speech_pipeline = SpeechEmotionPipeline()
        self.motion_pipeline = MotionAnalysisPipeline()
        self.fusion_engine = CognitiveFusionEngine()
        self.is_ready = False
        
    async def initialize(self):
        """Initialize all pipelines"""
        logger.info("Initializing ML Analysis Engine...")
        
        try:
            await self.facial_pipeline.initialize()
            await self.speech_pipeline.initialize()
            await self.motion_pipeline.initialize()
            await self.fusion_engine.initialize()
            
            self.is_ready = True
            logger.info("ML Analysis Engine fully initialized")
            
        except Exception as e:
            logger.error(f"Error initializing ML engine: {e}")
            self.is_ready = False
            raise
    
    def analyze_multimodal(self, face_image_b64: Optional[str], 
                          audio_clip_b64: Optional[str],
                          motion_data: Optional[List[Dict]]) -> Dict[str, Any]:
        """
        Analyze multimodal input and return fused cognitive state
        
        This method runs synchronously in a thread pool to avoid blocking
        """
        start_time = time.time()
        results = {}
        
        try:
            # Initialize default values
            face_emotion, face_confidence = "neutral", 0.1
            speech_emotion, speech_confidence = "neutral", 0.1
            motion_level, motion_confidence = "medium", 0.1
            
            # Analyze face emotion (synchronous in thread pool)
            if face_image_b64:
                try:
                    face_emotion, face_confidence = asyncio.run(
                        self.facial_pipeline.analyze(face_image_b64)
                    )
                except Exception as e:
                    logger.error(f"Face analysis error: {e}")
            
            # Analyze speech emotion (synchronous in thread pool)
            if audio_clip_b64:
                try:
                    speech_emotion, speech_confidence = asyncio.run(
                        self.speech_pipeline.analyze(audio_clip_b64)
                    )
                except Exception as e:
                    logger.error(f"Speech analysis error: {e}")
            
            # Analyze motion (synchronous)
            if motion_data:
                try:
                    motion_level, motion_confidence = asyncio.run(
                        self.motion_pipeline.analyze(motion_data)
                    )
                except InvalidMotionDataError:
                    raise  # Re-raise for proper HTTP 422 handling
                except Exception as e:
                    logger.error(f"Motion analysis error: {e}")
            
            # Fuse results
            fusion_result = self.fusion_engine.fuse_results(
                face_emotion, face_confidence,
                speech_emotion, speech_confidence,
                motion_level, motion_confidence
            )
            
            # Add processing time
            processing_time = (time.time() - start_time) * 1000
            fusion_result['processing_time_ms'] = processing_time
            
            return fusion_result
            
        except Exception as e:
            logger.error(f"Error in multimodal analysis: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up ML Analysis Engine...")
        
        if hasattr(self.facial_pipeline, 'executor'):
            self.facial_pipeline.executor.shutdown(wait=True)
        
        if hasattr(self.speech_pipeline, 'executor'):
            self.speech_pipeline.executor.shutdown(wait=True)
        
        logger.info("ML Analysis Engine cleanup complete")
