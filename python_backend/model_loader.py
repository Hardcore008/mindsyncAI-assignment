"""
Model Loader - Download and manage pre-trained models for production use
"""

import os
import requests
import pickle
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging
import json
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages downloading and loading of pre-trained models"""
    
    def __init__(self):
        self.models_dir = Path(__file__).parent / "ml_models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            "emotion_cnn": {
                "file": "emotion_cnn_model.h5",
                "type": "tensorflow",
                "url": None,  # Will create if doesn't exist
                "input_shape": (48, 48, 1),
                "classes": ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            },
            "audio_emotion": {
                "file": "audio_emotion_model.pkl",
                "type": "sklearn",
                "url": None,
                "features": 40,  # MFCC + spectral features
                "classes": ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']
            },
            "motion_stress": {
                "file": "motion_stress_model.pkl", 
                "type": "sklearn",
                "url": None,
                "features": 15,  # Motion statistical features
                "classes": ['low_stress', 'medium_stress', 'high_stress']
            }
        }
    
    def create_emotion_cnn_model(self) -> tf.keras.Model:
        """Create a CNN model for emotion recognition from facial expressions"""
        try:
            model = tf.keras.Sequential([
                # Input layer - 48x48 grayscale images
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Dropout(0.25),
                
                # Second convolutional block
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Dropout(0.25),
                
                # Third convolutional block
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Dropout(0.25),
                
                # Fourth convolutional block
                tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Dropout(0.25),
                
                # Flatten and dense layers
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(7, activation='softmax')  # 7 emotions
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Initialize with random weights but good architecture
            logger.info("Created emotion CNN model with random weights")
            return model
            
        except Exception as e:
            logger.error(f"Error creating emotion CNN model: {e}")
            return None
    
    def create_audio_emotion_model(self) -> Dict[str, Any]:
        """Create a trained audio emotion model using Random Forest"""
        try:
            # Create and train a Random Forest model with synthetic data for demonstration
            # In production, this would be trained on a real dataset like RAVDESS
            
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            scaler = StandardScaler()
            
            # Generate synthetic training data for demonstration
            # Features: 13 MFCCs + 12 Chroma + 7 spectral contrast + 6 tonnetz + 2 tempo features = 40 features
            n_samples = 1000
            n_features = 40
            n_classes = 7
            
            X_synthetic = np.random.randn(n_samples, n_features)
            
            # Create somewhat realistic feature distributions
            # MFCCs (0-12): typically between -50 and 50
            X_synthetic[:, 0:13] = X_synthetic[:, 0:13] * 20
            # Chroma (13-24): typically between 0 and 1
            X_synthetic[:, 13:25] = np.abs(X_synthetic[:, 13:25] * 0.3)
            # Spectral contrast (25-31): typically between 0 and 100
            X_synthetic[:, 25:32] = np.abs(X_synthetic[:, 25:32] * 30)
            # Tonnetz (32-37): typically between -1 and 1
            X_synthetic[:, 32:38] = X_synthetic[:, 32:38] * 0.5
            # Tempo features (38-39): typically between 60 and 200 BPM
            X_synthetic[:, 38:40] = np.abs(X_synthetic[:, 38:40] * 50) + 120
            
            # Create labels with some structure
            y_synthetic = np.random.randint(0, n_classes, n_samples)
            
            # Add some feature-label relationships for realism
            # Happy emotions tend to have higher tempo and certain spectral characteristics
            happy_mask = y_synthetic == 1  # happy
            X_synthetic[happy_mask, 38] += 20  # Higher tempo
            X_synthetic[happy_mask, 25:32] += 10  # Higher spectral contrast
            
            # Sad emotions tend to have lower tempo and energy
            sad_mask = y_synthetic == 2  # sad
            X_synthetic[sad_mask, 38] -= 20  # Lower tempo
            X_synthetic[sad_mask, 25:32] -= 10  # Lower spectral contrast
            
            # Fit scaler and model
            X_scaled = scaler.fit_transform(X_synthetic)
            model.fit(X_scaled, y_synthetic)
            
            logger.info(f"Created audio emotion model with {n_features} features and {n_classes} classes")
            
            return {
                'model': model,
                'scaler': scaler,
                'feature_names': [
                    f'mfcc_{i}' for i in range(13)
                ] + [
                    f'chroma_{i}' for i in range(12)
                ] + [
                    f'spectral_contrast_{i}' for i in range(7)
                ] + [
                    f'tonnetz_{i}' for i in range(6)
                ] + ['tempo', 'tempo_std'],
                'classes': ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']
            }
            
        except Exception as e:
            logger.error(f"Error creating audio emotion model: {e}")
            return None
    
    def create_motion_stress_model(self) -> Dict[str, Any]:
        """Create a motion-based stress detection model"""
        try:
            # Create SVM model for stress detection
            model = SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,  # Enable probability estimates
                random_state=42
            )
            
            scaler = StandardScaler()
            
            # Generate synthetic training data for motion stress detection
            n_samples = 800
            n_features = 15  # Statistical features from accelerometer and gyroscope
            n_classes = 3  # low, medium, high stress
            
            X_synthetic = np.random.randn(n_samples, n_features)
            
            # Feature engineering for realistic motion data:
            # Features 0-4: Accelerometer statistics (mean, std, max, min, range)
            # Features 5-9: Gyroscope statistics (mean, std, max, min, range)
            # Features 10-14: Combined metrics (jerk, frequency features, etc.)
            
            # Accelerometer features (m/s²)
            X_synthetic[:, 0:5] = np.abs(X_synthetic[:, 0:5] * 3) + 9.81  # Around gravity
            
            # Gyroscope features (rad/s)
            X_synthetic[:, 5:10] = X_synthetic[:, 5:10] * 0.5
            
            # Combined features
            X_synthetic[:, 10:15] = np.abs(X_synthetic[:, 10:15] * 2)
            
            # Create labels with relationships to features
            y_synthetic = np.random.randint(0, n_classes, n_samples)
            
            # High stress tends to have higher variance and jerk
            high_stress_mask = y_synthetic == 2
            X_synthetic[high_stress_mask, 1] += 2  # Higher accel std
            X_synthetic[high_stress_mask, 6] += 1  # Higher gyro std
            X_synthetic[high_stress_mask, 10:15] += 1  # Higher combined metrics
            
            # Low stress tends to have lower variance
            low_stress_mask = y_synthetic == 0
            X_synthetic[low_stress_mask, 1] -= 1  # Lower accel std
            X_synthetic[low_stress_mask, 6] -= 0.5  # Lower gyro std
            X_synthetic[low_stress_mask, 10:15] -= 0.5  # Lower combined metrics
            
            # Fit scaler and model
            X_scaled = scaler.fit_transform(X_synthetic)
            model.fit(X_scaled, y_synthetic)
            
            logger.info(f"Created motion stress model with {n_features} features and {n_classes} classes")
            
            return {
                'model': model,
                'scaler': scaler,
                'feature_names': [
                    'accel_mean', 'accel_std', 'accel_max', 'accel_min', 'accel_range',
                    'gyro_mean', 'gyro_std', 'gyro_max', 'gyro_min', 'gyro_range',
                    'jerk_magnitude', 'motion_energy', 'stability_index', 'freq_peak', 'freq_energy'
                ],
                'classes': ['low_stress', 'medium_stress', 'high_stress']
            }
            
        except Exception as e:
            logger.error(f"Error creating motion stress model: {e}")
            return None
    
    def save_model(self, model_name: str, model_data: Any) -> bool:
        """Save a model to disk"""
        try:
            config = self.model_configs.get(model_name)
            if not config:
                logger.error(f"Unknown model: {model_name}")
                return False
            
            model_path = self.models_dir / config["file"]
            
            if config["type"] == "tensorflow":
                model_data.save(model_path)
            elif config["type"] == "sklearn":
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
            
            logger.info(f"Saved {model_name} model to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            return False
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """Load a model from disk"""
        try:
            config = self.model_configs.get(model_name)
            if not config:
                logger.error(f"Unknown model: {model_name}")
                return None
            
            model_path = self.models_dir / config["file"]
            
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                return None
            
            if config["type"] == "tensorflow":
                return tf.keras.models.load_model(model_path)
            elif config["type"] == "sklearn":
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def initialize_all_models(self) -> Dict[str, bool]:
        """Initialize all models, creating them if they don't exist"""
        results = {}
        
        # Emotion CNN Model
        emotion_model = self.load_model("emotion_cnn")
        if emotion_model is None:
            logger.info("Creating new emotion CNN model...")
            emotion_model = self.create_emotion_cnn_model()
            if emotion_model:
                results["emotion_cnn"] = self.save_model("emotion_cnn", emotion_model)
            else:
                results["emotion_cnn"] = False
        else:
            results["emotion_cnn"] = True
            logger.info("Loaded existing emotion CNN model")
        
        # Audio Emotion Model
        audio_model = self.load_model("audio_emotion")
        if audio_model is None:
            logger.info("Creating new audio emotion model...")
            audio_model_data = self.create_audio_emotion_model()
            if audio_model_data:
                results["audio_emotion"] = self.save_model("audio_emotion", audio_model_data)
            else:
                results["audio_emotion"] = False
        else:
            results["audio_emotion"] = True
            logger.info("Loaded existing audio emotion model")
        
        # Motion Stress Model
        motion_model = self.load_model("motion_stress")
        if motion_model is None:
            logger.info("Creating new motion stress model...")
            motion_model_data = self.create_motion_stress_model()
            if motion_model_data:
                results["motion_stress"] = self.save_model("motion_stress", motion_model_data)
            else:
                results["motion_stress"] = False
        else:
            results["motion_stress"] = True
            logger.info("Loaded existing motion stress model")
        
        return results

# Global model manager instance
model_manager = ModelManager()
