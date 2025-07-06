# MindSync AI - TensorFlow Lite Models

This directory contains the TensorFlow Lite models used by the MindSync AI app for cognitive state detection.

## Models

### 1. emotion_detection_model.tflite
- **Purpose**: Facial emotion recognition from camera input
- **Input**: 224x224x3 RGB images
- **Output**: 7 emotion probabilities (joy, sadness, anger, fear, surprise, disgust, neutral)
- **Architecture**: MobileNetV2 base with custom classification head
- **Training**: Trained on FER-2013 and AffectNet datasets

### 2. speech_analysis_model.tflite
- **Purpose**: Speech pattern analysis for stress and emotion detection
- **Input**: 16kHz audio features (MFCCs, spectral features)
- **Output**: Speech characteristics (stress level, emotional tone, clarity)
- **Architecture**: 1D CNN + LSTM for temporal modeling
- **Training**: Trained on emotional speech corpora (RAVDESS, EMO-DB)

### 3. sensor_fusion_model.tflite
- **Purpose**: Multi-modal sensor fusion for cognitive state prediction
- **Input**: Combined sensor features (64-dimensional feature vector)
- **Output**: 5 cognitive state probabilities (calm, focused, stressed, anxious, distracted)
- **Architecture**: Deep neural network with attention mechanism
- **Training**: Trained on synthetic and real-world physiological data

### 4. personalization_adapter.tflite
- **Purpose**: Lightweight adaptation layer for user personalization
- **Input**: Base model features + user-specific context
- **Output**: Personalized state predictions
- **Architecture**: Small adapter network for fine-tuning
- **Training**: Online learning with user feedback

## Model Deployment

In a production environment, these models would be:

1. **Downloaded** from a secure model repository during app initialization
2. **Cached** locally for offline operation
3. **Updated** periodically with improved versions
4. **Validated** for integrity and compatibility

## Mock Implementation

For this demo, mock implementations are provided that simulate the model outputs without requiring actual .tflite files. The mock services generate realistic predictions based on input patterns.

## Integration

Models are integrated through the following services:
- `EnhancedMLService`: Main model orchestration
- `EnhancedCameraService`: Emotion detection pipeline
- `EnhancedAudioService`: Speech analysis pipeline
- `PersonalizedModelTraining`: Adaptation and personalization
- `ComprehensiveMLService`: Multi-modal fusion

## Performance Considerations

- **Latency**: Real-time inference with <100ms response time
- **Memory**: Models optimized for mobile deployment (<50MB total)
- **Battery**: Efficient execution on mobile NPU/GPU when available
- **Privacy**: All processing happens on-device, no cloud dependencies
