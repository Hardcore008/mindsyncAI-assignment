# ML Models Directory

This directory contains machine learning models for cognitive state analysis.

## Current Implementation

The current implementation uses real-time feature extraction and lightweight algorithms that don't require large pre-trained models:

1. **Emotion Detection**: Uses MediaPipe face mesh and OpenCV for facial feature extraction
2. **Audio Analysis**: Uses librosa for real-time audio feature extraction
3. **Motion Analysis**: Uses signal processing techniques for motion pattern analysis

## Future Enhancements

For production deployment, you may want to add:

1. **Pre-trained Models**:
   - Emotion recognition CNN models (e.g., FER2013 trained models)
   - Audio emotion models (e.g., RAVDESS trained models)
   - Motion activity classification models

2. **Model Files** (would be stored here):
   - `emotion_cnn_model.h5` - Trained emotion detection CNN
   - `audio_emotion_model.pkl` - Audio-based emotion classifier
   - `stress_detection_model.pkl` - Stress level prediction model
   - `attention_model.pkl` - Attention score prediction model

## Loading Custom Models

To use custom models, update the ML analysis classes in `ml_analysis.py`:

```python
# Example of loading a custom model
from tensorflow.keras.models import load_model

class EmotionDetector:
    def __init__(self):
        self.model = load_model('ml_models/emotion_cnn_model.h5')
```

## Model Requirements

All models should:
- Have sub-2s inference time
- Be optimized for CPU inference (or GPU if available)
- Use consistent input/output formats
- Include proper error handling
