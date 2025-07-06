# MindSync AI Backend

A production-ready FastAPI backend for real-time cognitive state analysis using multi-modal machine learning.

## Features

- **Real-time ML Analysis**: Face emotion detection, audio stress analysis, and motion pattern recognition
- **Multi-modal Fusion**: Combines results from multiple sensors for comprehensive cognitive assessment
- **Privacy-Compliant**: No raw sensor data stored, only extracted features and analysis results
- **RESTful API**: FastAPI with automatic OpenAPI documentation
- **Database Integration**: SQLite with SQLAlchemy ORM for session storage
- **Performance Monitoring**: Built-in metrics and health monitoring
- **Scalable Architecture**: Modular design with separation of concerns

## Architecture

```
python_backend/
├── enhanced_ml_main.py      # Main FastAPI application
├── database.py              # Database configuration and management
├── models.py                # SQLAlchemy database models
├── schemas.py               # Pydantic validation schemas
├── ml_analysis.py           # Machine learning processing engine
├── utils.py                 # Utility functions and helpers
├── enhanced_requirements.txt # Python dependencies
├── start_enhanced_backend.*  # Startup scripts
├── data/                    # Database and data files
└── ml_models/               # ML model files (optional)
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Pip package manager
- Windows PowerShell (for .ps1 script) or Command Prompt (for .bat script)

### Installation & Running

1. **Using PowerShell (Recommended):**
   ```powershell
   .\start_enhanced_backend.ps1
   ```

2. **Using Command Prompt:**
   ```cmd
   start_enhanced_backend.bat
   ```

3. **Manual Setup:**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   
   # Install dependencies
   pip install -r enhanced_requirements.txt
   
   # Start server
   python -m uvicorn enhanced_ml_main:app --host 0.0.0.0 --port 8000 --reload
   ```

The server will be available at:
- **API Endpoint**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## API Endpoints

### Core Analysis

- **POST `/analyze`** - Analyze face, audio, and motion data
- **GET `/sessions`** - Retrieve session history
- **GET `/session/{id}`** - Get specific session details

### System

- **GET `/health`** - Health check and system status
- **GET `/stats`** - Performance statistics
- **GET `/`** - API information

### User Management

- **POST `/preferences`** - Update user preferences
- **GET `/preferences/{user_id}`** - Get user preferences

## Request Format

### Analysis Request

```json
{
  "session_id": "optional-uuid",
  "user_id": "optional-user-id",
  "face_image": "base64-encoded-image",
  "audio_data": "base64-encoded-audio",
  "motion_data": [
    {
      "timestamp": 1640995200.0,
      "acceleration": {"x": 0.1, "y": 0.2, "z": 9.8},
      "gyroscope": {"x": 0.01, "y": 0.02, "z": 0.03}
    }
  ],
  "timestamp": "2023-12-31T12:00:00Z",
  "duration_seconds": 30.0
}
```

### Analysis Response

```json
{
  "session_id": "uuid",
  "timestamp": "2023-12-31T12:00:00Z",
  "status": "completed",
  "emotion_analysis": {
    "dominant_emotion": "happy",
    "emotions": {
      "happy": 0.7,
      "neutral": 0.2,
      "sad": 0.1
    },
    "confidence": 0.85,
    "valence": 0.6,
    "arousal": 0.4
  },
  "stress_level": 0.3,
  "attention_score": 0.8,
  "overall_score": 0.75,
  "quality_score": 0.9,
  "processing_time_ms": 1250,
  "recommendations": [
    "Maintain good hydration throughout the day",
    "Take regular breaks from screen time"
  ]
}
```

## Machine Learning Models

### Emotion Detection
- **Technology**: MediaPipe + OpenCV + Custom algorithms
- **Input**: Base64-encoded facial images
- **Output**: 7 emotion categories with confidence scores
- **Features**: Facial landmark extraction, geometric analysis

### Audio Analysis
- **Technology**: Librosa + Signal processing
- **Input**: Base64-encoded audio (WAV/MP3)
- **Output**: Stress level and audio quality metrics
- **Features**: MFCC, spectral features, energy analysis

### Motion Analysis
- **Technology**: NumPy + Signal processing
- **Input**: Accelerometer and gyroscope data
- **Output**: Attention score and activity level
- **Features**: Statistical analysis, frequency domain features

## Database Schema

### Sessions Table
- `id` - Unique session identifier
- `user_id` - Optional user identifier
- `timestamp` - Session timestamp
- `emotion_analysis` - JSON emotion results
- `stress_level` - Calculated stress (0-1)
- `attention_score` - Attention level (0-1)
- `overall_score` - Overall wellness (0-1)
- `quality_score` - Data quality (0-1)
- `status` - Processing status

### User Preferences
- Privacy settings and analysis preferences
- Data retention policies
- Feature toggles

### Analysis Metrics
- Performance monitoring data
- Quality metrics
- System resource usage

## Privacy & Security

- **No Raw Data Storage**: Only features and analysis results are stored
- **Data Retention**: Configurable automatic cleanup
- **User Control**: Privacy preferences per user
- **Secure Processing**: In-memory processing with immediate cleanup
- **Compliance Ready**: GDPR/privacy-compliant design

## Performance

- **Target Latency**: Sub-2 second analysis
- **Concurrent Sessions**: Supports multiple simultaneous analyses
- **Memory Efficient**: Streaming processing, minimal memory footprint
- **Monitoring**: Built-in performance tracking

## Configuration

### Environment Variables
- `PYTHONPATH` - Set automatically by startup scripts
- `DATABASE_URL` - SQLite database path (optional)
- `LOG_LEVEL` - Logging level (INFO, DEBUG, WARNING, ERROR)

### Model Configuration
- Models are loaded at startup for optimal performance
- Custom models can be added to `ml_models/` directory
- Model parameters can be adjusted in `ml_analysis.py`

## Development

### Project Structure
```
├── enhanced_ml_main.py      # FastAPI app with routes
├── database.py              # DB config and session management
├── models.py                # SQLAlchemy models
├── schemas.py               # Pydantic validation schemas
├── ml_analysis.py           # ML processing engine
├── utils.py                 # Helper functions
└── tests/                   # Unit tests (future)
```

### Adding New Models
1. Place model files in `ml_models/` directory
2. Update loading logic in corresponding analyzer class
3. Ensure models are compatible with existing input/output formats

### Testing
The API includes comprehensive error handling and validation. Test using:
- **Interactive Docs**: http://localhost:8000/docs
- **Postman/Thunder Client**: Import OpenAPI schema
- **curl**: Command line testing

### Logging
- Structured logging with timestamps
- Performance metrics logging
- Error tracking and debugging
- Configurable log levels

## Deployment

### Development
- Use startup scripts for local development
- Auto-reload enabled for code changes
- SQLite database for simplicity

### Production
- Use proper WSGI server (Gunicorn + Uvicorn)
- Configure proper database (PostgreSQL recommended)
- Set up monitoring and alerting
- Use reverse proxy (Nginx)
- Enable SSL/TLS

### Docker (Future)
```dockerfile
# Dockerfile example for production deployment
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "enhanced_ml_main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Database Errors**: Check write permissions in `data/` directory
3. **Model Loading Failures**: Verify model files and paths
4. **Performance Issues**: Monitor memory usage and CPU load

### Logs
- Check console output for real-time logs
- Performance metrics logged for each request
- Error details included in API responses

### Health Check
Visit http://localhost:8000/health for system status and diagnostics.

## License

This project is part of the MindSync AI application suite.

## Support

For issues and questions:
1. Check the API documentation at `/docs`
2. Review logs for error details
3. Verify system requirements and dependencies
4. Test with health check endpoint
