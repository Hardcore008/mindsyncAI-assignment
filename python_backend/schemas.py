"""
Pydantic schemas for API request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Any
from datetime import datetime
import uuid

class MotionDataSchema(BaseModel):
    """Schema for motion sensor data"""
    timestamp: float
    acceleration: Dict[str, float] = Field(..., description="x, y, z acceleration values")
    gyroscope: Dict[str, float] = Field(..., description="x, y, z gyroscope values")
    magnetometer: Optional[Dict[str, float]] = Field(None, description="x, y, z magnetometer values")

class SessionRequestSchema(BaseModel):
    """Schema for session analysis request"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    
    # Sensor data (base64 encoded)
    face_image: Optional[str] = Field(None, description="Base64 encoded face image")
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio data")
    motion_data: Optional[List[MotionDataSchema]] = Field(None, description="Motion sensor readings")
    
    # Session metadata
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    duration_seconds: Optional[float] = Field(None, description="Session duration")
    
    @validator('face_image')
    def validate_face_image(cls, v):
        if v and not v.startswith('data:image/'):
            # If not a data URL, assume it's base64
            return v
        return v
    
    @validator('audio_data')
    def validate_audio_data(cls, v):
        if v and not v.startswith('data:audio/'):
            # If not a data URL, assume it's base64
            return v
        return v

class EmotionAnalysisSchema(BaseModel):
    """Schema for emotion analysis results"""
    dominant_emotion: str = Field(..., description="Primary detected emotion")
    emotions: Dict[str, float] = Field(..., description="Emotion scores")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Analysis confidence")
    valence: float = Field(..., ge=-1.0, le=1.0, description="Emotional valence")
    arousal: float = Field(..., ge=-1.0, le=1.0, description="Emotional arousal")

class SessionResponseSchema(BaseModel):
    """Schema for session analysis response"""
    session_id: str
    timestamp: datetime
    status: str = Field(default="completed")
    
    # Analysis results
    emotion_analysis: Optional[EmotionAnalysisSchema] = None
    stress_level: Optional[float] = Field(None, ge=0.0, le=1.0, description="Stress level (0-1)")
    attention_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Attention score (0-1)")
    overall_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall wellness score (0-1)")
    
    # New cognitive state fields
    cognitive_state: Optional[str] = Field(None, description="Detected cognitive state")
    valence: Optional[float] = Field(None, ge=-1.0, le=1.0, description="Emotional valence")
    arousal: Optional[float] = Field(None, ge=0.0, le=1.0, description="Arousal level")
    
    # Quality metrics
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Data quality score")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    
    # Recommendations and insights
    recommendations: Optional[List[str]] = Field(None, description="Wellness recommendations")
    insights: Optional[List[str]] = Field(None, description="Analysis insights")
    
    # Graph data for visualization
    graph_data: Optional[str] = Field(None, description="Base64 encoded graph image")

class SessionHistorySchema(BaseModel):
    """Schema for session history item"""
    session_id: str
    timestamp: datetime
    emotion_analysis: Optional[EmotionAnalysisSchema] = None
    stress_level: Optional[float] = None
    attention_score: Optional[float] = None
    overall_score: Optional[float] = None
    duration_seconds: Optional[float] = None
    quality_score: Optional[float] = None
    status: str

class SessionListResponseSchema(BaseModel):
    """Schema for session list response"""
    sessions: List[SessionHistorySchema]
    total_count: int
    page: int = 1
    page_size: int = 50

class HealthCheckSchema(BaseModel):
    """Schema for health check response"""
    status: str
    timestamp: datetime
    database_connected: bool
    ml_models_loaded: bool
    system_info: Dict[str, Any]

class ErrorResponseSchema(BaseModel):
    """Schema for error responses"""
    error: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    details: Optional[Dict[str, Any]] = None

class UserPreferencesSchema(BaseModel):
    """Schema for user preferences"""
    user_id: str
    store_sessions: bool = True
    max_history_days: int = Field(30, ge=1, le=365)
    analysis_sensitivity: float = Field(0.5, ge=0.0, le=1.0)
    enable_audio_analysis: bool = True
    enable_face_analysis: bool = True
    enable_motion_analysis: bool = True

class AnalysisMetricsSchema(BaseModel):
    """Schema for analysis metrics"""
    processing_time_ms: float
    audio_processing_time_ms: Optional[float] = None
    face_processing_time_ms: Optional[float] = None
    motion_processing_time_ms: Optional[float] = None
    audio_quality_score: Optional[float] = None
    face_detection_confidence: Optional[float] = None
    motion_data_completeness: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None

# Configuration schemas
class MLModelConfigSchema(BaseModel):
    """Schema for ML model configuration"""
    model_name: str
    model_path: str
    enabled: bool = True
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    batch_size: int = Field(1, ge=1)

class SystemConfigSchema(BaseModel):
    """Schema for system configuration"""
    max_concurrent_sessions: int = Field(10, ge=1)
    session_timeout_minutes: int = Field(30, ge=1)
    max_file_size_mb: int = Field(10, ge=1)
    enable_analytics: bool = True
    debug_mode: bool = False
