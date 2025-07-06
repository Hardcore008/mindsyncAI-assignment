"""
Database models for MindSync AI Backend
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class Session(Base):
    """Session model for storing analysis sessions"""
    __tablename__ = "sessions"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=True)  # Optional user identification
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    
    # Analysis results (stored as JSON)
    emotion_analysis = Column(JSON, nullable=True)
    stress_level = Column(Float, nullable=True)
    attention_score = Column(Float, nullable=True)
    overall_score = Column(Float, nullable=True)
    
    # Features extracted from data (no raw data stored)
    audio_features = Column(JSON, nullable=True)
    face_features = Column(JSON, nullable=True)
    motion_features = Column(JSON, nullable=True)
    
    # Session metadata
    duration_seconds = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)  # Overall data quality
    status = Column(String, default="completed")  # completed, error, processing
    
    # Privacy compliance
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    def to_dict(self):
        """Convert session to dictionary for API responses"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "emotion_analysis": self.emotion_analysis,
            "stress_level": self.stress_level,
            "attention_score": self.attention_score,
            "overall_score": self.overall_score,
            "duration_seconds": self.duration_seconds,
            "quality_score": self.quality_score,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class UserPreferences(Base):
    """User preferences and settings"""
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True)
    
    # Privacy settings
    store_sessions = Column(Boolean, default=True)
    max_history_days = Column(Integer, default=30)
    
    # Analysis preferences
    analysis_sensitivity = Column(Float, default=0.5)  # 0.0 to 1.0
    enable_audio_analysis = Column(Boolean, default=True)
    enable_face_analysis = Column(Boolean, default=True)
    enable_motion_analysis = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class AnalysisMetrics(Base):
    """Analytics and metrics for system performance"""
    __tablename__ = "analysis_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=func.now())
    
    # Performance metrics
    processing_time_ms = Column(Float)
    audio_processing_time_ms = Column(Float)
    face_processing_time_ms = Column(Float)
    motion_processing_time_ms = Column(Float)
    
    # Quality metrics
    audio_quality_score = Column(Float)
    face_detection_confidence = Column(Float)
    motion_data_completeness = Column(Float)
    
    # System metrics
    memory_usage_mb = Column(Float)
    cpu_usage_percent = Column(Float)
    
    session_id = Column(String, nullable=True)
