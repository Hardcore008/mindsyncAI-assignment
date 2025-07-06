"""
FastAPI Main Application - Production-grade cognitive analysis service
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from database import get_db, init_db
from models import SessionResult
from schemas import (
    AnalyzeRequest, 
    AnalyzeResponse, 
    SessionResponse, 
    HealthResponse,
    ErrorResponse
)
from ml_engine import MLAnalysisEngine
from config import settings
from utils import setup_logging, correlation_id_middleware

# Setup logging
logger = setup_logging()

# Global ML engine instance
ml_engine: Optional[MLAnalysisEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global ml_engine
    
    logger.info("Starting MindSync AI Backend...")
    
    # Initialize database
    logger.info("Initializing database...")
    init_db()
    
    # Initialize ML engine
    logger.info("Initializing ML analysis engine...")
    ml_engine = MLAnalysisEngine()
    await ml_engine.initialize()
    
    # Validate configuration
    logger.info("Validating configuration...")
    await validate_startup_dependencies()
    
    logger.info("MindSync AI Backend ready for requests")
    
    yield
    
    # Cleanup
    logger.info("Shutting down MindSync AI Backend...")
    if ml_engine:
        await ml_engine.cleanup()

app = FastAPI(
    title="MindSync AI - Cognitive Analysis API",
    description="Production-grade multimodal cognitive state analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add correlation ID middleware
app.middleware("http")(correlation_id_middleware)

@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request, exc):
    """Handle database errors"""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    logger.error(f"Database error [{correlation_id}]: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Database error occurred", "correlation_id": correlation_id}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general errors"""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    logger.error(f"Unexpected error [{correlation_id}]: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "correlation_id": correlation_id}
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        ml_engine_ready=ml_engine is not None and ml_engine.is_ready
    )

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_session(
    request: AnalyzeRequest,
    db: Session = Depends(get_db)
) -> AnalyzeResponse:
    """
    Comprehensive multimodal cognitive analysis endpoint
    
    Processes face image, audio, and motion data to determine cognitive state
    """
    correlation_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"Starting analysis [{correlation_id}] - modalities: "
                f"face={bool(request.face_image_b64)}, "
                f"audio={bool(request.audio_clip_b64)}, "
                f"motion={len(request.motion_data) if request.motion_data else 0}")
    
    try:
        if not ml_engine or not ml_engine.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML analysis engine not ready"
            )
        
        # Run ML analysis in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        analysis_result = await loop.run_in_executor(
            None,
            ml_engine.analyze_multimodal,
            request.face_image_b64,
            request.audio_clip_b64,
            request.motion_data
        )
        
        # Create database record
        session_result = SessionResult(
            correlation_id=correlation_id,
            timestamp=datetime.utcnow(),
            face_emotion_label=analysis_result.get('face_emotion'),
            face_confidence=analysis_result.get('face_confidence', 0.0),
            speech_emotion_label=analysis_result.get('speech_emotion'),
            speech_confidence=analysis_result.get('speech_confidence', 0.0),
            motion_level=analysis_result.get('motion_level'),
            motion_confidence=analysis_result.get('motion_confidence', 0.0),
            cognitive_state=analysis_result.get('cognitive_state'),
            fused_confidence=analysis_result.get('confidence', 0.0),
            suggestions=analysis_result.get('recommendations', []),
            graph_data=analysis_result.get('graph_data'),
            processing_time_ms=analysis_result.get('processing_time_ms', 0)
        )
        
        # Save to database
        try:
            db.add(session_result)
            db.commit()
            db.refresh(session_result)
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Database error [{correlation_id}]: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save analysis result"
            )
        
        total_time = (time.time() - start_time) * 1000
        logger.info(f"Analysis completed [{correlation_id}] - "
                   f"state: {analysis_result.get('cognitive_state')}, "
                   f"confidence: {analysis_result.get('confidence', 0):.3f}, "
                   f"total_time: {total_time:.1f}ms")
        
        return AnalyzeResponse(
            session_id=session_result.id,
            correlation_id=correlation_id,
            cognitive_state=analysis_result.get('cognitive_state'),
            confidence=analysis_result.get('confidence', 0.0),
            valence=analysis_result.get('valence', 0.0),
            arousal=analysis_result.get('arousal', 0.5),
            attention=analysis_result.get('attention', 0.5),
            stress_level=analysis_result.get('stress_level', 0.5),
            face_emotion=analysis_result.get('face_emotion'),
            face_confidence=analysis_result.get('face_confidence', 0.0),
            speech_emotion=analysis_result.get('speech_emotion'),
            speech_confidence=analysis_result.get('speech_confidence', 0.0),
            motion_level=analysis_result.get('motion_level'),
            motion_confidence=analysis_result.get('motion_confidence', 0.0),
            suggestions=analysis_result.get('recommendations', []),
            graph_data=analysis_result.get('graph_data'),
            processing_time_ms=total_time,
            timestamp=session_result.timestamp
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error [{correlation_id}]: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/sessions", response_model=List[SessionResponse])
async def get_sessions(
    limit: int = Field(default=20, ge=1, le=100),
    offset: int = Field(default=0, ge=0),
    db: Session = Depends(get_db)
) -> List[SessionResponse]:
    """
    Get recent analysis sessions
    """
    correlation_id = str(uuid.uuid4())
    
    logger.info(f"Fetching sessions [{correlation_id}] - limit: {limit}, offset: {offset}")
    
    try:
        sessions = db.query(SessionResult)\
                    .order_by(SessionResult.timestamp.desc())\
                    .offset(offset)\
                    .limit(limit)\
                    .all()
        
        logger.info(f"Retrieved {len(sessions)} sessions [{correlation_id}]")
        
        return [
            SessionResponse(
                id=session.id,
                correlation_id=session.correlation_id,
                timestamp=session.timestamp,
                cognitive_state=session.cognitive_state,
                confidence=session.fused_confidence,
                valence=session.valence,
                arousal=session.arousal,
                attention=session.attention,
                stress_level=session.stress_level,
                face_emotion=session.face_emotion_label,
                face_confidence=session.face_confidence,
                speech_emotion=session.speech_emotion_label,
                speech_confidence=session.speech_confidence,
                motion_level=session.motion_level,
                motion_confidence=session.motion_confidence,
                suggestions=session.suggestions,
                graph_data=session.graph_data,
                processing_time_ms=session.processing_time_ms
            )
            for session in sessions
        ]
        
    except Exception as e:
        logger.error(f"Error fetching sessions [{correlation_id}]: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve sessions"
        )

@app.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: int,
    db: Session = Depends(get_db)
) -> SessionResponse:
    """
    Get specific analysis session by ID
    """
    correlation_id = str(uuid.uuid4())
    
    logger.info(f"Fetching session {session_id} [{correlation_id}]")
    
    try:
        session = db.query(SessionResult).filter(SessionResult.id == session_id).first()
        
        if not session:
            logger.warning(f"Session {session_id} not found [{correlation_id}]")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        logger.info(f"Retrieved session {session_id} [{correlation_id}]")
        
        return SessionResponse(
            id=session.id,
            correlation_id=session.correlation_id,
            timestamp=session.timestamp,
            cognitive_state=session.cognitive_state,
            confidence=session.fused_confidence,
            valence=session.valence,
            arousal=session.arousal,
            attention=session.attention,
            stress_level=session.stress_level,
            face_emotion=session.face_emotion_label,
            face_confidence=session.face_confidence,
            speech_emotion=session.speech_emotion_label,
            speech_confidence=session.speech_confidence,
            motion_level=session.motion_level,
            motion_confidence=session.motion_confidence,
            suggestions=session.suggestions,
            graph_data=session.graph_data,
            processing_time_ms=session.processing_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching session {session_id} [{correlation_id}]: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session"
        )

async def validate_startup_dependencies():
    """Validate all required dependencies at startup"""
    errors = []
    
    # Check ML model files
    if not settings.FACE_MODEL_PATH.exists():
        errors.append(f"Face model not found: {settings.FACE_MODEL_PATH}")
    
    if not settings.SPEECH_MODEL_PATH.exists():
        errors.append(f"Speech model not found: {settings.SPEECH_MODEL_PATH}")
    
    if not settings.MOTION_MODEL_PATH.exists():
        errors.append(f"Motion model not found: {settings.MOTION_MODEL_PATH}")
    
    if not settings.FUSION_MODEL_PATH.exists():
        errors.append(f"Fusion model not found: {settings.FUSION_MODEL_PATH}")
    
    # Check ML engine readiness
    if not ml_engine or not ml_engine.is_ready:
        errors.append("ML analysis engine failed to initialize")
    
    # Test database connectivity
    try:
        db_session = next(get_db())
        db_session.execute("SELECT 1")
        db_session.close()
    except Exception as e:
        errors.append(f"Database connectivity failed: {e}")
    
    if errors:
        for error in errors:
            logger.error(f"Startup validation error: {error}")
        raise Exception(f"Startup validation failed: {'; '.join(errors)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
