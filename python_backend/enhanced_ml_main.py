"""
Enhanced MindSync AI Backend - Production-Ready FastAPI Server
Real-time ML analysis server with comprehensive cognitive state assessment
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import asyncio
import uvicorn
import logging
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

# Import our modules
from database import get_database_session, init_database, check_database_health
from models import Session as SessionModel, AnalysisMetrics, UserPreferences
from schemas import (
    SessionRequestSchema, 
    SessionResponseSchema, 
    SessionHistorySchema,
    SessionListResponseSchema,
    HealthCheckSchema,
    ErrorResponseSchema,
    EmotionAnalysisSchema,
    UserPreferencesSchema
)
from production_ml_engine import production_ml_engine  # Use production engine instead
from utils import (
    generate_session_id, 
    get_system_info, 
    log_performance_metrics,
    generate_recommendations,
    create_error_response,
    validate_motion_data,
    performance_monitor,
    setup_logging
)

# Setup logging
setup_logging("INFO")
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MindSync AI Backend",
    description="Advanced cognitive state analysis with real ML models",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for Flutter frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("Starting MindSync AI Backend...")
    
    try:
        # Initialize database
        init_database()
        logger.info("Database initialized successfully")
        
        # Check ML models (they're initialized in ml_analysis.py)
        logger.info("ML models loaded successfully")
        
        # Log startup
        logger.info("MindSync AI Backend started successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down MindSync AI Backend...")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global error: {exc}")
    return JSONResponse(
        status_code=500,
        content=create_error_response("Internal server error", {"type": str(type(exc).__name__)})
    )

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "MindSync AI Backend",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthCheckSchema)
async def health_check():
    """Health check endpoint"""
    try:
        db_healthy = check_database_health()
        system_info = get_system_info()
        
        return HealthCheckSchema(
            status="healthy" if db_healthy else "degraded",
            timestamp=datetime.now(),
            database_connected=db_healthy,
            ml_models_loaded=True,  # ML models are always loaded
            system_info=system_info
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/analyze", response_model=SessionResponseSchema)
async def analyze_session(
    request: SessionRequestSchema,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_database_session)
):
    """
    Main analysis endpoint - processes face, audio, and motion data
    """
    start_time = time.time()
    session_id = request.session_id or generate_session_id()
    
    try:
        logger.info(f"Processing session {session_id}")
        
        # Validate input data
        if not any([request.face_image, request.audio_data, request.motion_data]):
            raise HTTPException(
                status_code=400, 
                detail="At least one data modality (face, audio, or motion) must be provided"
            )
        
        # Validate motion data format if provided
        if request.motion_data and not validate_motion_data([data.dict() for data in request.motion_data]):
            raise HTTPException(status_code=400, detail="Invalid motion data format")
        
        # Convert motion data to dict format for ML processing
        motion_dict = None
        if request.motion_data:
            motion_dict = [data.dict() for data in request.motion_data]
        
        # Run ML analysis using production engine for comprehensive analysis
        analysis_results = production_ml_engine.analyze_session(
            face_image=request.face_image,
            audio_data=request.audio_data,
            motion_data=motion_dict
        )
        
        # Check for analysis errors
        if 'error' in analysis_results.get('analysis_metadata', {}):
            logger.warning(f"Analysis failure for session {session_id}: {analysis_results['analysis_metadata']['error']}")
        
        # Extract results from new structure
        cognitive_state = analysis_results.get('cognitive_state', 'neutral')
        stress_level = analysis_results.get('stress_level', 0.5)
        attention_score = analysis_results.get('attention', 0.5)
        overall_score = (analysis_results.get('valence', 0.0) + 1) / 2  # Convert valence to 0-1 scale
        quality_score = np.mean([
            result.get('quality_score', 0.5) 
            for result in analysis_results.get('modality_results', {}).values()
        ]) if analysis_results.get('modality_results') else 0.5
        processing_time_ms = analysis_results.get('processing_time_ms', 0)
        
        # Extract emotion analysis from modality results
        emotion_analysis = None
        modality_results = analysis_results.get('modality_results', {})
        if 'emotion' in modality_results:
            emotion_result = modality_results['emotion']
            emotion_analysis = {
                'emotions': emotion_result.get('raw_predictions', {}),
                'confidence': emotion_result.get('confidence', 0.5),
                'valence': analysis_results.get('valence', 0.0),
                'arousal': analysis_results.get('arousal', 0.5),
                'dominant_emotion': max(emotion_result.get('raw_predictions', {'neutral': 1.0}).items(), key=lambda x: x[1])[0] if emotion_result.get('raw_predictions') else 'neutral'
            }
        
        # Use insights and recommendations from fusion engine
        recommendations = analysis_results.get('recommendations', [
            "Continue monitoring your cognitive state",
            "Practice mindfulness if stress levels are elevated"
        ])
        
        # Create emotion analysis schema if available
        emotion_schema = None
        if emotion_analysis and 'emotions' in emotion_analysis:
            emotion_schema = EmotionAnalysisSchema(
                dominant_emotion=emotion_analysis.get('dominant_emotion', 'neutral'),
                emotions=emotion_analysis['emotions'],
                confidence=emotion_analysis.get('confidence', 0.5),
                valence=emotion_analysis.get('valence', 0.0),
                arousal=emotion_analysis.get('arousal', 0.0)
            )
        
        # Create response with all enhanced fields
        response = SessionResponseSchema(
            session_id=session_id,
            timestamp=request.timestamp or datetime.now(),
            status="completed",
            emotion_analysis=emotion_schema,
            stress_level=stress_level,
            attention_score=attention_score,
            overall_score=overall_score,
            cognitive_state=cognitive_state,
            valence=analysis_results.get('valence', 0.0),
            arousal=analysis_results.get('arousal', 0.5),
            quality_score=quality_score,
            processing_time_ms=processing_time_ms,
            recommendations=recommendations,
            insights=analysis_results.get('insights', []),
            graph_data=analysis_results.get('graph_data')
        )
        
        # Store session in database (background task)
        background_tasks.add_task(
            store_session_data,
            db, session_id, request, analysis_results, processing_time_ms
        )
        
        # Log performance metrics
        total_time = (time.time() - start_time) * 1000
        performance_monitor.record_request(total_time)
        log_performance_metrics("/analyze", total_time, "success")
        
        logger.info(f"Session {session_id} processed successfully in {total_time:.2f}ms")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        # Log error and record metrics
        total_time = (time.time() - start_time) * 1000
        performance_monitor.record_request(total_time, error=True)
        log_performance_metrics("/analyze", total_time, "error", {"error": str(e)})
        
        logger.error(f"Analysis failed for session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/sessions", response_model=SessionListResponseSchema)
async def get_sessions(
    user_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_database_session)
):
    """Get session history"""
    try:
        # Build query
        query = db.query(SessionModel)
        
        if user_id:
            query = query.filter(SessionModel.user_id == user_id)
        
        # Get total count
        total_count = query.count()
        
        # Get sessions with pagination
        sessions = query.order_by(SessionModel.timestamp.desc()).offset(offset).limit(limit).all()
        
        # Convert to schema format
        session_schemas = []
        for session in sessions:
            emotion_schema = None
            if session.emotion_analysis:
                emotion_data = session.emotion_analysis
                emotion_schema = EmotionAnalysisSchema(
                    dominant_emotion=emotion_data.get('dominant_emotion', 'neutral'),
                    emotions=emotion_data.get('emotions', {}),
                    confidence=emotion_data.get('confidence', 0.5),
                    valence=emotion_data.get('valence', 0.0),
                    arousal=emotion_data.get('arousal', 0.0)
                )
            
            session_schemas.append(SessionHistorySchema(
                session_id=session.id,
                timestamp=session.timestamp,
                emotion_analysis=emotion_schema,
                stress_level=session.stress_level,
                attention_score=session.attention_score,
                overall_score=session.overall_score,
                duration_seconds=session.duration_seconds,
                quality_score=session.quality_score,
                status=session.status
            ))
        
        return SessionListResponseSchema(
            sessions=session_schemas,
            total_count=total_count,
            page=offset // limit + 1,
            page_size=limit
        )
        
    except Exception as e:
        logger.error(f"Error retrieving sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions")

@app.get("/session/{session_id}", response_model=SessionResponseSchema)
async def get_session_details(
    session_id: str,
    db: Session = Depends(get_database_session)
):
    """Get detailed session information"""
    try:
        session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Convert to response format
        emotion_schema = None
        if session.emotion_analysis:
            emotion_data = session.emotion_analysis
            emotion_schema = EmotionAnalysisSchema(
                dominant_emotion=emotion_data.get('dominant_emotion', 'neutral'),
                emotions=emotion_data.get('emotions', {}),
                confidence=emotion_data.get('confidence', 0.5),
                valence=emotion_data.get('valence', 0.0),
                arousal=emotion_data.get('arousal', 0.0)
            )
        
        return SessionResponseSchema(
            session_id=session.id,
            timestamp=session.timestamp,
            status=session.status,
            emotion_analysis=emotion_schema,
            stress_level=session.stress_level,
            attention_score=session.attention_score,
            overall_score=session.overall_score,
            quality_score=session.quality_score,
            processing_time_ms=None,  # Not stored in DB
            recommendations=None  # Would need to regenerate
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session")

@app.get("/stats", response_model=Dict[str, Any])
async def get_performance_stats():
    """Get performance statistics"""
    try:
        return performance_monitor.get_stats()
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

@app.post("/preferences", response_model=UserPreferencesSchema)
async def update_user_preferences(
    preferences: UserPreferencesSchema,
    db: Session = Depends(get_database_session)
):
    """Update user preferences"""
    try:
        # Check if preferences exist
        existing = db.query(UserPreferences).filter(
            UserPreferences.user_id == preferences.user_id
        ).first()
        
        if existing:
            # Update existing preferences
            for key, value in preferences.dict(exclude={'user_id'}).items():
                setattr(existing, key, value)
            db.commit()
            return preferences
        else:
            # Create new preferences
            new_prefs = UserPreferences(**preferences.dict())
            db.add(new_prefs)
            db.commit()
            return preferences
            
    except Exception as e:
        logger.error(f"Error updating preferences: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update preferences")

@app.get("/preferences/{user_id}", response_model=UserPreferencesSchema)
async def get_user_preferences(
    user_id: str,
    db: Session = Depends(get_database_session)
):
    """Get user preferences"""
    try:
        preferences = db.query(UserPreferences).filter(
            UserPreferences.user_id == user_id
        ).first()
        
        if not preferences:
            # Return default preferences
            return UserPreferencesSchema(user_id=user_id)
        
        return UserPreferencesSchema(
            user_id=preferences.user_id,
            store_sessions=preferences.store_sessions,
            max_history_days=preferences.max_history_days,
            analysis_sensitivity=preferences.analysis_sensitivity,
            enable_audio_analysis=preferences.enable_audio_analysis,
            enable_face_analysis=preferences.enable_face_analysis,
            enable_motion_analysis=preferences.enable_motion_analysis
        )
        
    except Exception as e:
        logger.error(f"Error getting preferences for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get preferences")

async def store_session_data(
    db: Session, session_id: str, request: SessionRequestSchema, 
    analysis_results: Dict[str, Any], processing_time_ms: float
):
    """Background task to store session data in database"""
    try:
        # Extract analysis results
        emotion_analysis = analysis_results.get('emotion_analysis')
        stress_level = analysis_results.get('stress_level')
        attention_score = analysis_results.get('attention_score')
        overall_score = analysis_results.get('overall_score')
        quality_score = analysis_results.get('quality_score')
        
        # Create session record (no raw data stored for privacy)
        session = SessionModel(
            id=session_id,
            user_id=request.user_id,
            timestamp=request.timestamp or datetime.now(),
            emotion_analysis=emotion_analysis,
            stress_level=stress_level,
            attention_score=attention_score,
            overall_score=overall_score,
            duration_seconds=request.duration_seconds,
            quality_score=quality_score,
            status="completed"
        )
        
        db.add(session)
        
        # Store metrics
        metrics = AnalysisMetrics(
            session_id=session_id,
            processing_time_ms=processing_time_ms,
            audio_processing_time_ms=analysis_results.get('modality_results', {}).get('audio', {}).get('processing_time_ms'),
            face_processing_time_ms=analysis_results.get('modality_results', {}).get('emotion', {}).get('processing_time_ms'),
            motion_processing_time_ms=analysis_results.get('modality_results', {}).get('motion', {}).get('processing_time_ms'),
            audio_quality_score=analysis_results.get('modality_results', {}).get('audio', {}).get('confidence'),
            face_detection_confidence=analysis_results.get('modality_results', {}).get('emotion', {}).get('confidence'),
            motion_data_completeness=analysis_results.get('modality_results', {}).get('motion', {}).get('quality_score'),
            memory_usage_mb=None,  # Could be added if needed
            cpu_usage_percent=None  # Could be added if needed
        )
        
        db.add(metrics)
        db.commit()
        
        logger.info(f"Session {session_id} stored in database")
        
    except Exception as e:
        logger.error(f"Error storing session {session_id}: {e}")
        db.rollback()

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "enhanced_ml_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
        