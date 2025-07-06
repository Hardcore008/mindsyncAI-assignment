"""
Utility functions for MindSync AI Backend
"""

import os
import sys
import psutil
import time
import logging
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

def generate_session_id() -> str:
    """Generate a unique session ID"""
    return str(uuid.uuid4())

def get_system_info() -> Dict[str, Any]:
    """Get system information for health checks"""
    try:
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent,
            "python_version": sys.version,
            "platform": sys.platform
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {"error": str(e)}

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    except Exception:
        return 0.0

def get_cpu_usage() -> float:
    """Get current CPU usage percentage"""
    try:
        return psutil.cpu_percent(interval=0.1)
    except Exception:
        return 0.0

def hash_data(data: str) -> str:
    """Create a hash of sensitive data for privacy"""
    return hashlib.sha256(data.encode()).hexdigest()

def clean_old_sessions(max_age_days: int = 30) -> int:
    """Clean old session data from database (placeholder)"""
    # This would be implemented with actual database cleanup
    logger.info(f"Cleaning sessions older than {max_age_days} days")
    return 0

def validate_base64_data(data: str, max_size_mb: int = 10) -> bool:
    """Validate base64 encoded data"""
    try:
        if not data:
            return False
        
        # Remove data URL prefix if present
        if data.startswith('data:'):
            data = data.split(',')[1]
        
        # Check if it's valid base64
        import base64
        base64.b64decode(data)
        
        # Check size
        size_mb = len(data) * 3 / 4 / 1024 / 1024  # Approximate decoded size
        return size_mb <= max_size_mb
        
    except Exception:
        return False

def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for API responses"""
    return timestamp.isoformat() if timestamp else None

def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """Parse timestamp string"""
    try:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except Exception:
        return None

def calculate_session_duration(start_time: datetime, end_time: Optional[datetime] = None) -> float:
    """Calculate session duration in seconds"""
    if end_time is None:
        end_time = datetime.now()
    
    duration = end_time - start_time
    return duration.total_seconds()

def generate_recommendations(stress_level: float, attention_score: float, 
                           emotion_analysis: Optional[Dict]) -> List[str]:
    """Generate wellness recommendations based on analysis results"""
    recommendations = []
    
    try:
        # Stress-based recommendations
        if stress_level > 0.7:
            recommendations.extend([
                "Consider taking a short break to reduce stress",
                "Try deep breathing exercises",
                "Practice mindfulness or meditation"
            ])
        elif stress_level > 0.5:
            recommendations.append("Monitor your stress levels throughout the day")
        
        # Attention-based recommendations
        if attention_score < 0.4:
            recommendations.extend([
                "Try to minimize distractions in your environment",
                "Consider the Pomodoro technique for better focus",
                "Ensure you're getting adequate sleep"
            ])
        elif attention_score < 0.6:
            recommendations.append("Take short breaks to maintain focus")
        
        # Emotion-based recommendations
        if emotion_analysis and 'emotions' in emotion_analysis:
            emotions = emotion_analysis['emotions']
            
            if emotions.get('sad', 0) > 0.5:
                recommendations.extend([
                    "Consider engaging in activities you enjoy",
                    "Connect with friends or family"
                ])
            
            if emotions.get('angry', 0) > 0.5:
                recommendations.extend([
                    "Try relaxation techniques to manage anger",
                    "Consider physical exercise to release tension"
                ])
            
            if emotions.get('fear', 0) > 0.5:
                recommendations.append("Practice grounding techniques if feeling anxious")
        
        # General wellness recommendations
        if len(recommendations) == 0:
            recommendations.extend([
                "Maintain good hydration throughout the day",
                "Take regular breaks from screen time",
                "Keep up with regular physical activity"
            ])
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        recommendations = ["Focus on maintaining a healthy lifestyle"]
    
    return recommendations[:5]  # Limit to 5 recommendations

def log_performance_metrics(endpoint: str, processing_time_ms: float, 
                          status: str, additional_info: Optional[Dict] = None):
    """Log performance metrics for monitoring"""
    try:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "processing_time_ms": processing_time_ms,
            "status": status,
            "memory_usage_mb": get_memory_usage(),
            "cpu_usage_percent": get_cpu_usage()
        }
        
        if additional_info:
            metrics.update(additional_info)
        
        logger.info(f"Performance metrics: {json.dumps(metrics)}")
        
    except Exception as e:
        logger.error(f"Error logging performance metrics: {e}")

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=handlers
    )

def create_error_response(error_message: str, details: Optional[Dict] = None) -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "error": "analysis_error",
        "message": error_message,
        "timestamp": datetime.now().isoformat(),
        "details": details or {}
    }

def validate_motion_data(motion_data: List[Dict]) -> bool:
    """Validate motion sensor data format"""
    try:
        if not motion_data or not isinstance(motion_data, list):
            return False
        
        for sample in motion_data:
            if not isinstance(sample, dict):
                return False
            
            # Check required fields
            if 'timestamp' not in sample:
                return False
            
            if 'acceleration' not in sample and 'gyroscope' not in sample:
                return False
            
            # Validate acceleration data
            if 'acceleration' in sample:
                acc = sample['acceleration']
                if not isinstance(acc, dict) or not all(k in acc for k in ['x', 'y', 'z']):
                    return False
            
            # Validate gyroscope data
            if 'gyroscope' in sample:
                gyro = sample['gyroscope']
                if not isinstance(gyro, dict) or not all(k in gyro for k in ['x', 'y', 'z']):
                    return False
        
        return True
        
    except Exception:
        return False

def ensure_directory_exists(directory_path: str):
    """Ensure a directory exists, create if it doesn't"""
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def get_file_size(file_path: str) -> int:
    """Get file size in bytes"""
    try:
        return os.path.getsize(file_path)
    except Exception:
        return 0

def backup_file(source_path: str, backup_dir: str) -> bool:
    """Create a backup of a file"""
    try:
        import shutil
        ensure_directory_exists(backup_dir)
        
        source_file = Path(source_path)
        if not source_file.exists():
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{source_file.stem}_{timestamp}{source_file.suffix}"
        backup_path = Path(backup_dir) / backup_name
        
        shutil.copy2(source_path, backup_path)
        return True
        
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        return False

class PerformanceMonitor:
    """Monitor system performance and resource usage"""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
    
    def record_request(self, processing_time_ms: float, error: bool = False):
        """Record a request for monitoring"""
        self.request_count += 1
        self.total_processing_time += processing_time_ms
        
        if error:
            self.error_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        uptime = time.time() - self.start_time
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "average_processing_time_ms": self.total_processing_time / max(self.request_count, 1),
            "requests_per_second": self.request_count / max(uptime, 1),
            "system_info": get_system_info()
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0

# Global performance monitor
performance_monitor = PerformanceMonitor()
