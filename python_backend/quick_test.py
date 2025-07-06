#!/usr/bin/env python3
"""
Quick Backend Test Script
Test the MindSync AI backend without full environment setup
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing basic Python imports...")
    
    try:
        import json
        import time
        import base64
        from datetime import datetime
        print("✓ Basic Python modules OK")
    except ImportError as e:
        print(f"✗ Basic modules failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy OK")
    except ImportError:
        print("✗ NumPy not available - installing...")
        os.system("pip install numpy")
    
    try:
        import cv2
        print("✓ OpenCV OK")
    except ImportError:
        print("✗ OpenCV not available - will use fallback")
    
    try:
        import fastapi
        from pydantic import BaseModel
        print("✓ FastAPI and Pydantic OK")
    except ImportError:
        print("✗ FastAPI not available - installing core packages...")
        os.system("pip install fastapi uvicorn pydantic")
    
    return True

def test_backend_modules():
    """Test our backend modules"""
    print("\nTesting backend modules...")
    
    try:
        from schemas import SessionRequestSchema, SessionResponseSchema
        print("✓ Schemas module OK")
    except ImportError as e:
        print(f"✗ Schemas import failed: {e}")
        return False
    
    try:
        from database import init_database
        print("✓ Database module OK")
    except ImportError as e:
        print(f"✗ Database import failed: {e}")
        return False
    
    try:
        from utils import generate_session_id, get_system_info
        print("✓ Utils module OK")
    except ImportError as e:
        print(f"✗ Utils import failed: {e}")
        return False
    
    return True

def quick_ml_test():
    """Quick test of ML functionality"""
    print("\nTesting ML functionality...")
    
    try:
        from ml_analysis import ml_engine
        
        # Test with minimal data
        result = ml_engine.analyze_session(
            face_image=None,
            audio_data=None,
            motion_data=[{
                'timestamp': time.time(),
                'acceleration': {'x': 0.1, 'y': 0.2, 'z': 9.8},
                'gyroscope': {'x': 0.0, 'y': 0.0, 'z': 0.0}
            }]
        )
        
        print("✓ ML analysis completed")
        print(f"  - Overall score: {result.get('overall_score', 'N/A')}")
        print(f"  - Processing time: {result.get('processing_time_ms', 'N/A')}ms")
        
        return True
        
    except Exception as e:
        print(f"✗ ML test failed: {e}")
        return False

def start_minimal_server():
    """Start a minimal version of the server"""
    print("\nStarting minimal FastAPI server...")
    
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        import uvicorn
        
        app = FastAPI(title="MindSync AI - Quick Test")
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @app.get("/")
        def root():
            return {"message": "MindSync AI Backend - Quick Test", "status": "running"}
        
        @app.get("/health")
        def health():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        print("✓ FastAPI app created")
        print("Starting server on http://localhost:8000")
        print("Visit http://localhost:8000/docs for API documentation")
        print("Press Ctrl+C to stop")
        
        uvicorn.run(app, host="0.0.0.0", port=8000)
        
    except KeyboardInterrupt:
        print("\n✓ Server stopped by user")
    except Exception as e:
        print(f"✗ Server failed: {e}")

if __name__ == "__main__":
    print("MindSync AI Backend - Quick Setup & Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n✗ Basic imports failed. Please install Python dependencies.")
        sys.exit(1)
    
    # Test backend modules
    if not test_backend_modules():
        print("\n✗ Backend modules failed. Check file paths and syntax.")
        sys.exit(1)
    
    # Test ML functionality
    if not quick_ml_test():
        print("\n⚠ ML test failed, but continuing with basic server...")
    
    # Start server
    print("\n" + "=" * 50)
    start_minimal_server()
