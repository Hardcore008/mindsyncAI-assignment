@echo off
REM Enhanced MindSync AI Backend Startup Script (Batch)
REM Production-ready FastAPI server with real ML models

echo Starting Enhanced MindSync AI Backend...
echo.

cd /d "%~dp0"

echo Checking Python installation...
python --version
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Installing/upgrading dependencies...
python -m pip install --upgrade pip
pip install -r enhanced_requirements.txt

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install dependencies. Check enhanced_requirements.txt
    pause
    exit /b 1
)

echo Dependencies installed successfully!

REM Create necessary directories
if not exist "data" (
    mkdir data
    echo Created data directory
)

if not exist "ml_models" (
    mkdir ml_models
    echo Created ml_models directory
)

echo.
echo Starting Enhanced FastAPI Backend Server...
echo.
echo Backend Features:
echo - Real-time facial emotion detection with MediaPipe
echo - Audio stress analysis with librosa feature extraction  
echo - Motion pattern analysis for attention scoring
echo - Multi-modal cognitive state fusion
echo - SQLite database with privacy-compliant storage
echo - RESTful API with automatic documentation
echo.
echo Server will be available at: http://localhost:8000
echo API documentation at: http://localhost:8000/docs
echo Press Ctrl+C to stop the server
echo.

REM Set environment variables
set PYTHONPATH=%CD%

REM Start the FastAPI server
python -m uvicorn enhanced_ml_main:app --host 0.0.0.0 --port 8000 --reload

if %ERRORLEVEL% neq 0 (
    echo Error starting server
    pause
)

echo Server stopped.
pause
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing required packages...
python -m pip install --upgrade pip
pip install -r enhanced_requirements.txt

if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies. Check enhanced_requirements.txt
    pause
    exit /b 1
) else (
    echo Dependencies installed successfully!
)

REM Create necessary directories
if not exist "data" (
    mkdir data
    echo Created data directory
)

if not exist "ml_models" (
    mkdir ml_models
    echo Created ml_models directory
)

echo.
echo Starting Enhanced ML Backend Server...
echo.
echo Backend Features:
echo - Real-time emotion detection from facial expressions
echo - Audio-based stress and emotion analysis
echo - Motion sensor analysis for attention tracking
echo - Multi-modal cognitive state fusion
echo - SQLite database with privacy compliance
echo - RESTful API with comprehensive documentation
echo.
echo Server will be available at: http://localhost:8000
echo API documentation at: http://localhost:8000/docs
echo Press Ctrl+C to stop the server
echo.

REM Set environment variables
set PYTHONPATH=%cd%

REM Start the FastAPI server
python -m uvicorn enhanced_ml_main:app --host 0.0.0.0 --port 8000 --reload

echo.
echo Server stopped.
pause
