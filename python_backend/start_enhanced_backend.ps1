# Enhanced MindSync AI Backend Startup Script (PowerShell)
# Production-ready FastAPI server with real ML models

Write-Host "Starting Enhanced MindSync AI Backend..." -ForegroundColor Green

# Change to backend directory
Set-Location -Path $PSScriptRoot

# Check if Python is available
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Create virtual environment if it doesn't exist
if (!(Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
try {
    & "venv\Scripts\Activate.ps1"
    Write-Host "Virtual environment activated" -ForegroundColor Green
} catch {
    Write-Host "Warning: Could not activate virtual environment, using system Python" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Installing required packages..." -ForegroundColor Yellow
# Install/upgrade requirements
python -m pip install --upgrade pip
pip install -r enhanced_requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies. Check enhanced_requirements.txt" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
} else {
    Write-Host "Dependencies installed successfully!" -ForegroundColor Green
}

# Create necessary directories
if (!(Test-Path "data")) {
    New-Item -ItemType Directory -Path "data"
    Write-Host "Created data directory" -ForegroundColor Yellow
}

if (!(Test-Path "ml_models")) {
    New-Item -ItemType Directory -Path "ml_models"
    Write-Host "Created ml_models directory" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Starting Enhanced ML Backend Server..." -ForegroundColor Cyan
Write-Host ""
Write-Host "Backend Features:" -ForegroundColor White
Write-Host "- Real-time emotion detection from facial expressions" -ForegroundColor Gray
Write-Host "- Audio-based stress and emotion analysis" -ForegroundColor Gray
Write-Host "- Motion sensor analysis for attention tracking" -ForegroundColor Gray
Write-Host "- Multi-modal cognitive state fusion" -ForegroundColor Gray
Write-Host "- SQLite database with privacy compliance" -ForegroundColor Gray
Write-Host "- RESTful API with comprehensive documentation" -ForegroundColor Gray
Write-Host ""
Write-Host "Server will be available at: http://localhost:8000" -ForegroundColor Magenta
Write-Host "API documentation at: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Set environment variables
$env:PYTHONPATH = Get-Location

# Start the FastAPI server
try {
    python -m uvicorn enhanced_ml_main:app --host 0.0.0.0 --port 8000 --reload
}
catch {
    Write-Host "Error starting server: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
}

Write-Host "Server stopped." -ForegroundColor Yellow
