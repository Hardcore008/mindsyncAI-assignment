#!/usr/bin/env python3
"""
Test script to verify enhanced audio generation and processing
"""

import base64
import numpy as np
import json
import requests
from io import BytesIO
import wave

def test_enhanced_audio_processing():
    """Test enhanced audio data with the backend"""
    
    # Generate sophisticated speech-like audio (similar to our Flutter implementation)
    sample_rate = 22050
    duration_seconds = 5
    total_samples = sample_rate * duration_seconds
    
    print(f"Generating {duration_seconds}s of sophisticated speech-like audio at {sample_rate}Hz...")
    
    # Generate audio samples
    audio_samples = []
    
    for i in range(total_samples):
        time = i / sample_rate
        wave = 0.0
        
        # Fundamental frequency modulation (voice pitch)
        pitch_modulation = 1.0 + 0.3 * np.sin(2 * np.pi * 2 * time)
        fundamental_freq = 120 * pitch_modulation
        
        # Speech formants
        wave += 0.4 * np.sin(2 * np.pi * fundamental_freq * time)           # Fundamental
        wave += 0.25 * np.sin(2 * np.pi * fundamental_freq * 2 * time)      # Second harmonic
        wave += 0.15 * np.sin(2 * np.pi * fundamental_freq * 3 * time)      # Third harmonic
        
        # Formant frequencies (vowel sounds)
        wave += 0.2 * np.sin(2 * np.pi * 700 * time)   # F1
        wave += 0.15 * np.sin(2 * np.pi * 1200 * time) # F2
        wave += 0.1 * np.sin(2 * np.pi * 2500 * time)  # F3
        
        # Random consonant-like bursts
        if np.random.random() < 0.02:
            wave += 0.5 * (np.random.random() - 0.5)
        
        # Breath noise
        wave += 0.05 * (np.random.random() - 0.5)
        
        # Amplitude envelope
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * time + np.random.random())
        wave *= envelope
        
        # Convert to 16-bit PCM
        sample = int(np.clip(wave * 20000, -32768, 32767))
        audio_samples.extend([sample & 0xFF, (sample >> 8) & 0xFF])
    
    # Create WAV file
    wav_data = create_wav_file(audio_samples, sample_rate, 1)
    audio_base64 = base64.b64encode(wav_data).decode('utf-8')
    
    print(f"Generated WAV audio: {len(wav_data)} bytes")
    print(f"Base64 length: {len(audio_base64)}")
    
    # Test with backend
    test_data = {
        "face_image": "",  # Empty for audio-only test
        "audio_data": audio_base64,
        "motion_data": [
            {
                "timestamp": 1234567890.0,
                "acceleration": {"x": 0.1, "y": 0.2, "z": 9.8},
                "gyroscope": {"x": 0.01, "y": 0.02, "z": 0.03}
            }
        ]
    }
    
    try:
        print("Testing with backend...")
        response = requests.post(
            "http://localhost:8000/analyze", 
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Backend successfully processed enhanced audio!")
            print(f"Audio analysis: {result.get('audio_analysis', {})}")
            
            # Check for meaningful analysis
            audio_result = result.get('audio_analysis', {})
            if audio_result.get('emotion') and audio_result.get('confidence', 0) > 0:
                print(f"✅ Got meaningful audio emotion: {audio_result['emotion']} (confidence: {audio_result['confidence']})")
            else:
                print("⚠️  Audio analysis returned but with low confidence or no emotion")
                
        else:
            print(f"❌ Backend error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        print("Make sure the backend is running on http://localhost:8000")

def create_wav_file(pcm_data, sample_rate, channels):
    """Create WAV file from PCM data"""
    byte_rate = sample_rate * channels * 2
    data_size = len(pcm_data)
    file_size = 36 + data_size
    
    wav_header = [
        # RIFF header
        0x52, 0x49, 0x46, 0x46,  # "RIFF"
        file_size & 0xFF, (file_size >> 8) & 0xFF, (file_size >> 16) & 0xFF, (file_size >> 24) & 0xFF,
        0x57, 0x41, 0x56, 0x45,  # "WAVE"
        
        # fmt chunk
        0x66, 0x6D, 0x74, 0x20,  # "fmt "
        0x10, 0x00, 0x00, 0x00,  # chunk size (16)
        0x01, 0x00,              # audio format (PCM)
        channels & 0xFF, (channels >> 8) & 0xFF,  # number of channels
        sample_rate & 0xFF, (sample_rate >> 8) & 0xFF, (sample_rate >> 16) & 0xFF, (sample_rate >> 24) & 0xFF,
        byte_rate & 0xFF, (byte_rate >> 8) & 0xFF, (byte_rate >> 16) & 0xFF, (byte_rate >> 24) & 0xFF,
        (channels * 2) & 0xFF, ((channels * 2) >> 8) & 0xFF,  # block align
        0x10, 0x00,              # bits per sample (16)
        
        # data chunk
        0x64, 0x61, 0x74, 0x61,  # "data"
        data_size & 0xFF, (data_size >> 8) & 0xFF, (data_size >> 16) & 0xFF, (data_size >> 24) & 0xFF,
    ]
    
    return bytes(wav_header + pcm_data)

if __name__ == "__main__":
    test_enhanced_audio_processing()
