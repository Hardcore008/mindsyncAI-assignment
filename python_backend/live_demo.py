#!/usr/bin/env python3
"""
MindSync AI Backend Demo - Live API Testing
Show off the production-grade cognitive analysis capabilities!
"""

import requests
import json
import base64
import time
from datetime import datetime

# Server URL
BASE_URL = "http://localhost:8000"

def create_sample_data():
    """Create sample multimodal data for testing"""
    
    # Sample face image (small 1x1 pixel PNG in base64)
    face_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    # Sample audio data (tiny WAV file in base64)
    audio_data = "UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSmN0O7CeS4F"
    
    # Sample motion data
    motion_data = [
        {
            "timestamp": time.time() * 1000,
            "acceleration": {"x": 0.1, "y": 9.8, "z": 0.2},
            "gyroscope": {"x": 0.01, "y": 0.02, "z": -0.01}
        },
        {
            "timestamp": time.time() * 1000 + 100,
            "acceleration": {"x": 0.15, "y": 9.7, "z": 0.25},
            "gyroscope": {"x": 0.02, "y": 0.01, "z": 0.01}
        },
        {
            "timestamp": time.time() * 1000 + 200,
            "acceleration": {"x": 0.08, "y": 9.9, "z": 0.18},
            "gyroscope": {"x": -0.01, "y": 0.03, "z": 0.02}
        }
    ]
    
    return {
        "session_id": f"demo_{int(time.time())}",
        "user_id": "demo_user", 
        "face_image": face_image,
        "audio_data": audio_data,
        "motion_data": motion_data,
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": 5.0
    }

def test_api_endpoints():
    """Test all the beautiful API endpoints"""
    
    print("🧠 MindSync AI Backend - Live Demo")
    print("=" * 50)
    
    # Test 1: Root endpoint
    print("\n🏠 Testing Root Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {data['status']}")
            print(f"   ✅ Version: {data['version']}")
            print(f"   ✅ Message: {data['message']}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Health check
    print("\n💗 Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {data['status']}")
            print(f"   ✅ Database: {'Connected' if data['database_connected'] else 'Disconnected'}")
            print(f"   ✅ ML Models: {'Loaded' if data['ml_models_loaded'] else 'Not Loaded'}")
            print(f"   ✅ CPU Usage: {data['system_info']['cpu_percent']}%")
            print(f"   ✅ Memory Usage: {data['system_info']['memory_percent']}%")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Performance stats  
    print("\n📊 Testing Performance Stats...")
    try:
        response = requests.get(f"{BASE_URL}/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Total Requests: {data.get('total_requests', 0)}")
            print(f"   ✅ Average Response Time: {data.get('average_response_time', 0):.2f}ms")
            print(f"   ✅ Success Rate: {data.get('success_rate', 100):.1f}%")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Main analysis endpoint (THE BIG ONE!)
    print("\n🚀 Testing Cognitive Analysis Endpoint...")
    print("   📸 Processing face image...")
    print("   🎤 Processing audio data...")
    print("   📱 Processing motion data...")
    
    try:
        sample_data = create_sample_data()
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/analyze",
            json=sample_data,
            headers={"Content-Type": "application/json"}
        )
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n   🎉 ANALYSIS COMPLETE! ({(end_time - start_time)*1000:.0f}ms)")
            print(f"   ✅ Session ID: {data['session_id']}")
            print(f"   ✅ Status: {data['status']}")
            print(f"   ✅ Cognitive State: {data.get('cognitive_state', 'N/A').upper()}")
            print(f"   ✅ Stress Level: {data.get('stress_level', 0):.2f}/1.0")
            print(f"   ✅ Attention Score: {data.get('attention_score', 0):.2f}/1.0")
            print(f"   ✅ Overall Score: {data.get('overall_score', 0):.2f}/1.0")
            print(f"   ✅ Valence: {data.get('valence', 0):.2f} (emotional positivity)")
            print(f"   ✅ Arousal: {data.get('arousal', 0):.2f} (energy level)")
            print(f"   ✅ Quality Score: {data.get('quality_score', 0):.2f}/1.0")
            print(f"   ✅ Processing Time: {data.get('processing_time_ms', 0):.0f}ms")
            
            # Show emotion analysis if available
            if data.get('emotion_analysis'):
                emotion = data['emotion_analysis']
                print(f"   ✅ Dominant Emotion: {emotion.get('dominant_emotion', 'N/A').upper()}")
                print(f"   ✅ Emotion Confidence: {emotion.get('confidence', 0):.2f}")
                
                # Show top emotions
                emotions = emotion.get('emotions', {})
                if emotions:
                    top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                    print("   📊 Top Emotions:")
                    for emo, score in top_emotions:
                        print(f"      • {emo.capitalize()}: {score:.2f}")
            
            # Show insights
            insights = data.get('insights', [])
            if insights:
                print("   💡 Insights:")
                for insight in insights:
                    print(f"      • {insight}")
            
            # Show recommendations
            recommendations = data.get('recommendations', [])
            if recommendations:
                print("   🎯 Recommendations:")
                for rec in recommendations:
                    print(f"      • {rec}")
                    
            # Graph data
            if data.get('graph_data'):
                print("   📈 Visualization: Generated successfully (Base64 encoded)")
            
        else:
            print(f"   ❌ Failed: {response.status_code}")
            print(f"   ❌ Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 5: Session history
    print("\n📚 Testing Session History...")
    try:
        response = requests.get(f"{BASE_URL}/sessions?limit=5")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Total Sessions: {data.get('total_count', 0)}")
            print(f"   ✅ Current Page: {data.get('page', 1)}")
            print(f"   ✅ Sessions Retrieved: {len(data.get('sessions', []))}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n🎉 Demo Complete! Check out the Swagger docs at http://localhost:8000/docs")
    print("=" * 50)

if __name__ == "__main__":
    test_api_endpoints()
