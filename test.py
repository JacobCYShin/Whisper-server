#!/usr/bin/env python3
"""
Whisper API Server 테스트 스크립트
"""

import requests
import json
import time
import sys
from pathlib import Path

def test_health_check(base_url="http://127.0.0.1:7010"):
    """헬스 체크 테스트"""
    print("=== Health Check Test ===")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Health check passed: {result}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_models_endpoint(base_url="http://127.0.0.1:7010"):
    """모델 정보 테스트"""
    print("\n=== Models Endpoint Test ===")
    try:
        response = requests.get(f"{base_url}/models")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Models info: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
        return False

def test_languages_endpoint(base_url="http://127.0.0.1:7010"):
    """언어 정보 테스트"""
    print("\n=== Languages Endpoint Test ===")
    try:
        response = requests.get(f"{base_url}/languages")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Languages supported: {len(result.get('languages', {}))} languages")
            return True
        else:
            print(f"❌ Languages endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Languages endpoint error: {e}")
        return False

def create_test_audio():
    """테스트용 오디오 파일 생성"""
    try:
        import numpy as np
        import soundfile as sf
        
        # 1초 사인파 생성 (440Hz)
        sample_rate = 16000
        duration = 1.0
        frequency = 440
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        test_file = "test_audio.wav"
        sf.write(test_file, audio, sample_rate)
        print(f"✅ Test audio file created: {test_file}")
        return test_file
        
    except ImportError:
        print("❌ Cannot create test audio: soundfile or numpy not available")
        return None
    except Exception as e:
        print(f"❌ Error creating test audio: {e}")
        return None

def test_transcribe_endpoint(base_url="http://127.0.0.1:7010", audio_file=None):
    """음성 인식 테스트"""
    print("\n=== Transcribe Endpoint Test ===")
    
    if not audio_file:
        audio_file = create_test_audio()
        if not audio_file:
            print("❌ No audio file available for testing")
            return False
    
    try:
        with open(audio_file, 'rb') as f:
            files = {'audio': (audio_file, f, 'audio/wav')}
            data = {
                'task': 'transcribe',
                'language': 'en',
                'return_timestamps': 'true'
            }
            
            print(f"📤 Sending request to {base_url}/transcribe...")
            start_time = time.time()
            response = requests.post(f"{base_url}/transcribe", files=files, data=data)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Transcription successful!")
                print(f"   Text: {result.get('text', 'N/A')}")
                print(f"   Language: {result.get('language', 'N/A')}")
                print(f"   Duration: {result.get('duration', 'N/A'):.2f}s")
                print(f"   Processing time: {end_time - start_time:.2f}s")
                
                if result.get('segments'):
                    print(f"   Segments: {len(result['segments'])} segments")
                
                return True
            else:
                print(f"❌ Transcribe failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Transcribe error: {e}")
        return False

def run_all_tests(base_url="http://127.0.0.1:7010", audio_file=None):
    """모든 테스트 실행"""
    print("🚀 Starting Whisper API Server Tests...")
    print(f"📍 Base URL: {base_url}")
    
    tests = [
        ("Health Check", lambda: test_health_check(base_url)),
        ("Models Endpoint", lambda: test_models_endpoint(base_url)),
        ("Languages Endpoint", lambda: test_languages_endpoint(base_url)),
        ("Transcribe Endpoint", lambda: test_transcribe_endpoint(base_url, audio_file))
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
        
        time.sleep(1)  # 테스트 간 간격
    
    print("\n" + "="*50)
    print("📊 TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Total: {passed}/{len(results)} tests passed")
    
    # 테스트 파일 정리
    test_file = "test_audio.wav"
    if Path(test_file).exists():
        Path(test_file).unlink()
        print(f"🧹 Cleaned up test file: {test_file}")
    
    return passed == len(results)

if __name__ == "__main__":
    # 명령행 인수 처리
    base_url = "http://127.0.0.1:7010"
    audio_file = None
    
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    if len(sys.argv) > 2:
        audio_file = sys.argv[2]
    
    success = run_all_tests(base_url, audio_file)
    sys.exit(0 if success else 1)
