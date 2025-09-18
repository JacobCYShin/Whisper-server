#!/usr/bin/env python3
"""
Whisper 모델 다운로드 스크립트 (Hugging Face CLI 방식)
Docker 빌드 시 미리 모델을 다운로드해둡니다.
"""

import os
import subprocess
from pathlib import Path

def download_whisper_models():
    """Hugging Face CLI로 Whisper 모델들을 미리 다운로드"""
    print("🔽 Downloading Whisper models via Hugging Face CLI...")
    
    # 모델 저장 경로 설정
    models_dir = Path("./hf_models")
    models_dir.mkdir(exist_ok=True)
    
    # 다운로드할 모델 목록
    model_configs = [
        {
            "repo_id": "openai/whisper-large-v3-turbo",
            "description": "Whisper Large V3 Turbo (기본 모델, 809M params)"
        },
        {
            "repo_id": "openai/whisper-base", 
            "description": "Whisper Base (빠른 테스트용, 74M params)"
        },
        {
            "repo_id": "openai/whisper-small",
            "description": "Whisper Small (중간 성능, 244M params)"
        }
    ]
    
    for config in model_configs:
        try:
            repo_id = config["repo_id"]
            print(f"📥 Downloading {config['description']}...")
            print(f"   Repository: {repo_id}")
            
            # huggingface-hub CLI 명령어 실행
            cmd = [
                "huggingface-cli", "download", repo_id,
                "--repo-type", "model",
                "--local-dir", str(models_dir / repo_id.replace("/", "--")),
                "--local-dir-use-symlinks", "False"
            ]
            
            print(f"   Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            print(f"✅ {config['description']} downloaded successfully")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to download {config['description']}: {e}")
            print(f"   Error output: {e.stderr}")
            continue
        except Exception as e:
            print(f"❌ Unexpected error downloading {config['description']}: {e}")
            continue
    
    print("🎉 Model download completed!")
    
    # 다운로드된 모델 확인
    print("\n📋 Downloaded models:")
    total_size = 0
    for model_dir in models_dir.rglob("*"):
        if model_dir.is_file():
            size_mb = model_dir.stat().st_size / (1024 * 1024)
            total_size += size_mb
            if size_mb > 1:  # 1MB 이상 파일만 표시
                print(f"  - {model_dir.relative_to(models_dir)}: {size_mb:.1f} MB")
    
    print(f"\n💾 Total cache size: {total_size:.1f} MB")

def check_hf_cli():
    """Hugging Face CLI 설치 확인"""
    try:
        result = subprocess.run(["huggingface-cli", "--version"], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Hugging Face CLI detected: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Hugging Face CLI not found!")
        print("   Install with: pip install huggingface_hub[cli]")
        return False

if __name__ == "__main__":
    print("🚀 Starting Whisper model download (Hugging Face CLI)...")
    
    # CLI 확인
    if not check_hf_cli():
        print("💡 Installing huggingface_hub[cli]...")
        try:
            subprocess.run(["pip", "install", "huggingface_hub[cli]"], check=True)
            print("✅ huggingface_hub[cli] installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install huggingface_hub[cli]: {e}")
            exit(1)
    
    # 모델 다운로드
    download_whisper_models()
    
    print("🎯 Model download completed successfully!")
