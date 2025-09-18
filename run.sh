#!/bin/bash

# Whisper API Server 실행 스크립트
# WSL Ubuntu 환경용

set -e

echo "🚀 Starting Whisper API Server..."

# 기본 설정
PORT=${PORT:-7010}
HOST=${HOST:-0.0.0.0}
WORKERS=${WORKERS:-1}
LOG_LEVEL=${LOG_LEVEL:-info}

# GPU 확인
if command -v nvidia-smi &> /dev/null; then
    echo "📊 GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
else
    echo "⚠️  GPU not detected or nvidia-smi not available"
fi

# Python 가상환경 확인
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "🐍 Using virtual environment: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Consider using: source .venv/bin/activate"
fi

# 의존성 확인
echo "🔍 Checking dependencies..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import whisper; print(f'Whisper: {whisper.__version__}')"
python3 -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"

# 로그 디렉토리 생성
mkdir -p logs

# 서버 실행
echo "🌐 Starting server on $HOST:$PORT"
echo "📝 Logs will be saved to logs/server.log"

if [[ "$1" == "background" ]]; then
    echo "🔄 Running in background mode..."
    nohup uvicorn whisper_server:app \
        --host $HOST \
        --port $PORT \
        --workers $WORKERS \
        --log-level $LOG_LEVEL \
        > logs/server.log 2>&1 &
    
    PID=$!
    echo "✅ Server started with PID: $PID"
    echo "📋 To check status: ps aux | grep $PID"
    echo "📄 To view logs: tail -f logs/server.log"
    echo "🛑 To stop: kill $PID"
    
else
    echo "🖥️  Running in foreground mode..."
    uvicorn whisper_server:app \
        --host $HOST \
        --port $PORT \
        --workers $WORKERS \
        --log-level $LOG_LEVEL
fi
