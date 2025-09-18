#!/bin/bash

# Whisper Server Docker 실행 참고 명령어들

echo "🐳 Whisper API Server Docker Commands"
echo "======================================"

# Docker 이미지 빌드
echo "1. 📦 Build Docker Image:"
echo "docker build -t whisper-server ."
echo ""

# Docker 컨테이너 실행 (개발용 - 대화형)
echo "2. 🔧 Run Development Container (Interactive):"
echo "docker run --rm -it --gpus all -p 7010:7010 -v \$(pwd):/workspace -w /workspace --entrypoint /bin/bash whisper-server"
echo ""

# Docker 컨테이너 실행 (프로덕션용 - 서버 모드)
echo "3. 🚀 Run Production Container (Server Mode):"
echo "docker run --rm -d --gpus all -p 7010:7010 -v \$(pwd):/workspace -w /workspace --name whisper-api whisper-server uvicorn whisper_server:app --host 0.0.0.0 --port 7010"
echo ""

# Docker 컨테이너 실행 (포그라운드 서버 모드)
echo "4. 🖥️  Run Foreground Server:"
echo "docker run --rm --gpus all -p 7010:7010 -v \$(pwd):/workspace -w /workspace whisper-server uvicorn whisper_server:app --host 0.0.0.0 --port 7010"
echo ""

# 실행 중인 컨테이너 확인
echo "5. 📋 Check Running Containers:"
echo "docker ps"
echo ""

# 컨테이너 로그 확인
echo "6. 📄 Check Container Logs:"
echo "docker logs whisper-api"
echo "docker logs -f whisper-api  # Follow logs"
echo ""

# 컨테이너 중지
echo "7. 🛑 Stop Container:"
echo "docker stop whisper-api"
echo ""

# API 테스트 명령어
echo "8. 🧪 Test API:"
echo "# Health check"
echo "curl -X GET \"http://127.0.0.1:7010/health\""
echo ""
echo "# Transcribe audio (replace sample.wav with your audio file)"
echo "curl -X POST \"http://127.0.0.1:7010/transcribe\" \\"
echo "  -F \"audio=@sample.wav\" \\"
echo "  -F \"task=transcribe\" \\"
echo "  -F \"language=ko\" \\"
echo "  -F \"return_timestamps=true\""
echo ""

# 컨테이너 내부 접속
echo "9. 🔍 Access Running Container:"
echo "docker exec -it whisper-api /bin/bash"
echo ""

# 이미지 정리
echo "10. 🧹 Clean Up:"
echo "docker rmi whisper-server"
echo "docker system prune"
echo ""

echo "💡 Tips:"
echo "- GPU가 없으면 --gpus all 옵션 제거"
echo "- 포트 변경: -p HOST_PORT:7010"
echo "- 백그라운드 실행: -d 옵션 추가"
echo "- 로그 파일: 컨테이너 내 /workspace/logs/server.log"
