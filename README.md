# Whisper ASR API Server

OpenAI Whisper 기반의 음성 인식 API 서버입니다. FastAPI를 사용하여 구현되었으며, TTS 서버와 유사한 구조로 설계되었습니다.

## 주요 기능

- **고성능 음성 인식**: Whisper Large V3 Turbo 모델 사용
- **다양한 오디오 포맷 지원**: wav, mp3, m4a, flac, ogg 등
- **다국어 지원**: 자동 언어 감지 또는 수동 설정
- **배치 처리**: 여러 파일 동시 처리
- **타임스탬프**: 구간별 시간 정보 제공
- **GPU 가속**: CUDA 지원으로 빠른 추론
- **RESTful API**: 간단하고 직관적인 API 인터페이스

## 시스템 요구사항

- Python 3.8+
- CUDA 지원 GPU (권장)
- 최소 8GB RAM
- 최소 4GB GPU 메모리

## 설치 및 실행

### 1. WSL Ubuntu 환경에서 직접 실행

```bash
# 프로젝트 디렉토리로 이동
cd /home/hanati/code/whisper_server_only

# 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
uvicorn whisper_server:app --host 0.0.0.0 --port 7010

# 백그라운드 실행
nohup uvicorn whisper_server:app --host 0.0.0.0 --port 7010 > server.log 2>&1 &
```

### 2. Docker 실행 (권장)

#### 이미지 빌드
```bash
# Whisper 모델이 미리 다운로드된 Docker 이미지 빌드
docker build -t whisper-server .
```

#### 개발용 실행 (대화형)
```bash
# workspace 마운트로 코드 수정 가능
docker run --rm -it --gpus all -p 7010:7010 \
  -v $(pwd):/workspace -w /workspace \
  --entrypoint /bin/bash whisper-server
```

#### 프로덕션 서버 실행
```bash
# 백그라운드 서버 실행
docker run --rm -d --gpus all -p 7010:7010 \
  -v $(pwd):/workspace -w /workspace \
  --name whisper-api whisper-server \
  uvicorn whisper_server:app --host 0.0.0.0 --port 7010

# 포그라운드 서버 실행
docker run --rm --gpus all -p 7010:7010 \
  -v $(pwd):/workspace -w /workspace \
  whisper-server uvicorn whisper_server:app --host 0.0.0.0 --port 7010
```

#### 참고 명령어
```bash
# 모든 Docker 명령어 확인
cat REF.sh

# 컨테이너 로그 확인
docker logs -f whisper-api

# 실행 중인 컨테이너에 접속
docker exec -it whisper-api /bin/bash
```

## API 사용법

### 1. 기본 음성 인식

```bash
curl -X POST "http://127.0.0.1:7010/transcribe" \
  -F "audio=@sample.wav" \
  -F "task=transcribe" \
  -F "language=ko" \
  -F "return_timestamps=true"
```

**응답 예시:**
```json
{
  "text": "안녕하세요, 오늘 날씨가 좋네요.",
  "language": "ko",
  "duration": 3.2,
  "segments": [
    {
      "start": 0.0,
      "end": 1.5,
      "text": "안녕하세요,"
    },
    {
      "start": 1.5,
      "end": 3.2,
      "text": "오늘 날씨가 좋네요."
    }
  ]
}
```

### 2. 번역 (다른 언어 → 영어)

```bash
curl -X POST "http://127.0.0.1:7010/transcribe" \
  -F "audio=@korean_audio.wav" \
  -F "task=translate" \
  -F "language=ko"
```

### 3. 배치 처리

```bash
curl -X POST "http://127.0.0.1:7010/transcribe_batch" \
  -F "files=@audio1.wav" \
  -F "files=@audio2.mp3" \
  -F "files=@audio3.m4a" \
  -F "task=transcribe" \
  -F "return_timestamps=true"
```

### 4. 헬스 체크

```bash
curl -X GET "http://127.0.0.1:7010/health"
```

### 5. 지원 모델 및 언어 확인

```bash
# 사용 가능한 모델
curl -X GET "http://127.0.0.1:7010/models"

# 지원 언어
curl -X GET "http://127.0.0.1:7010/languages"
```

## API 파라미터

### `/transcribe` 엔드포인트

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `audio` | File | 필수 | 음성 파일 |
| `task` | string | "transcribe" | "transcribe" 또는 "translate" |
| `language` | string | null | 언어 코드 (예: "ko", "en", "ja") |
| `return_timestamps` | boolean | false | 타임스탬프 포함 여부 |
| `temperature` | float | 0.0 | 샘플링 온도 (0.0-1.0) |
| `beam_size` | int | null | 빔 서치 크기 (1-5) |
| `best_of` | int | null | 후보 수 (1-5) |
| `patience` | float | null | 빔 서치 patience |

## 지원하는 오디오 포맷

- WAV
- MP3
- M4A
- FLAC
- OGG
- AAC
- WMA

## 지원하는 언어 (일부)

- 한국어 (ko)
- 영어 (en)
- 일본어 (ja)
- 중국어 (zh)
- 스페인어 (es)
- 프랑스어 (fr)
- 독일어 (de)
- 이탈리아어 (it)
- 포르투갈어 (pt)
- 러시아어 (ru)
- 아랍어 (ar)
- 힌디어 (hi)
- 태국어 (th)
- 베트남어 (vi)

## 성능 최적화

### GPU 메모리 사용량
- **tiny**: ~1GB
- **base**: ~1GB
- **small**: ~2GB
- **medium**: ~5GB
- **large**: ~10GB
- **large-v3-turbo**: ~6GB (권장)

### 처리 속도 (RTX 4090 기준)
- **실시간 처리**: 1분 오디오 → 약 3-5초
- **배치 처리**: 10개 파일 → 약 30-60초

## 로그 및 모니터링

```bash
# 실시간 로그 확인
tail -f server.log

# GPU 사용량 모니터링
watch -n 1 nvidia-smi
```

## 문제 해결

### 일반적인 오류

1. **CUDA 메모리 부족**
   ```
   RuntimeError: CUDA out of memory
   ```
   → 더 작은 모델 사용 또는 배치 크기 감소

2. **오디오 포맷 오류**
   ```
   Audio preprocessing failed
   ```
   → 지원되는 포맷으로 변환 또는 ffmpeg 설치 확인

3. **모델 다운로드 실패**
   ```
   Failed to initialize Whisper model
   ```
   → 인터넷 연결 확인 및 재시도

### 성능 튜닝

```python
# 메모리 사용량 감소
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 모델 캐시 위치 설정
export WHISPER_CACHE_DIR=/path/to/cache
```

## 라이선스

MIT License

## 기여

버그 리포트나 기능 요청은 이슈로 등록해주세요.

## 관련 프로젝트

- [TTS Server](../TTS_server_only/): 음성 합성 API 서버
- [OpenAI Whisper](https://github.com/openai/whisper): 원본 Whisper 모델
