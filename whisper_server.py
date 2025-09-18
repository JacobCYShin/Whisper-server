# 2025.09.16 version
# Whisper ASR API Server
# written by ANDY

import os
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

import torch
import librosa
import numpy as np
import soundfile as sf
import io
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal, List
from fastapi.responses import JSONResponse
from dataclasses import dataclass
from time import time
import logging
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperRequest(BaseModel):
    task: Literal["transcribe", "translate"] = Field(default="transcribe", description="전사 또는 번역")
    language: Optional[str] = Field(default=None, description="음성 언어 (자동 감지시 None)")
    return_timestamps: bool = Field(default=False, description="타임스탬프 반환 여부")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="샘플링 온도")
    beam_size: Optional[int] = Field(default=None, ge=1, le=5, description="빔 서치 크기")
    best_of: Optional[int] = Field(default=None, ge=1, le=5, description="후보 수")
    patience: Optional[float] = Field(default=None, ge=0.0, description="빔 서치 patience")

class WhisperResponse(BaseModel):
    text: str
    language: str
    duration: float
    segments: Optional[List[dict]] = None

@dataclass
class WhisperConfig:
    model_id: str = 'openai/whisper-large-v3-turbo'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

app = FastAPI(title="Whisper ASR API Server", version="1.0.0")

# CORS 미들웨어 추가 (브라우저에서 접근 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 origin 허용 (개발용)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

whisper_pipeline = None

def initialize_whisper(config: WhisperConfig):
    """Whisper 모델 초기화 (Hugging Face 방식)"""
    global whisper_pipeline
    logger.info(f"Loading Whisper model: {config.model_id} on {config.device}")
    
    try:
        # 모델 디렉토리 설정 (HF CLI로 다운로드된 모델 사용)
        import os
        model_paths = ["/app/hf_models", "./hf_models"]
        model_dir = None
        
        for base_path in model_paths:
            # HF CLI는 repo_id를 "openai--whisper-large-v3-turbo" 형식으로 저장
            repo_dir = Path(base_path) / config.model_id.replace("/", "--")
            if repo_dir.exists():
                model_dir = str(repo_dir)
                logger.info(f"Using local model directory: {model_dir}")
                break
        
        # 로컬 모델이 없으면 온라인에서 다운로드
        if not model_dir:
            model_dir = config.model_id
            logger.info(f"Local model not found, downloading from: {model_dir}")
        
        # 모델과 프로세서 로드
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_dir,
            torch_dtype=config.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            local_files_only=(model_dir != config.model_id)  # 로컬 파일만 사용할지 결정
        )
        model.to(config.device)
        
        processor = AutoProcessor.from_pretrained(
            model_dir,
            local_files_only=(model_dir != config.model_id)
        )
        
        # Pipeline 생성
        whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=config.torch_dtype,
            device=config.device,
        )
        
        # Warmup
        logger.info("Warming up model...")
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1초 무음
        result = whisper_pipeline(dummy_audio)
        logger.info(f"Model warmup completed. Sample result: {result['text'][:50]}")
        
        return whisper_pipeline
    except Exception as e:
        logger.error(f"Failed to initialize Whisper model: {str(e)}")
        raise e

def preprocess_audio(audio_file: UploadFile) -> np.ndarray:
    """오디오 파일 전처리"""
    try:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.filename.split('.')[-1]}") as tmp_file:
            content = audio_file.file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # librosa로 오디오 로드 (16kHz로 리샘플링)
        audio, sr = librosa.load(tmp_file_path, sr=16000, mono=True)
        
        # 임시 파일 삭제
        os.unlink(tmp_file_path)
        
        logger.info(f"Audio processed: duration={len(audio)/16000:.2f}s, sr={sr}")
        return audio
        
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Audio preprocessing failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """서버 시작시 모델 로드"""
    logger.info("FastAPI Whisper server is starting...")
    try:
        initialize_whisper(WhisperConfig())
        logger.info("Whisper pipeline loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Whisper pipeline: {str(e)}")
        raise e

@app.post("/transcribe", response_model=WhisperResponse)
async def transcribe_audio(
    audio: UploadFile = File(..., description="오디오 파일 (wav, mp3, m4a, flac 등)"),
    task: str = Form(default="transcribe"),
    language: Optional[str] = Form(default=None),
    return_timestamps: bool = Form(default=False),
    temperature: float = Form(default=0.0),
    beam_size: Optional[int] = Form(default=None),
    best_of: Optional[int] = Form(default=None),
    patience: Optional[float] = Form(default=None)
):
    """음성 인식 API 엔드포인트"""
    try:
        if not whisper_pipeline:
            raise HTTPException(status_code=500, detail="Whisper pipeline not loaded")
        
        # 파일 확장자 검증
        allowed_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma'}
        file_extension = Path(audio.filename).suffix.lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_extension}")
        
        start_time = time()
        logger.info(f"Processing file: {audio.filename}")
        
        # 오디오 전처리
        audio_array = preprocess_audio(audio)
        duration = len(audio_array) / 16000
        
        # Hugging Face pipeline 파라미터 설정
        generate_kwargs = {
            "task": task,
            "return_timestamps": return_timestamps,
        }
        
        if language:
            generate_kwargs["language"] = language
        if temperature != 0.0:
            generate_kwargs["temperature"] = temperature
        if beam_size:
            generate_kwargs["num_beams"] = beam_size
            
        # Whisper 추론
        # 30초 초과(long-form) 입력은 타임스탬프/청크 분할이 필요
        logger.info(f"Starting Whisper inference with params: {generate_kwargs}")
        top_level_kwargs = {}
        if "return_timestamps" in generate_kwargs:
            top_level_kwargs["return_timestamps"] = generate_kwargs.pop("return_timestamps")

        if duration > 30.0:
            if not top_level_kwargs.get("return_timestamps", False):
                logger.info("Input >30s detected. Enabling return_timestamps automatically.")
                top_level_kwargs["return_timestamps"] = True
            # 청크 분할로 메모리/성능 안정화
            top_level_kwargs.setdefault("chunk_length_s", 30)
            top_level_kwargs.setdefault("stride_length_s", [4, 2])

        result = whisper_pipeline(
            audio_array,
            **top_level_kwargs,
            generate_kwargs=generate_kwargs,
        )
        
        inference_time = time() - start_time
        logger.info(f"Inference completed in {inference_time:.2f}s")
        
        # 응답 구성
        response = WhisperResponse(
            text=result["text"].strip(),
            language="auto-detected",  # HF pipeline doesn't return language directly
            duration=duration
        )
        
        if return_timestamps and "chunks" in result:
            response.segments = [
                {
                    "start": chunk["timestamp"][0] if chunk["timestamp"] else 0.0,
                    "end": chunk["timestamp"][1] if chunk["timestamp"] else duration,
                    "text": chunk["text"].strip()
                }
                for chunk in result["chunks"]
            ]
        
        logger.info(f"Transcription result: {response.text[:100]}...")
        return response
        
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/transcribe_batch")
async def transcribe_batch(
    files: List[UploadFile] = File(..., description="여러 오디오 파일들"),
    task: str = Form(default="transcribe"),
    language: Optional[str] = Form(default=None),
    return_timestamps: bool = Form(default=False)
):
    """배치 음성 인식 API"""
    try:
        if not whisper_pipeline:
            raise HTTPException(status_code=500, detail="Whisper pipeline not loaded")
        
        if len(files) > 10:  # 배치 크기 제한
            raise HTTPException(status_code=400, detail="Too many files. Maximum 10 files per batch.")
        
        results = []
        for i, audio_file in enumerate(files):
            try:
                logger.info(f"Processing batch file {i+1}/{len(files)}: {audio_file.filename}")
                
                # 개별 파일 처리
                audio_array = preprocess_audio(audio_file)
                duration = len(audio_array) / 16000
                
                generate_kwargs = {
                    "task": task,
                    "return_timestamps": return_timestamps,
                }
                
                if language:
                    generate_kwargs["language"] = language
                
                # 일부 파라미터는 파이프라인 상위 인자로 전달해야 동작함 (특히 return_timestamps)
                batch_top_level_kwargs = {}
                if "return_timestamps" in generate_kwargs:
                    batch_top_level_kwargs["return_timestamps"] = generate_kwargs.pop("return_timestamps")

                if duration > 30.0 and not batch_top_level_kwargs.get("return_timestamps", False):
                    logger.info("Input >30s in batch. Enabling return_timestamps automatically.")
                    batch_top_level_kwargs["return_timestamps"] = True
                    batch_top_level_kwargs.setdefault("chunk_length_s", 30)
                    batch_top_level_kwargs.setdefault("stride_length_s", [4, 2])

                result = whisper_pipeline(
                    audio_array,
                    **batch_top_level_kwargs,
                    generate_kwargs=generate_kwargs,
                )
                
                file_result = {
                    "filename": audio_file.filename,
                    "text": result["text"].strip(),
                    "language": "auto-detected",
                    "duration": duration
                }
                
                if return_timestamps and "chunks" in result:
                    file_result["segments"] = [
                        {
                            "start": chunk["timestamp"][0] if chunk["timestamp"] else 0.0,
                            "end": chunk["timestamp"][1] if chunk["timestamp"] else duration,
                            "text": chunk["text"].strip()
                        }
                        for chunk in result["chunks"]
                    ]
                
                results.append(file_result)
                
            except Exception as e:
                logger.error(f"Failed to process {audio_file.filename}: {str(e)}")
                results.append({
                    "filename": audio_file.filename,
                    "error": str(e)
                })
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Batch transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch transcription failed: {str(e)}")

@app.get("/models")
async def get_available_models():
    """사용 가능한 Whisper 모델 목록"""
    return {
        "available_models": [
            "openai/whisper-tiny", "openai/whisper-tiny.en",
            "openai/whisper-base", "openai/whisper-base.en", 
            "openai/whisper-small", "openai/whisper-small.en",
            "openai/whisper-medium", "openai/whisper-medium.en",
            "openai/whisper-large", "openai/whisper-large-v2",
            "openai/whisper-large-v3", "openai/whisper-large-v3-turbo"
        ],
        "current_model": "openai/whisper-large-v3-turbo",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_dtype": "float16" if torch.cuda.is_available() else "float32"
    }

@app.get("/languages")
async def get_supported_languages():
    """지원되는 언어 목록"""
    # Whisper 지원 언어 목록 (주요 언어만)
    languages = {
        "en": "english", "ko": "korean", "ja": "japanese", "zh": "chinese",
        "es": "spanish", "fr": "french", "de": "german", "it": "italian",
        "pt": "portuguese", "ru": "russian", "ar": "arabic", "hi": "hindi",
        "th": "thai", "vi": "vietnamese", "id": "indonesian", "ms": "malay",
        "tl": "tagalog", "uk": "ukrainian", "pl": "polish", "nl": "dutch",
        "sv": "swedish", "da": "danish", "no": "norwegian", "fi": "finnish"
    }
    return {
        "languages": languages,
        "total_supported": len(languages),
        "note": "Set language=None for automatic detection. Full list: 99 languages supported"
    }

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "pipeline_loaded": whisper_pipeline is not None,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "model_id": "openai/whisper-large-v3-turbo"
    }

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Whisper ASR API Server",
        "endpoints": {
            "transcribe": "POST /transcribe - Single file transcription",
            "transcribe_batch": "POST /transcribe_batch - Batch transcription",
            "models": "GET /models - Available models",
            "languages": "GET /languages - Supported languages",
            "health": "GET /health - Health check"
        }
    }

'''
WSL Ubuntu 환경에서 실행:

# 가상환경 활성화
source /home/hanati/code/whisper_server_only/.venv/bin/activate

# 서버 실행
nohup uvicorn whisper_server:app --host 0.0.0.0 --port 7010 > server.log 2>&1 &

# 테스트
curl -X POST "http://127.0.0.1:7010/transcribe" \
  -F "audio=@sample.wav" \
  -F "task=transcribe" \
  -F "language=ko" \
  -F "return_timestamps=true"

curl -X GET "http://127.0.0.1:7010/health"
'''
