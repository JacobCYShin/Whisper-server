FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    python3 \
    python3-pip \
    git \
    wget \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# pip 최신화
RUN python3 -m pip install --upgrade pip

WORKDIR /app

# 빌드 시 필요한 파일들만 복사
COPY requirements.txt /app/

# 런타임에 마운트될 파일들은 복사하지 않음
# - whisper_server.py (런타임에 마운트)
# - test.py (런타임에 마운트)
# - run.sh (런타임에 마운트)

# 추가 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
RUN pip3 install -r requirements.txt
