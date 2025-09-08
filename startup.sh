#!/bin/bash

# LLM 추론 서버에 필요한 패키지 설치
python -m pip install uvicorn FastAPI transformers sentence_transformers accelerate bitsandbytes

# 여기에 다른 시작 명령어를 추가할 수 있습니다.
# 예: python main.py 또는 uvicorn main:app --host 0.0.0.0 --port 8000
