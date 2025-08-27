#!/usr/bin/env python3
"""
모델 사전 로딩 스크립트
Flask 서버 시작 전에 Ollama 모델들을 메모리에 사전 로딩합니다.
"""

import os
import sys
import time
import requests
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_ollama_server():
    """Ollama 서버가 실행 중인지 확인"""
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        return response.status_code == 200
    except:
        return False

def preload_model(model_name, keep_alive="60m"):
    """단일 모델을 사전 로딩"""
    try:
        logger.info(f"사전 로딩 시작: {model_name}")
        start_time = time.time()
        
        # Ollama API를 통해 모델 로딩
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": "Hello",
                "stream": False,
                "options": {
                    "num_predict": 1,  # 최소한의 생성
                },
                "keep_alive": keep_alive
            },
            timeout=120
        )
        
        load_time = time.time() - start_time
        
        if response.status_code == 200:
            logger.info(f"{model_name} 로딩 완료 ({load_time:.2f}초)")
            return True, load_time
        else:
            logger.error(f"{model_name} 로딩 실패: HTTP {response.status_code}")
            return False, load_time
            
    except requests.exceptions.Timeout:
        logger.error(f"{model_name} 로딩 타임아웃")
        return False, 0
    except Exception as e:
        logger.error(f"{model_name} 로딩 오류: {e}")
        return False, 0

def preload_embedding_model(model_name, keep_alive="60m"):
    """임베딩 모델을 사전 로딩"""
    try:
        logger.info(f"임베딩 모델 사전 로딩 시작: {model_name}")
        start_time = time.time()
        
        # 임베딩 API를 통해 모델 로딩
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": model_name,
                "prompt": "test",
                "options": {
                    "num_thread": -1,
                },
                "keep_alive": keep_alive
            },
            timeout=120
        )
        
        load_time = time.time() - start_time
        
        if response.status_code == 200:
            logger.info(f"{model_name} 임베딩 모델 로딩 완료 ({load_time:.2f}초)")
            return True, load_time
        else:
            logger.error(f"{model_name} 임베딩 모델 로딩 실패: HTTP {response.status_code}")
            return False, load_time
            
    except requests.exceptions.Timeout:
        logger.error(f"{model_name} 임베딩 모델 로딩 타임아웃")
        return False, 0
    except Exception as e:
        logger.error(f"{model_name} 임베딩 모델 로딩 오류: {e}")
        return False, 0

def get_models_to_preload():
    """사전 로딩할 모델 목록 반환"""
    # 환경 변수 또는 기본값 사용
    generation_model = os.environ.get('OLLAMA_MODEL', 'exaone3.5:7.8b')
    embedding_model = os.environ.get('OLLAMA_EMBEDDING_MODEL', 'llama3')
    keep_alive = os.environ.get('OLLAMA_KEEP_ALIVE', '60m')
    
    return {
        'generation': generation_model,
        'embedding': embedding_model,
        'keep_alive': keep_alive
    }

def preload_all_models():
    """모든 필요한 모델을 사전 로딩"""
    logger.info("모델 사전 로딩 시작")
    
    # Ollama 서버 확인
    if not check_ollama_server():
        logger.error("Ollama 서버가 실행되지 않았습니다. ollama serve를 먼저 실행하세요.")
        return False
    
    models = get_models_to_preload()
    logger.info(f"사전 로딩 대상 모델:")
    logger.info(f"  - 생성 모델: {models['generation']}")
    logger.info(f"  - 임베딩 모델: {models['embedding']}")
    logger.info(f"  - 메모리 유지 시간: {models['keep_alive']}")
    
    success_count = 0
    total_time = 0
    
    # ThreadPoolExecutor를 사용하여 병렬 로딩
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(preload_model, models['generation'], models['keep_alive']): "generation",
            executor.submit(preload_embedding_model, models['embedding'], models['keep_alive']): "embedding"
        }
        
        for future in as_completed(futures):
            model_type = futures[future]
            try:
                success, load_time = future.result()
                if success:
                    success_count += 1
                total_time += load_time
            except Exception as e:
                logger.error(f"{model_type} 모델 로딩 중 오류: {e}")
    
    logger.info(f"모델 사전 로딩 완료: {success_count}/2 성공, 총 소요시간 {total_time:.2f}초")
    return success_count == 2

def main():
    """메인 실행 함수"""
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        # 모델 상태만 확인
        try:
            response = requests.get("http://localhost:11434/api/ps", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                logger.info(f"현재 메모리에 로딩된 모델: {len(models)}개")
                for model in models:
                    logger.info(f"  - {model.get('name', 'Unknown')}: {model.get('size_vram', 'Unknown')} 메모리 사용")
            else:
                logger.error("모델 상태 확인 실패")
        except Exception as e:
            logger.error(f"모델 상태 확인 오류: {e}")
        return
    
    # 실제 사전 로딩 수행
    success = preload_all_models()
    if success:
        logger.info("모든 모델 사전 로딩 완료! Flask 서버를 시작할 수 있습니다.")
        sys.exit(0)
    else:
        logger.error("모델 사전 로딩 실패!")
        sys.exit(1)

if __name__ == "__main__":
    main()