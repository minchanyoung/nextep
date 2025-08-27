#!/usr/bin/env python3
"""
NEXTEP 서버 시작 스크립트
모델 사전 로딩과 Flask 서버 시작을 통합 관리합니다.
"""

import os
import sys
import time
import subprocess
import argparse

def check_ollama_running():
    """Ollama 서버가 실행 중인지 확인"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/version", timeout=3)
        return response.status_code == 200
    except:
        return False

def start_ollama_server():
    """Ollama 서버 시작"""
    print("Ollama 서버를 시작합니다...")
    try:
        # Windows에서 Ollama 서버 시작
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        
        # 서버가 시작될 때까지 대기
        for i in range(30):  # 최대 30초 대기
            if check_ollama_running():
                print("Ollama 서버가 성공적으로 시작되었습니다!")
                return True
            time.sleep(1)
            if i % 5 == 0:
                print(f"Ollama 서버 시작 대기 중... ({i+1}/30초)")
        
        print("Ollama 서버 시작 실패 (타임아웃)")
        return False
        
    except Exception as e:
        print(f"Ollama 서버 시작 중 오류: {e}")
        return False

def preload_models():
    """모델 사전 로딩 실행"""
    print("모델 사전 로딩을 시작합니다...")
    try:
        result = subprocess.run([sys.executable, "preload_models.py"], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("모델 사전 로딩 완료!")
            return True
        else:
            print("모델 사전 로딩 실패:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("모델 사전 로딩 타임아웃 (5분 초과)")
        return False
    except Exception as e:
        print(f"모델 사전 로딩 중 오류: {e}")
        return False

def start_flask_server(skip_preload=False):
    """Flask 서버 시작"""
    print("Flask 서버를 시작합니다...")
    
    env = os.environ.copy()
    if skip_preload:
        env['SKIP_MODEL_PRELOAD'] = '1'
    
    try:
        subprocess.run([sys.executable, "run.py"], env=env)
    except KeyboardInterrupt:
        print("\n 서버가 중단되었습니다.")
    except Exception as e:
        print(f"Flask 서버 시작 중 오류: {e}")

def main():
    parser = argparse.ArgumentParser(description="NEXTEP 서버 시작 스크립트")
    parser.add_argument("--skip-preload", action="store_true", 
                       help="모델 사전 로딩을 건너뜁니다")
    parser.add_argument("--check-models", action="store_true",
                       help="현재 로딩된 모델 상태만 확인합니다")
    parser.add_argument("--start-ollama", action="store_true",
                       help="Ollama 서버도 함께 시작합니다")
    
    args = parser.parse_args()
    
    # 모델 상태만 확인
    if args.check_models:
        subprocess.run([sys.executable, "preload_models.py", "--check"])
        return
    
    print("=" * 60)
    print("NEXTEP 서버 시작 프로세스")
    print("=" * 60)
    
    # 1. Ollama 서버 확인/시작
    if not check_ollama_running():
        if args.start_ollama:
            if not start_ollama_server():
                print("Ollama 서버를 시작할 수 없습니다. 수동으로 'ollama serve'를 실행해주세요.")
                return
        else:
            print("Ollama 서버가 실행되지 않았습니다.")
            print("'ollama serve'를 먼저 실행하거나 --start-ollama 옵션을 사용하세요.")
            return
    else:
        print("Ollama 서버가 이미 실행 중입니다.")
    
    # 2. 모델 사전 로딩
    if not args.skip_preload:
        preload_success = preload_models()
        if not preload_success:
            print("모델 사전 로딩에 실패했지만 서버를 시작합니다.")
    else:
        print("모델 사전 로딩을 건너뜁니다.")
    
    # 3. Flask 서버 시작
    start_flask_server(skip_preload=args.skip_preload)

if __name__ == "__main__":
    main()