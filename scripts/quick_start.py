#!/usr/bin/env python3
"""
NEXTEP 빠른 시작 스크립트
개발 중 빠른 테스트를 위해 모든 사전 로딩을 건너뛰고 서버를 시작합니다.
"""

import os
import subprocess
import sys

def main():
    print("=" * 60)
    print("NEXTEP 빠른 시작 모드")
    print("=" * 60)
    print("설정:")
    print("  - ML 모델 로딩: (필수)")
    print("  - LLM 모델 사전 로딩: (건너뜀)")
    print("  - RAG 임베딩 생성: (첫 요청 시)")
    print("=" * 60)
    print("예상 시작 시간: 5-10초")
    print("첫 번째 AI 조언 요청은 느릴 수 있습니다 (30-60초)")
    print("=" * 60)
    
    # 환경변수 설정
    env = os.environ.copy()
    env['SKIP_MODEL_PRELOAD'] = '1'
    env['FAST_START'] = '1'
    
    try:
        # Flask 서버 시작
        subprocess.run([sys.executable, "run.py"], env=env)
    except KeyboardInterrupt:
        print("\n 서버가 중단되었습니다.")
    except Exception as e:
        print(f"서버 시작 중 오류: {e}")

if __name__ == "__main__":
    main()