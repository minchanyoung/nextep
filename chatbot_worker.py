#!/usr/bin/env python3
"""
Oracle과 완전 분리된 독립 챗봇 워커 프로세스
subprocess로 실행되어 메모리 공간을 완전히 격리
"""

import os
import sys
import json
import tempfile
import logging
from typing import Dict, List

# Oracle 관련 모든 모듈 임포트 차단
sys.modules['oracledb'] = None
sys.modules['cx_Oracle'] = None

# 환경 변수로 Oracle 초기화 완전 비활성화
os.environ['SKIP_ORACLE_INIT'] = '1'
os.environ['DATABASE_URI'] = 'sqlite:///:memory:'  # 메모리 SQLite 사용

def setup_logging():
    """격리된 로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='[CHATBOT-WORKER] %(levelname)s: %(message)s'
    )
    return logging.getLogger(__name__)

def init_llm_services():
    """LLM 서비스만 초기화 (Oracle 없이)"""
    try:
        # Flask 앱 초기화 (Oracle 없이)
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

        from app.llm_service import LLMService
        from app.rag_manager import RAGManager
        from app.prompt_templates import prompt_manager

        # LLM 서비스 초기화
        llm_service = LLMService()
        llm_service._load_timeouts_from_settings()

        # RAG 매니저 초기화 (Vector DB만 사용)
        rag_manager = RAGManager()

        return llm_service, rag_manager, prompt_manager

    except Exception as e:
        logger.error(f"LLM 서비스 초기화 실패: {e}")
        return None, None, None

def process_chat_request(request_data: Dict) -> Dict:
    """챗봇 요청 처리"""
    try:
        user_message = request_data.get('message', '').strip()
        chat_history = request_data.get('chat_history', [])
        context_summary = request_data.get('context_summary', '')
        is_streaming = request_data.get('streaming', False)

        if not user_message:
            return {'error': '메시지가 없습니다.'}

        llm_service, rag_manager, prompt_manager = init_llm_services()

        if not llm_service:
            return {'error': 'LLM 서비스 초기화 실패'}

        # RAG 검색
        additional_context = ""
        if rag_manager:
            try:
                additional_context = rag_manager.get_career_advice(user_message)
            except Exception as e:
                logger.warning(f"RAG 검색 실패: {e}")

        # 대화 히스토리 구성
        history_text = ""
        if chat_history:
            recent_history = chat_history[-6:]
            for msg in recent_history:
                role = "사용자" if msg.get("role") == "user" else "AI"
                content = msg.get("content", "")
                history_text += f"{role}: {content}\n"

        # 시스템 프롬프트
        system_prompt = prompt_manager.get_system_prompt("conversational")

        # 메시지 구성
        messages = [
            {
                "role": "system",
                "content": f"""{system_prompt}

대화 맥락: {context_summary or "없음"}
최근 대화: {history_text or "없음"}
참고 정보: {additional_context or "없음"}

사용자의 질문에 대해 구체적이고 실용적인 조언을 한국어로 제공해주세요."""
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

        if is_streaming:
            # 스트리밍 응답
            chunks = []
            for chunk in llm_service.chat_stream(messages):
                chunks.append(chunk)
            return {'response': ''.join(chunks), 'chunks': chunks}
        else:
            # 동기 응답
            response = llm_service.chat_sync(messages)
            return {'response': response}

    except Exception as e:
        logger.error(f"챗봇 요청 처리 실패: {e}")
        return {'error': f'처리 중 오류: {str(e)}'}

def main():
    """메인 실행 함수"""
    global logger
    logger = setup_logging()

    if len(sys.argv) != 3:
        logger.error("사용법: chatbot_worker.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        # 입력 파일에서 요청 데이터 읽기
        with open(input_file, 'r', encoding='utf-8') as f:
            request_data = json.load(f)

        logger.info(f"챗봇 요청 처리 시작: {request_data.get('message', '')[:50]}")

        # 요청 처리
        result = process_chat_request(request_data)

        # 결과를 출력 파일에 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info("챗봇 요청 처리 완료")

    except Exception as e:
        logger.error(f"워커 프로세스 실행 실패: {e}")
        # 오류 결과 저장
        error_result = {'error': f'워커 프로세스 오류: {str(e)}'}
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, ensure_ascii=False, indent=2)
        except:
            pass
        sys.exit(1)

if __name__ == '__main__':
    main()