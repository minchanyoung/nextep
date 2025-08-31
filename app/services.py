"""
LangChain 기반 통합 서비스 레이어 (완전 마이그레이션 버전)
기존 서비스들을 모두 LangChain으로 완전 교체
"""

import sys
import bcrypt
import json
import os
import logging
from typing import List, Dict, Iterator, Optional
from flask import current_app

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.llm_service import LLMService, get_llm_service
from app.rag_manager import RAGManager, get_rag_manager
from app import db
from app.models import User
from app.rag_data import LABOR_MARKET_TRENDS, LEARNING_RECOMMENDATIONS
from app.utils.db_utils import safe_db_operation, execute_db_transaction
from app.prompt_templates import prompt_manager
from app.core.exceptions import LLMServiceError, RAGError
import numpy as np

# 로거 인스턴스 가져오기
logger = logging.getLogger(__name__)





def chat_complete(messages, temperature=0.6, num_ctx=8192) -> str:
    """LangChain 기반 채팅 완성 (기존 API 호환)"""
    try:
        llm_service: LLMService = current_app.extensions["llm_service"]
        response = llm_service.chat_sync(
            messages, 
            options={"temperature": float(temperature), "num_ctx": int(num_ctx)}
        )
        
        # 응답이 문자열이 아닐 경우 문자열로 강제 변환
        if not isinstance(response, str):
            logger.warning(f"chat_complete에서 예상치 못한 응답 형식 수신: {response}. 문자열로 변환합니다.")
            return str(response)
        return response
        
    except Exception as e:
        logger.error(f"LangChain 채팅 완성 오류: {e}")
        raise LLMServiceError(f"채팅 완성 실패: {str(e)}")


def chat_stream(messages, temperature=0.6, num_ctx=8192):
    """LangChain 기반 스트리밍 응답 (기존 API 호환)"""
    try:
        llm_service: LLMService = current_app.extensions["llm_service"]
        for chunk in llm_service.chat_stream(
            messages, 
            options={"temperature": float(temperature), "num_ctx": int(num_ctx)}
        ):
            yield chunk
    except Exception as e:
        logger.error(f"LangChain 스트리밍 중 오류 발생: {e}")
        yield f"[오류] 응답 생성 중 문제가 발생했습니다: {str(e)}"


def get_user_by_username(username):
    """사용자 이름으로 User 객체를 조회합니다."""
    return User.query.filter_by(username=username).first()


@safe_db_operation("사용자 프로필 업데이트")
def update_user_profile(user_id, age, gender, education, monthly_income, job_category, job_satisfaction, **satisfaction_factors):
    """사용자 프로필 정보를 업데이트합니다."""
    user = User.query.get(user_id)
    if user:
        user.age = age
        user.gender = gender
        user.education = education
        user.monthly_income = monthly_income
        user.job_category = job_category
        user.job_satisfaction = job_satisfaction
        
        # 9개 만족도 요인 업데이트
        satisfaction_fields = [
            'satis_wage', 'satis_stability', 'satis_growth', 'satis_task_content',
            'satis_work_env', 'satis_work_time', 'satis_communication', 
            'satis_fair_eval', 'satis_welfare'
        ]
        
        for field in satisfaction_fields:
            if field in satisfaction_factors:
                setattr(user, field, satisfaction_factors[field])
        
        return True
    return False


def verify_user(username, password):
    """사용자 이름과 비밀번호를 확인하여 로그인 인증을 수행합니다."""
    user = User.query.filter_by(username=username).first()
    if user and bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
        return True
    return False


def create_user(username, password, email):
    """새로운 사용자를 데이터베이스에 추가합니다."""
    # 사용자 이름 또는 이메일 중복 확인
    if User.query.filter_by(username=username).first():
        return '이미 존재하는 사용자 이름입니다.'
    if User.query.filter_by(email=email).first():
        return '이미 존재하는 이메일입니다.'

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    new_user = User(username=username, email=email, password=hashed_password)
    try:
        db.session.add(new_user)
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        logger.error(f"회원가입 중 데이터베이스 오류 발생: {e}")
        return '회원가입 중 오류가 발생했습니다.'


def run_prediction(user_input):
    """사용자 입력을 받아 예측 시나리오를 구성하고, 머신러닝 모델을 호출합니다."""
    from app.ml.preprocessing import prepare_prediction_features
    from app.ml import routes as ml_routes

    # MLPredictor 클래스를 여기서 정의하여 필요한 데이터만 명시적으로 전달
    class MLPredictor:
        def __init__(self):
            ml_resources = current_app.extensions.get('ml_resources', {})
            self.job_category_stats = ml_resources.get('job_category_stats')
            self.klips_df = ml_resources.get('klips_df')

    ml_predictor_instance = MLPredictor()

    if ml_predictor_instance.job_category_stats is None:
        logger.error("job_category_stats가 초기화되지 않았습니다. ML 모델이 제대로 로드되지 않았습니다.")
        raise RuntimeError("ML 모델이 제대로 초기화되지 않았습니다. 애플리케이션을 다시 시작해주세요.")

    # 새로운 피처 매칭 방식 사용
    logger.info("새로운 피처 매칭 방식으로 예측을 시작합니다.")
    results = ml_routes.run_prediction_with_proper_features(user_input)
    logger.info(f"예측 결과 받음: {results}")
    
    # KLIPS 취업자 데이터(월소득 100만원+) 기반 현실적 범위 적용
    # 무소득→취업 등 특수한 케이스를 제외한 일반 취업자 분포 기준
    # 취업자(88.7%) 분석 결과: 90.7%가 -50%~+50% 범위, 95분위수 80%
    # 평균 14.47%, 표준편차 18.97% (훨씬 현실적)
    
    REALISTIC_MIN_INCOME_CHANGE = -0.50  # -50% (취업자 1분위수)
    REALISTIC_MAX_INCOME_CHANGE = 0.50   # +50% (취업자 90분위수, 일반적 상한)
    REALISTIC_MIN_SATIS_CHANGE = -2.0    # -2점 (KLIPS 실제 분포)
    REALISTIC_MAX_SATIS_CHANGE = 2.0     # +2점 (KLIPS 실제 분포)
    
    for result in results:
        if 'income_change_rate' in result:
            original_rate = float(result['income_change_rate'])
            # 현실적 범위로 제한 (KLIPS 94.7% 데이터 범위)
            realistic_rate = max(REALISTIC_MIN_INCOME_CHANGE, 
                               min(REALISTIC_MAX_INCOME_CHANGE, original_rate))
            result['income_change_rate'] = float(realistic_rate)
            
            if abs(original_rate - realistic_rate) > 0.001:
                logger.warning(f"비현실적 예측값 조정: {original_rate*100:.2f}% -> {realistic_rate*100:.2f}%")
        
        if 'satisfaction_change_score' in result:
            original_satis = float(result['satisfaction_change_score'])
            # 현실적 범위로 제한
            realistic_satis = max(REALISTIC_MIN_SATIS_CHANGE,
                                min(REALISTIC_MAX_SATIS_CHANGE, original_satis))
            result['satisfaction_change_score'] = float(realistic_satis)
            
            if abs(original_satis - realistic_satis) > 0.001:
                logger.warning(f"비현실적 만족도 조정: {original_satis:.3f}점 -> {realistic_satis:.3f}점")
    
    logger.info(f"KLIPS 현실적 범위 적용 완료: {results}")
    return results


def example_db_query():
    """데이터베이스 연결을 테스트합니다."""
    try:
        db.session.execute(db.text("SELECT 1 FROM DUAL"))
        return "DB 연결 성공"
    except Exception as e:
        logger.error(f"DB 테스트 실패: {e}")
        return "DB 연결 실패"


def retrieve_labor_market_info(query_text: str, top_n: int = 2) -> str:
    """LangChain 기반 노동시장 트렌드 정보 검색"""
    try:
        rag_manager = get_rag_manager()
        if not rag_manager:
            logger.warning("RAG Manager를 사용할 수 없습니다.")
            return ""
        
        # LangChain RAG를 통한 검색
        result = rag_manager.get_labor_market_info(query_text, top_k=top_n)
        return result if result else ""
        
    except Exception as e:
        logger.error(f"노동시장 정보 검색 실패: {e}")
        return ""


def retrieve_learning_recommendations(query_text: str, top_n: int = 2) -> str:
    """LangChain 기반 학습 추천 정보 검색"""
    try:
        rag_manager = get_rag_manager()
        if not rag_manager:
            logger.warning("RAG Manager를 사용할 수 없습니다.")
            return ""
        
        # LangChain RAG를 통한 검색
        result = rag_manager.get_learning_recommendations(query_text, top_k=top_n)
        return result if result else ""
        
    except Exception as e:
        logger.error(f"학습 추천 검색 실패: {e}")
        return ""


def generate_career_advice_hf(user_input, prediction_results, job_category_map, satis_factor_map):
    """LangChain 기반 커리어 조언 생성 (PDF 데이터 통합 RAG)"""
    try:
        llm_service = get_llm_service()
        rag_manager = get_rag_manager()
        
        if not llm_service:
            raise LLMServiceError("LLM 서비스를 사용할 수 없습니다.")
        
        # RAG 검색을 위한 통합 쿼리 생성
        current_job_name = job_category_map.get(user_input.get('current_job_category', ''), '알 수 없음')
        job_a_name = job_category_map.get(user_input.get('job_A_category', ''), '알 수 없음')
        job_b_name = job_category_map.get(user_input.get('job_B_category', ''), '알 수 없음')
        focus_key_name = satis_factor_map.get(user_input.get('satis_focus_key'), '지정되지 않음')
        
        comprehensive_query = f"{current_job_name} 직업의 전망과 {focus_key_name} 가치를 높이기 위한 역량 개발 방법. {job_a_name} 또는 {job_b_name}으로의 이직 고려. 2024년 및 2025년 노동 시장 동향 포함."
        
        rag_context = ""
        if rag_manager:
            # 통합된 쿼리로 PDF 전체에서 관련 정보 검색
            rag_context = rag_manager.get_career_advice(comprehensive_query, top_k=5)

        # 프롬프트 템플릿 사용 (통합된 RAG 컨텍스트 전달)
        messages = prompt_manager.get_career_advice_prompt(
            user_input, prediction_results, job_category_map, satis_factor_map,
            labor_market_context=rag_context,  # labor_market_context 자리에 통합 컨텍스트 전달
            learning_context=""  # learning_context는 비움
        )
        
        # LangChain으로 응답 생성
        response = llm_service.chat_sync(messages)
        return response
        
    except Exception as e:
        logger.error(f"커리어 조언 생성 실패: {e}")
        raise LLMServiceError(f"커리어 조언 생성 중 오류: {str(e)}")


def generate_follow_up_advice(user_message: str, chat_history: List[Dict], context_summary: str = "") -> str:
    """LangChain 기반 추가 질문 응답"""
    try:
        llm_service = get_llm_service()
        rag_manager = get_rag_manager()
        
        if not llm_service:
            raise LLMServiceError("LLM 서비스를 사용할 수 없습니다.")
        
        # RAG 검색으로 관련 정보 찾기
        additional_context = ""
        if rag_manager:
            try:
                additional_context = rag_manager.get_career_advice(user_message)
            except Exception as e:
                logger.warning(f"RAG 검색 실패, 기본 응답 모드로 전환: {e}")
        
        # 대화 히스토리 포맷팅
        history_text = ""
        if chat_history:
            recent_history = chat_history[-6:]  # 최근 3회 대화만 포함
            for msg in recent_history:
                role = "사용자" if msg.get("role") == "user" else "AI"
                content = msg.get("content", "")
                history_text += f"{role}: {content}\n"
        
        # LangChain 프롬프트 템플릿
        template = """당신은 친근하고 전문적인 커리어 컨설턴트입니다. 
사용자의 추가 질문에 대해 맥락을 고려하여 도움이 되는 답변을 제공해주세요.

## 대화 맥락
{context_summary}

## 최근 대화 기록
{chat_history}

## 관련 참고 정보
{additional_context}

## 사용자의 현재 질문
{user_question}

위 정보들을 종합하여 사용자의 질문에 대해 구체적이고 실용적인 조언을 제공해주세요. 
한국어로 자연스럽게 답변하며, 가능한 한 실행 가능한 구체적인 방향을 제시해주세요."""

        prompt = ChatPromptTemplate.from_template(template)
        
        # 체인 구성 및 실행
        chain = prompt | llm_service.chat_model | StrOutputParser()
        
        response = chain.invoke({
            "context_summary": context_summary or "이전 맥락 정보 없음",
            "chat_history": history_text or "이전 대화 기록 없음",
            "additional_context": additional_context or "추가 참고 정보 없음",
            "user_question": user_message
        })
        
        return response
        
    except Exception as e:
        logger.error(f"추가 조언 생성 실패: {e}")
        raise LLMServiceError(f"추가 조언 생성 중 오류: {str(e)}")


def generate_follow_up_advice_stream(user_message: str, chat_history: List[Dict], context_summary: str = "") -> Iterator[str]:
    """LangChain 기반 스트리밍 추가 질문 응답"""
    try:
        llm_service = get_llm_service()
        rag_manager = get_rag_manager()
        
        if not llm_service:
            raise LLMServiceError("LLM 서비스를 사용할 수 없습니다.")
        
        # RAG 검색
        additional_context = ""
        if rag_manager:
            try:
                additional_context = rag_manager.get_career_advice(user_message)
            except Exception as e:
                logger.warning(f"RAG 검색 실패: {e}")
        
        # 대화 히스토리 포맷팅
        history_text = ""
        if chat_history:
            recent_history = chat_history[-6:]
            for msg in recent_history:
                role = "사용자" if msg.get("role") == "user" else "AI"
                content = msg.get("content", "")
                history_text += f"{role}: {content}\n"
        
        # 스트리밍용 메시지 구성
        messages = [
            {
                "role": "system", 
                "content": f"""당신은 친근하고 전문적인 커리어 컨설턴트입니다.

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
        
        # 스트리밍 응답 생성
        for chunk in llm_service.chat_stream(messages):
            yield chunk
                
    except Exception as e:
        logger.error(f"스트리밍 조언 생성 실패: {e}")
        yield "스트리밍 응답 생성 중 오류가 발생했습니다."


def summarize_context(user_input: Dict, prediction_results: List, job_category_map: Dict, satis_factor_map: Dict) -> str:
    """컨텍스트 요약 (대화 세션용)"""
    try:
        current_job = job_category_map.get(user_input.get('current_job_category', ''), '알 수 없음')
        focus_key = satis_factor_map.get(user_input.get('satis_focus_key'), '지정되지 않음')
        
        summary = f"""사용자 정보: {user_input.get('age', '?')}세, {current_job}, 월소득 {user_input.get('monthly_income', '?')}만원
중요 가치: {focus_key}
AI 예측을 바탕으로 커리어 조언을 제공했습니다."""
        
        return summary
        
    except Exception as e:
        logger.error(f"컨텍스트 요약 실패: {e}")
        return "사용자의 커리어 예측 및 조언을 제공했습니다."


def get_enhanced_career_advice(user_message: str, rag_results: List[Dict]) -> str:
    """RAG 결과를 활용한 향상된 커리어 조언"""
    try:
        llm_service = get_llm_service()
        if not llm_service:
            raise LLMServiceError("LLM 서비스를 사용할 수 없습니다.")
        
        # RAG 결과 포맷팅
        context = prompt_manager.get_rag_enhanced_context(user_message, rag_results)
        
        # 향상된 조언 생성 프롬프트
        template = """당신은 한국의 전문 커리어 컨설턴트입니다.
아래 전문 자료들을 참고하여 사용자의 질문에 대해 구체적이고 실용적인 조언을 제공해주세요.

{context}

사용자 질문: {question}

전문 자료를 바탕으로 3-5개의 구체적인 조언을 제공해주세요:"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm_service.chat_model | StrOutputParser()
        
        response = chain.invoke({
            "context": context,
            "question": user_message
        })
        
        return response
        
    except Exception as e:
        logger.error(f"향상된 커리어 조언 생성 실패: {e}")
        raise LLMServiceError(f"향상된 조언 생성 중 오류: {str(e)}")


# 호환성을 위한 유틸리티 함수들
from app.utils.math_utils import cosine_similarity
from app.utils.flask_utils import get_llm_service as _get_llm_service