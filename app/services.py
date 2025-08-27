import sys
import bcrypt
import json
import os
import logging
from flask import current_app
from app.llm_service import LLMService
from app import db
from app.models import User
from app.rag_data import LABOR_MARKET_TRENDS, LEARNING_RECOMMENDATIONS
from app.utils.db_utils import safe_db_operation, execute_db_transaction
import numpy as np

# 로거 인스턴스 가져오기
logger = logging.getLogger(__name__)

# RAG 데이터 임베딩 캐시
_labor_market_embeddings = None
_learning_recommendation_embeddings = None

def initialize_rag_embeddings():
    """
    Flask 앱 시작 시 RAG 임베딩을 사전 생성합니다.
    """
    global _labor_market_embeddings, _learning_recommendation_embeddings
    
    llm_service = _get_llm_service()
    if not llm_service:
        logger.warning("LLMService를 사용할 수 없어 RAG 임베딩 초기화를 건너뜁니다.")
        return False
    
    try:
        logger.info("RAG 임베딩 사전 생성 시작...")
        
        # 노동 시장 트렌드 임베딩 생성
        if _labor_market_embeddings is None:
            _labor_market_embeddings = _generate_embeddings_for_rag_data(
                LABOR_MARKET_TRENDS, llm_service.ollama_embedding_model
            )
            logger.info(f"노동 시장 트렌드 임베딩 생성 완료: {len(_labor_market_embeddings)}개")
        
        # 학습 추천 임베딩 생성  
        if _learning_recommendation_embeddings is None:
            _learning_recommendation_embeddings = _generate_embeddings_for_rag_data(
                LEARNING_RECOMMENDATIONS, llm_service.ollama_embedding_model
            )
            logger.info(f"학습 추천 임베딩 생성 완료: {len(_learning_recommendation_embeddings)}개")
        
        logger.info("RAG 임베딩 사전 생성 완료!")
        return True
        
    except Exception as e:
        logger.error(f"RAG 임베딩 초기화 중 오류: {e}")
        return False

def chat_complete(messages, temperature=0.6, num_ctx=8192) -> str:
    llm_service: LLMService = current_app.extensions["llm_service"]
    response = llm_service.chat_sync(messages, options={"temperature": float(temperature), "num_ctx": int(num_ctx)})
    # 응답이 문자열이 아닐 경우 문자열로 강제 변환
    if not isinstance(response, str):
        logger.warning(f"chat_complete에서 예상치 못한 응답 형식 수신: {response}. 문자열로 변환합니다.")
        return str(response)
    return response

def chat_stream(messages, temperature=0.6, num_ctx=8192):
    """스트리밍 응답을 위한 제너레이터 함수"""
    llm_service: LLMService = current_app.extensions["llm_service"]
    try:
        for chunk in llm_service.chat_stream(messages, options={"temperature": float(temperature), "num_ctx": int(num_ctx)}):
            yield chunk
    except Exception as e:
        logger.error(f"스트리밍 중 오류 발생: {e}")
        yield f"[오류] 응답 생성 중 문제가 발생했습니다: {str(e)}"

def get_user_by_username(username):
    """
    사용자 이름으로 User 객체를 조회합니다.
    """
    return User.query.filter_by(username=username).first()

@safe_db_operation("사용자 프로필 업데이트")
def update_user_profile(user_id, age, gender, education, monthly_income, job_category, job_satisfaction, satis_focus_key):
    """
    사용자 프로필 정보를 업데이트합니다.
    """
    user = User.query.get(user_id)
    if user:
        user.age = age
        user.gender = gender
        user.education = education
        user.monthly_income = monthly_income
        user.job_category = job_category
        user.job_satisfaction = job_satisfaction
        user.satis_focus_key = satis_focus_key
        return True
    return False

def verify_user(username, password):
    """
    사용자 이름과 비밀번호를 확인하여 로그인 인증을 수행합니다.
    """
    user = User.query.filter_by(username=username).first()
    if user and bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
        return True
    return False

def create_user(username, password, email):
    """
    새로운 사용자를 데이터베이스에 추가합니다.
    """
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

from app.ml.preprocessing import prepare_prediction_features

def run_prediction(user_input):
    """
    사용자 입력을 받아 예측 시나리오를 구성하고, 머신러닝 모델을 호출합니다.
    """
    from app.ml import routes as ml_predictor
    scenarios = prepare_prediction_features(user_input, ml_predictor)
    return ml_predictor.run_prediction(scenarios)

def example_db_query():
    """
    데이터베이스 연결을 테스트합니다.
    """
    try:
        # 간단한 쿼리를 실행하여 DB 연결 확인
        db.session.execute(db.text("SELECT 1 FROM DUAL"))
        return "DB 연결 성공"
    except Exception as e:
        logger.error(f"DB 테스트 실패: {e}")
        return "DB 연결 실패"

# 중복 함수 제거 - utils 모듈로 이동됨
from app.utils.math_utils import cosine_similarity
from app.utils.flask_utils import get_llm_service as _get_llm_service

def _generate_embeddings_for_rag_data(data_list, model_name):
    """
    RAG 데이터 리스트에 대해 임베딩을 생성하고 캐시합니다.
    """
    llm_service = _get_llm_service()
    if not llm_service:
        logger.error("LLMService not available for embedding generation.")
        return []

    embeddings = []
    for item in data_list:
        # content 키가 있으면 사용, 없으면 description 사용 (LEARNING_RECOMMENDATIONS용)
        text_content = item.get("content") or item.get("description", "")
        embedding = llm_service.generate_embedding(text_content, model_name=model_name)
        if embedding:
            embeddings.append(embedding)
        else:
            embeddings.append([]) # 실패 시 빈 리스트 추가
    return embeddings

def retrieve_labor_market_info(query_text: str, top_n: int = 2) -> str:
    """
    벡터 데이터베이스 기반 노동시장 트렌드 정보 검색 (기존 함수와 호환성 유지)
    """
    from app.rag_manager import get_rag_manager
    
    llm_service = _get_llm_service()
    rag_manager = get_rag_manager()
    
    if rag_manager:
        # 새로운 벡터 데이터베이스 시스템 사용
        try:
            result = rag_manager.get_labor_market_info(query_text, top_n, llm_service)
            if result:
                return result
        except Exception as e:
            logger.error(f"벡터 데이터베이스 검색 실패, 기존 방식으로 대체: {e}")
    
    # 기존 방식으로 폴백
    global _labor_market_embeddings
    if not llm_service:
        return ""

    if _labor_market_embeddings is None:
        _labor_market_embeddings = _generate_embeddings_for_rag_data(LABOR_MARKET_TRENDS, llm_service.ollama_embedding_model)

    if not _labor_market_embeddings or not query_text:
        return ""

    query_embedding = llm_service.generate_embedding(query_text, model_name=llm_service.ollama_embedding_model)
    if not query_embedding:
        return ""

    similarities = []
    for i, doc_embedding in enumerate(_labor_market_embeddings):
        if doc_embedding:
            sim = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((sim, LABOR_MARKET_TRENDS[i]["content"]))

    similarities.sort(key=lambda x: x[0], reverse=True)
    top_results = [content for sim, content in similarities[:top_n] if sim > 0.3]
    return "\n".join(top_results)

def retrieve_learning_recommendations(query_text: str, top_n: int = 2) -> str:
    """
    벡터 데이터베이스 기반 학습 및 역량 강화 추천 검색 (기존 함수와 호환성 유지)
    """
    from app.rag_manager import get_rag_manager
    
    llm_service = _get_llm_service()
    rag_manager = get_rag_manager()
    
    if rag_manager:
        # 새로운 벡터 데이터베이스 시스템 사용
        try:
            result = rag_manager.get_learning_recommendations(query_text, top_n, llm_service)
            if result:
                return result
        except Exception as e:
            logger.error(f"벡터 데이터베이스 검색 실패, 기존 방식으로 대체: {e}")
    
    # 기존 방식으로 폴백
    global _learning_recommendation_embeddings
    if not llm_service:
        return ""

    if _learning_recommendation_embeddings is None:
        _learning_recommendation_embeddings = _generate_embeddings_for_rag_data(LEARNING_RECOMMENDATIONS, llm_service.ollama_embedding_model)

    if not _learning_recommendation_embeddings or not query_text:
        return ""

    query_embedding = llm_service.generate_embedding(query_text, model_name=llm_service.ollama_embedding_model)
    if not query_embedding:
        return ""

    similarities = []
    for i, doc_embedding in enumerate(_learning_recommendation_embeddings):
        if doc_embedding:
            sim = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((sim, LEARNING_RECOMMENDATIONS[i]))

    similarities.sort(key=lambda x: x[0], reverse=True)
    top_results_content = []
    for sim, rec_item in similarities[:top_n]:
        if sim > 0.3:
            resources = ", ".join([f"{r['name']} ({r['type']})" for r in rec_item["learning_resources"]])
            top_results_content.append(f"- {rec_item['skill_name']}: {rec_item['description']} (참고 자료: {resources})")

    return "\n".join(top_results_content)

def translate_or_rephrase_rag_result(text: str) -> str:
    """
    검색된 RAG 결과를 EXAONE 모델을 사용하여 한국어로 요약/재구성합니다.
    """
    if not text.strip():
        return ""

    system_prompt = {"role": "system", "content": "주어진 텍스트를 한국어로 간결하게 요약하거나, 이미 한국어인 경우 더 자연스럽고 핵심적인 내용으로 재구성해주세요. 불필요한 서론 없이 핵심 내용만 전달합니다."}
    user_prompt = {"role": "user", "content": f"다음 텍스트를 요약/재구성해주세요:\n\n{text}"}
    
    try:
        return chat_complete([system_prompt, user_prompt], temperature=0.3, num_ctx=2048) # 요약/재구성이므로 낮은 temperature
    except Exception as e:
        logger.error(f"RAG 결과 번역/재구성 중 오류 발생: {e}")
        return text # 오류 발생 시 원본 텍스트 반환

def generate_career_advice_hf(user_input, prediction_results, job_category_map, satis_factor_map):
    """기존 호환성 유지용 함수 - 새로운 프롬프트 시스템 사용"""
    try:
        return generate_enhanced_career_advice(user_input, prediction_results, job_category_map, satis_factor_map)
    except Exception as e:
        logger.error(f"Enhanced 조언 생성 실패, 기존 방식으로 폴백: {e}")
        return _generate_career_advice_legacy(user_input, prediction_results, job_category_map, satis_factor_map)

def generate_enhanced_career_advice(user_input, prediction_results, job_category_map, satis_factor_map):
    """개선된 프롬프트 엔지니어링을 사용한 커리어 조언 생성"""
    from app.prompt_templates import prompt_manager
    from app.rag_manager import get_rag_manager
    
    # RAG 검색 개선
    current_job_name = job_category_map.get(user_input.get('current_job_category', ''), '알 수 없음')
    job_a_name = job_category_map.get(user_input.get('job_A_category', ''), '알 수 없음') 
    job_b_name = job_category_map.get(user_input.get('job_B_category', ''), '알 수 없음')
    focus_key_name = satis_factor_map.get(user_input.get('satis_focus_key'), '지정되지 않음')
    
    # RAG 검색 쿼리 최적화
    labor_query = f"2024 2025 노동시장 전망 {current_job_name} {job_a_name} {job_b_name} 고용 취업 트렌드"
    learning_query = f"{current_job_name} {focus_key_name} 역량 개발 교육 훈련 추천"
    
    # RAG 매니저에서 검색 결과 가져오기
    rag_manager = get_rag_manager()
    llm_service = _get_llm_service()
    
    labor_market_context = ""
    learning_context = ""
    
    if rag_manager:
        try:
            # 노동시장 정보 검색
            labor_results = rag_manager.search_documents(labor_query, n_results=3, llm_service=llm_service)
            if labor_results:
                labor_market_context = prompt_manager.get_rag_enhanced_context(labor_query, labor_results)
            
            # 학습 추천 정보 검색
            learning_results = rag_manager.search_documents(learning_query, n_results=2, llm_service=llm_service)
            if learning_results:
                learning_context = prompt_manager.get_rag_enhanced_context(learning_query, learning_results)
        
        except Exception as e:
            logger.warning(f"RAG 검색 중 오류, 기본 검색으로 대체: {e}")
            labor_market_context = retrieve_labor_market_info(labor_query)
            learning_context = retrieve_learning_recommendations(learning_query)
    
    # 개선된 프롬프트 시스템으로 메시지 생성
    messages = prompt_manager.get_career_advice_prompt(
        user_input, prediction_results, job_category_map, satis_factor_map,
        labor_market_context, learning_context
    )
    
    # 고품질 응답 생성
    try:
        return chat_complete(messages, temperature=0.7, num_ctx=6144)  # 컨텍스트 확장
    except Exception as e:
        logger.error(f"Enhanced LLM 호출 오류: {e}")
        raise e

def _generate_career_advice_legacy(user_input, prediction_results, job_category_map, satis_factor_map):
    """기존 방식 커리어 조언 생성 (폴백용)"""
    current_job_name = job_category_map.get(user_input.get('current_job_category', ''), '알 수 없음')
    job_a_name = job_category_map.get(user_input.get('job_A_category', ''), '알 수 없음')
    job_b_name = job_category_map.get(user_input.get('job_B_category', ''), '알 수 없음')
    satis_focus_key = user_input.get('satis_focus_key')
    focus_key_name = satis_factor_map.get(satis_focus_key, '지정되지 않음')
    gender_text = '여성' if str(user_input.get('gender')) == '1' else '남성'

    p0 = prediction_results[0]
    p1 = prediction_results[1]
    p2 = prediction_results[2]
    details = (
        f"1) 현직 유지({current_job_name}): 소득 변화율 {p0['income_change_rate']:.2%}, 만족도 변화 {p0['satisfaction_change_score']:.2f}\n"
        f"2) 이직 A({job_a_name}): 소득 변화율 {p1['income_change_rate']:.2%}, 만족도 변화 {p1['satisfaction_change_score']:.2f}\n"
        f"3) 이직 B({job_b_name}): 소득 변화율 {p2['income_change_rate']:.2%}, 만족도 변화 {p2['satisfaction_change_score']:.2f}"
    )

    # RAG: 노동 시장 트렌드 정보 검색 (Llama 3 임베딩 기반)
    labor_market_query = f"{current_job_name} {job_a_name} {job_b_name} 노동 시장 트렌드"
    raw_labor_market_info = retrieve_labor_market_info(labor_market_query)
    # 성능 최적화: RAG 재구성 단계 제거 - 원본 데이터 직접 사용
    if raw_labor_market_info:
        labor_market_context = f"\n\n[관련 노동 시장 트렌드]\n{raw_labor_market_info}"
    else:
        labor_market_context = ""

    # RAG: 학습 및 역량 강화 추천 정보 검색 (Llama 3 임베딩 기반)
    learning_query = f"{current_job_name} {job_a_name} {job_b_name} {focus_key_name} 관련 학습 및 역량 강화"
    raw_learning_recs = retrieve_learning_recommendations(learning_query)
    # 성능 최적화: RAG 재구성 단계 제거 - 원본 데이터 직접 사용
    if raw_learning_recs:
        learning_context = f"\n\n[추천 학습 및 역량 강화]\n{raw_learning_recs}"
    else:
        learning_context = ""

    system = {"role": "system", "content": "당신은 친절하고 전문적인 커리어 코치입니다. 한국어로 구체적이고 실용적인 조언을 제공합니다. 다음 지침을 따르세요:\n1. 예측 결과를 바탕으로 사용자의 상황에 맞는 개인화된 커리어 조언을 제공합니다.\n2. 5가지 이상의 핵심 팁을 1~2문장으로 간결하게 제시합니다.\n3. 가능하면 수치나 근거를 포함하여 신뢰도를 높입니다.\n4. 마지막에 두 문장으로 핵심 내용을 요약하고 결론을 제시합니다.\n5. 동어반복이나 모호한 표현을 피하고, '추가 정보가 없으므로'와 같은 메타 발화는 하지 않습니다.\n6. 응답의 가독성을 위해 불릿 포인트(-)를 사용할 수 있습니다."}
    user = {
        "role": "user",
        "content": (
            f"[사용자]\n"
            f"- 나이: {user_input.get('age','N/A')}세\n"
            f"- 성별: {gender_text}\n"
            f"- 현재 직업군: {current_job_name}\n"
            f"- 현재 월 소득: {user_input.get('monthly_income','N/A')}만원\n"
            f"- 중요 만족도 요인: {focus_key_name}\n\n"
            f"[예측 결과]\n{details}\n"
            f"{labor_market_context}"
            f"{learning_context}"
            f"\n[요청]\n"
            f"각 예측 결과 설명 및 이유: {current_job_name} 의 현재 트렌드에 맞춘 설명 제시\n"
            f"추천 선택지 설명\n"
            f"개인화된 조언\n"
        )
    }
    try:
        # 성능 최적화: 컨텍스트 길이 및 생성 토큰 수 최적화
        return chat_complete([system, user], temperature=0.7, num_ctx=4096)
    except Exception as e:
        logger.error(f"LLM 호출 오류 (generate_career_advice_hf): {e}")
        return "AI 조언을 생성하는 중 오류가 발생했습니다. Ollama가 실행 중인지와 모델 이름을 확인해주세요."

def generate_follow_up_advice(user_message, history, context_summary=""): 
    """개선된 후속 대화 처리 (매번 RAG 검색 수행)"""
    from app.prompt_templates import prompt_manager
    from app.rag_manager import get_rag_manager

    # RAG 검색 수행
    rag_context = ""
    rag_manager = get_rag_manager()
    llm_service = _get_llm_service()
    if rag_manager and llm_service:
        try:
            search_results = rag_manager.search_documents(user_message, n_results=5, llm_service=llm_service)
            if search_results:
                rag_context = prompt_manager.get_rag_enhanced_context(user_message, search_results)
        except Exception as e:
            logger.warning(f"후속 질문 RAG 검색 중 오류: {e}")

    # 개선된 시스템 프롬프트 사용
    messages = [{"role": "system", "content": prompt_manager.get_follow_up_system_prompt()}]
    
    # 컨텍스트 요약 추가
    if context_summary:
        messages.append({"role": "assistant", "content": f"[상담 시작 컨텍스트]\n{context_summary}"})

    if isinstance(history, list):
        for item in history:
            role = "assistant" if item.get("sender") == "ai" else "user"
            messages.append({role: item.get("text", "")})
    
    # RAG 컨텍스트와 사용자 메시지를 함께 전달
    final_user_message = user_message
    if rag_context:
        final_user_message = f"{rag_context}\n\n[사용자 질문]\n{user_message}"
        
    messages.append({"role": "user", "content": final_user_message})
    
    try:
        return chat_complete(messages, temperature=0.6, num_ctx=8192)
    except Exception as e:
        logger.error(f"LLM 호출 오류 (generate_follow_up_advice): {e}")
        raise e

def generate_follow_up_advice_stream(user_message, history, context_summary=""):
    """스트리밍 방식 후속 대화 처리 (매번 RAG 검색 수행)"""
    from app.prompt_templates import prompt_manager
    from app.rag_manager import get_rag_manager

    # RAG 검색 수행
    rag_context = ""
    rag_manager = get_rag_manager()
    llm_service = _get_llm_service()
    if rag_manager and llm_service:
        try:
            search_results = rag_manager.search_documents(user_message, n_results=5, llm_service=llm_service)
            if search_results:
                rag_context = prompt_manager.get_rag_enhanced_context(user_message, search_results)
        except Exception as e:
            logger.warning(f"후속 질문 스트리밍 RAG 검색 중 오류: {e}")

    # 개선된 시스템 프롬프트 사용
    messages = [{"role": "system", "content": prompt_manager.get_follow_up_system_prompt()}]

    # 컨텍스트 요약 추가
    if context_summary:
        messages.append({"role": "assistant", "content": f"[상담 시작 컨텍스트]\n{context_summary}"})
    
    if isinstance(history, list):
        for item in history:
            role = "assistant" if item.get("sender") == "ai" else "user"
            messages.append({role: item.get("text", "")})
    
    # RAG 컨텍스트와 사용자 메시지를 함께 전달
    final_user_message = user_message
    if rag_context:
        final_user_message = f"{rag_context}\n\n[사용자 질문]\n{user_message}"

    messages.append({"role": "user", "content": final_user_message})
    
    try:
        for chunk in chat_stream(messages, temperature=0.6, num_ctx=8192):
            yield chunk
    except Exception as e:
        logger.error(f"스트리밍 LLM 호출 오류: {e}")
        yield f"[오류] 응답 생성 중 문제가 발생했습니다: {str(e)}"

def summarize_context(user_input, prediction_results, job_category_map, satis_factor_map):
    current_job_name = job_category_map.get(user_input.get('current_job_category', ''), '알 수 없음')
    job_a_name = job_category_map.get(user_input.get('job_A_category', ''), '알 수 없음')
    job_b_name = job_category_map.get(user_input.get('job_B_category', ''), '알 수 없음')
    focus_key_name = satis_factor_map.get(user_input.get('satis_focus_key'), '지정되지 않음')
    gender_text = '여성' if str(user_input.get('gender')) == '1' else '남성'
    age = user_input.get('age', 'N/A')
    income = user_input.get('monthly_income', 'N/A')
    p0, p1, p2 = prediction_results
    return (
        f"[요약]\n"
        f"- 나이: {age}세, 성별: {gender_text}\n"
        f"- 현재 직업군: {current_job_name}, 현재 월 소득: {income}만원\n"
        f"- 중요 만족도 요인: {focus_key_name}\n"
        f"- 시나리오1(현직): 소득 {p0['income_change_rate']:.2%}, 만족 {p0['satisfaction_change_score']:.2f}\n"
        f"- 시나리오2(A:{job_a_name}): 소득 {p1['income_change_rate']:.2%}, 만족 {p1['satisfaction_change_score']:.2f}\n"
        f"- 시나리오3(B:{job_b_name}): 소득 {p2['income_change_rate']:.2%}, 만족 {p2['satisfaction_change_score']:.2f}"
    )