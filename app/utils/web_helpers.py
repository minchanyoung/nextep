# app/utils/web_helpers.py
"""
웹 애플리케이션의 인증, 세션, 응답 처리를 위한 헬퍼 함수 모음
"""

import logging
import uuid
from functools import wraps
from typing import Any, Dict, Optional

from flask import flash, jsonify, redirect, session, url_for, request

# 이 파일이 다른 유틸리티에 의해 순환 참조될 수 있으므로, 
# 뷰 함수나 모델 임포트는 함수 내에서 지역적으로 수행합니다.

logger = logging.getLogger(__name__)

# === 인증 관련 (Auth) ===

def login_required(f):
    """로그인이 필요한 뷰 함수를 위한 데코레이터"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_user_logged_in():
            # AJAX 요청인지 확인
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'error': '로그인이 필요합니다.'}), 401
            flash('로그인이 필요합니다.')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function

def get_current_user():
    """현재 로그인된 사용자 객체 반환"""
    from app.services import get_user_by_username
    username = get_current_username()
    if username:
        return get_user_by_username(username)
    return None

# === 세션 관련 (Session) ===

def get_session_value(key: str, default: Any = None) -> Any:
    """세션에서 안전하게 값 가져오기"""
    return session.get(key, default)

def set_session_value(key: str, value: Any) -> None:
    """세션에 값 설정"""
    session[key] = value

def get_current_username() -> Optional[str]:
    """현재 로그인된 사용자 이름 반환"""
    return session.get('user')

def is_user_logged_in() -> bool:
    """사용자 로그인 상태 확인"""
    return 'user' in session and session['user'] is not None

def set_user_session(username: str) -> None:
    """사용자 세션 설정"""
    session['user'] = username

def clear_user_session() -> None:
    """사용자 세션 초기화"""
    session.pop('user', None)
    session.pop('prediction_data', None)
    session.pop('chat_messages', None)

def get_prediction_data() -> Optional[Dict]:
    """예측 데이터 가져오기"""
    return session.get('prediction_data')

def set_prediction_data(user_input: Dict, prediction_results: list) -> None:
    """예측 데이터 설정"""
    session['prediction_data'] = {
        'user_input': user_input,
        'prediction_results': prediction_results
    }

def get_chat_messages() -> list:
    """채팅 메시지 가져오기"""
    return session.get('chat_messages', [])

def set_chat_messages(messages: list) -> None:
    """채팅 메시지 설정"""
    session['chat_messages'] = messages

def add_chat_message(role: str, content: str) -> None:
    """채팅 메시지 추가"""
    messages = get_chat_messages()
    messages.append({"role": role, "content": content})
    set_chat_messages(messages)

# === 응답 관련 (Response) ===

def json_response(data: Dict[str, Any], status_code: int = 200) -> tuple:
    """JSON 응답 생성"""
    return jsonify(data), status_code

def success_response(message: str = "성공", data: Optional[Dict] = None, **kwargs) -> tuple:
    """성공 응답 생성"""
    response_data = {'status': 'success', 'message': message}
    if data:
        response_data['data'] = data
    response_data.update(kwargs)
    return json_response(response_data)

def error_response(message: str = "오류 발생", status_code: int = 500, **kwargs) -> tuple:
    """오류 응답 생성"""
    response_data = {'status': 'error', 'message': message}
    response_data.update(kwargs)
    return json_response(response_data, status_code)

def handle_api_exception(func):
    """API 예외 처리 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"입력 검증 오류 in {func.__name__}: {e}")
            return error_response("입력값이 올바르지 않습니다.", 400)
        except Exception as e:
            logger.error(f"API 오류 in {func.__name__}: {e}", exc_info=True)
            return error_response("서버 오류가 발생했습니다.", 500)
    return wrapper
