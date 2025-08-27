# app/utils/session_utils.py
"""
세션 관리 공통 유틸리티
"""

from flask import session
from typing import Any, Optional, Dict
import uuid
import logging

logger = logging.getLogger(__name__)


def get_session_value(key: str, default: Any = None) -> Any:
    """세션에서 안전하게 값 가져오기"""
    return session.get(key, default)


def set_session_value(key: str, value: Any) -> None:
    """세션에 값 설정"""
    session[key] = value


def get_or_create_session_id() -> str:
    """세션 ID 가져오기 또는 생성"""
    session_id = session.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
    return session_id


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


def get_prediction_data() -> Optional[Dict]:
    """예측 데이터 가져오기"""
    return session.get('prediction_data')


def set_prediction_data(user_input: Dict, prediction_results: list) -> None:
    """예측 데이터 설정"""
    session['prediction_data'] = {
        'user_input': user_input,
        'prediction_results': prediction_results
    }


def clear_prediction_data() -> None:
    """예측 데이터 초기화"""
    session.pop('prediction_data', None)


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


def clear_session_data(*keys: str) -> None:
    """지정된 세션 키들 삭제"""
    for key in keys:
        session.pop(key, None)