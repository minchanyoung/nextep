# app/utils/flask_utils.py
"""
Flask 관련 공통 유틸리티
"""

from typing import Optional
from flask import current_app
import logging

logger = logging.getLogger(__name__)


def get_llm_service():
    """
    현재 애플리케이션 컨텍스트에서 LLMService 인스턴스를 가져옵니다.
    
    Returns:
        LLMService 인스턴스 또는 None
    """
    if current_app and 'llm_service' in current_app.extensions:
        return current_app.extensions['llm_service']
    return None


def get_rag_manager():
    """
    현재 애플리케이션 컨텍스트에서 RAGManager 인스턴스를 가져옵니다.
    
    Returns:
        RAGManager 인스턴스 또는 None
    """
    if current_app and 'rag_manager' in current_app.extensions:
        return current_app.extensions['rag_manager']
    return None


def get_ml_service():
    """
    현재 애플리케이션 컨텍스트에서 ML 서비스를 가져옵니다.
    
    Returns:
        ML 서비스 모듈 또는 None
    """
    try:
        from app.ml import routes as ml_service
        return ml_service
    except ImportError:
        logger.error("ML 서비스 모듈을 찾을 수 없습니다.")
        return None


def is_app_context_available() -> bool:
    """
    Flask 앱 컨텍스트가 사용 가능한지 확인합니다.
    
    Returns:
        bool: 앱 컨텍스트 사용 가능 여부
    """
    try:
        return current_app is not None
    except RuntimeError:
        return False


def get_app_config(key: str, default=None):
    """
    앱 설정값을 안전하게 가져옵니다.
    
    Args:
        key: 설정 키
        default: 기본값
    
    Returns:
        설정값 또는 기본값
    """
    if is_app_context_available():
        return current_app.config.get(key, default)
    return default


def log_app_info(message: str, level: str = "info"):
    """
    앱 컨텍스트에서 로그를 기록합니다.
    
    Args:
        message: 로그 메시지
        level: 로그 레벨 (debug, info, warning, error)
    """
    if is_app_context_available():
        app_logger = current_app.logger
        getattr(app_logger, level, app_logger.info)(message)
    else:
        getattr(logger, level, logger.info)(message)