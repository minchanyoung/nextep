# app/utils/error_handler.py
"""
통합 에러 핸들링 시스템
"""

import logging
import traceback
import functools
from typing import Any, Callable, Optional, Dict
from flask import jsonify, request, current_app
from app.core.exceptions import NextEPError, LLMServiceError, RAGError, MLModelError

logger = logging.getLogger(__name__)


class ErrorHandler:
    """중앙화된 에러 핸들링"""
    
    def __init__(self):
        self.error_counts = {}
        self.error_patterns = {}
    
    def handle_error(self, error: Exception, context: Optional[Dict] = None) -> Dict[str, Any]:
        """에러를 처리하고 적절한 응답 생성"""
        error_type = type(error).__name__
        error_msg = str(error)
        
        # 에러 통계 업데이트
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # 컨텍스트 정보 추가
        context = context or {}
        context.update({
            'error_type': error_type,
            'request_path': getattr(request, 'path', 'N/A'),
            'user_agent': getattr(request, 'user_agent', 'N/A').string if hasattr(request, 'user_agent') else 'N/A'
        })
        
        # 에러 레벨에 따른 로깅
        if isinstance(error, NextEPError):
            logger.warning(f"Business error: {error_msg}", extra=context)
            return self._create_error_response(error_msg, 400, error_type)
        elif isinstance(error, (LLMServiceError, RAGError, MLModelError)):
            logger.error(f"Service error: {error_msg}", extra=context)
            return self._create_error_response("서비스 처리 중 오류가 발생했습니다.", 500, error_type)
        else:
            # 예상치 못한 에러
            logger.error(f"Unexpected error: {error_msg}\n{traceback.format_exc()}", extra=context)
            return self._create_error_response("예기치 못한 오류가 발생했습니다.", 500, error_type)
    
    def _create_error_response(self, message: str, status_code: int, error_type: str) -> Dict[str, Any]:
        """에러 응답 생성"""
        return {
            'error': True,
            'message': message,
            'error_type': error_type,
            'status_code': status_code,
            'timestamp': self._get_current_timestamp()
        }
    
    def _get_current_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def get_error_stats(self) -> Dict[str, int]:
        """에러 통계 반환"""
        return self.error_counts.copy()


# 글로벌 에러 핸들러 인스턴스
_error_handler = ErrorHandler()


def handle_exceptions(func: Callable) -> Callable:
    """함수에 대한 예외 처리 데코레이터"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_response = _error_handler.handle_error(e, {
                'function': func.__name__,
                'module': func.__module__
            })
            
            # Flask 앱 컨텍스트에서는 JSON 응답 반환
            try:
                return jsonify(error_response), error_response['status_code']
            except:
                # 컨텍스트 밖에서는 딕셔너리 반환
                return error_response
    
    return wrapper


def handle_service_exceptions(service_name: str):
    """서비스별 예외 처리 데코레이터"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_response = _error_handler.handle_error(e, {
                    'service': service_name,
                    'function': func.__name__
                })
                
                # 서비스 레이어에서는 예외를 다시 발생시킴
                raise NextEPError(f"{service_name} 서비스 오류: {error_response['message']}")
        
        return wrapper
    return decorator


def safe_execute(func: Callable, default_return=None, context: Optional[Dict] = None):
    """안전한 함수 실행"""
    try:
        return func()
    except Exception as e:
        _error_handler.handle_error(e, context)
        return default_return


def get_error_handler() -> ErrorHandler:
    """에러 핸들러 인스턴스 반환"""
    return _error_handler