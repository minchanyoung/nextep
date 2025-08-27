# app/utils/response_utils.py
"""
응답 처리 공통 유틸리티
"""

import json
import logging
from flask import jsonify, current_app
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def json_response(data: Dict[str, Any], status_code: int = 200) -> tuple:
    """JSON 응답 생성"""
    return json.dumps(data, ensure_ascii=False), status_code


def success_response(message: str = "성공", data: Optional[Dict] = None) -> tuple:
    """성공 응답 생성"""
    response_data = {'status': 'success', 'message': message}
    if data:
        response_data['data'] = data
    return json_response(response_data)


def error_response(message: str = "오류 발생", status_code: int = 500, error_code: Optional[str] = None) -> tuple:
    """오류 응답 생성"""
    response_data = {'status': 'error', 'message': message}
    if error_code:
        response_data['error_code'] = error_code
    return json_response(response_data, status_code)


def validation_error_response(errors: list, message: str = "입력 검증 실패") -> tuple:
    """입력 검증 오류 응답"""
    return json_response({
        'status': 'validation_error',
        'message': message,
        'errors': errors
    }, 400)


def handle_api_exception(func):
    """API 예외 처리 데코레이터"""
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"입력 검증 오류 in {func.__name__}: {e}")
            return error_response("입력값이 올바르지 않습니다.", 400)
        except Exception as e:
            logger.error(f"API 오류 in {func.__name__}: {e}")
            return error_response("서버 오류가 발생했습니다.", 500)
    return wrapper


def paginated_response(items: list, page: int, per_page: int, total: int, **kwargs) -> Dict:
    """페이지네이션 응답 생성"""
    return {
        'items': items,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': total,
            'pages': (total + per_page - 1) // per_page,
            'has_next': page * per_page < total,
            'has_prev': page > 1
        },
        **kwargs
    }


class APIResponse:
    """API 응답 빌더 클래스"""
    
    def __init__(self):
        self.data = {}
        self.status_code = 200
    
    def success(self, message: str = "성공", data: Optional[Dict] = None):
        self.data = {'status': 'success', 'message': message}
        if data:
            self.data['data'] = data
        self.status_code = 200
        return self
    
    def error(self, message: str = "오류 발생", status_code: int = 500):
        self.data = {'status': 'error', 'message': message}
        self.status_code = status_code
        return self
    
    def with_data(self, **kwargs):
        self.data.update(kwargs)
        return self
    
    def build(self) -> tuple:
        return json_response(self.data, self.status_code)