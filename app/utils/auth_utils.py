# app/utils/auth_utils.py
"""
인증 관련 공통 유틸리티
"""

import logging
from functools import wraps
from flask import redirect, url_for, flash
from app.utils.session_utils import is_user_logged_in, get_current_username
from app.services import get_user_by_username

logger = logging.getLogger(__name__)


def login_required(f):
    """로그인이 필요한 뷰 함수를 위한 데코레이터"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_user_logged_in():
            flash('로그인이 필요합니다.')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function


def get_current_user():
    """현재 로그인된 사용자 객체 반환"""
    username = get_current_username()
    if username:
        return get_user_by_username(username)
    return None


def require_profile_complete(required_fields=None):
    """프로필 완성이 필요한 뷰 함수를 위한 데코레이터"""
    if required_fields is None:
        required_fields = ['age', 'gender', 'education', 'monthly_income', 'job_category', 'job_satisfaction', 'satis_focus_key']
    
    def decorator(f):
        @wraps(f)
        @login_required
        def decorated_function(*args, **kwargs):
            user = get_current_user()
            if not user:
                flash('사용자 정보를 찾을 수 없습니다.')
                return redirect(url_for('auth.login'))
            
            # 필수 프로필 필드 확인
            missing_fields = []
            for field in required_fields:
                if getattr(user, field, None) is None:
                    missing_fields.append(field)
            
            if missing_fields:
                flash(f'예측을 위해 프로필 정보를 모두 입력해주세요. 누락: {", ".join(missing_fields)}')
                return redirect(url_for('main.profile'))
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def validate_user_input(data: dict, required_fields: list) -> tuple[bool, list]:
    """사용자 입력 검증"""
    missing_fields = []
    for field in required_fields:
        if field not in data or data[field] is None or data[field] == '':
            missing_fields.append(field)
    
    return len(missing_fields) == 0, missing_fields