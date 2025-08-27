# app/utils/db_utils.py
"""
데이터베이스 공통 유틸리티
"""

import logging
from typing import Callable, Any, Optional
from app import db
from functools import wraps

logger = logging.getLogger(__name__)


def safe_db_operation(operation_name: str = "Database operation"):
    """데이터베이스 작업을 안전하게 실행하는 데코레이터"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                result = func(*args, **kwargs)
                db.session.commit()
                return result
            except Exception as e:
                db.session.rollback()
                logger.error(f"{operation_name} 중 오류 발생: {e}")
                return None
        return wrapper
    return decorator


def execute_db_transaction(func: Callable, *args, **kwargs) -> tuple[bool, Any]:
    """
    데이터베이스 트랜잭션 실행
    Returns: (성공여부, 결과)
    """
    try:
        result = func(*args, **kwargs)
        db.session.commit()
        return True, result
    except Exception as e:
        db.session.rollback()
        logger.error(f"데이터베이스 트랜잭션 실패: {e}")
        return False, str(e)


def safe_query(query_func: Callable, default: Any = None, log_error: bool = True) -> Any:
    """안전한 데이터베이스 쿼리 실행"""
    try:
        return query_func()
    except Exception as e:
        if log_error:
            logger.error(f"데이터베이스 쿼리 오류: {e}")
        return default


def get_or_create(model_class, **kwargs):
    """객체 가져오기 또는 생성"""
    try:
        instance = model_class.query.filter_by(**kwargs).first()
        if instance:
            return instance, False
        else:
            instance = model_class(**kwargs)
            db.session.add(instance)
            db.session.commit()
            return instance, True
    except Exception as e:
        db.session.rollback()
        logger.error(f"get_or_create 오류 ({model_class.__name__}): {e}")
        return None, False


def bulk_update(model_class, updates: list) -> bool:
    """대량 업데이트 실행"""
    try:
        for update_data in updates:
            obj_id = update_data.pop('id')
            model_class.query.filter_by(id=obj_id).update(update_data)
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        logger.error(f"대량 업데이트 실패 ({model_class.__name__}): {e}")
        return False


def delete_by_condition(model_class, **conditions) -> int:
    """조건에 따른 삭제"""
    try:
        deleted_count = model_class.query.filter_by(**conditions).delete()
        db.session.commit()
        return deleted_count
    except Exception as e:
        db.session.rollback()
        logger.error(f"조건부 삭제 실패 ({model_class.__name__}): {e}")
        return 0