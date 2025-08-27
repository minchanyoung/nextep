# app/core/exceptions.py
"""
커스텀 예외 클래스들
"""


class NextEPError(Exception):
    """NextEP 프로젝트의 기본 예외 클래스"""
    pass


class LLMServiceError(NextEPError):
    """LLM 서비스 관련 예외"""
    pass


class RAGError(NextEPError):
    """RAG 시스템 관련 예외"""
    pass


class MLModelError(NextEPError):
    """머신러닝 모델 관련 예외"""
    pass


class DatabaseError(NextEPError):
    """데이터베이스 관련 예외"""
    pass


class ValidationError(NextEPError):
    """데이터 검증 예외"""
    pass


class ConfigurationError(NextEPError):
    """설정 관련 예외"""
    pass


class CacheError(NextEPError):
    """캐시 시스템 관련 예외"""
    pass