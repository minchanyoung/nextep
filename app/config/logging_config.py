# app/config/logging_config.py
"""
로깅 설정 관리
"""

import os
import logging
import logging.handlers
from typing import Dict, Any


def setup_logging(settings) -> None:
    """로깅 설정 초기화"""
    
    # 로그 디렉토리 생성
    log_dir = os.path.dirname(settings.logging.file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # 로거 설정
    logging.basicConfig(
        level=getattr(logging, settings.logging.level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 루트 로거
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.logging.level))
    
    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # 콘솔 핸들러
    if settings.logging.to_stdout:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # 파일 핸들러 (Rotating)
    if settings.logging.file:
        file_handler = logging.handlers.RotatingFileHandler(
            settings.logging.file,
            maxBytes=settings.logging.max_file_size,
            backupCount=settings.logging.backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 특정 라이브러리 로그 레벨 조정
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    
    # LangChain 관련 로그 레벨 조정
    logging.getLogger('langchain').setLevel(logging.INFO)
    logging.getLogger('langchain_core').setLevel(logging.WARNING)
    logging.getLogger('langchain_community').setLevel(logging.WARNING)
    logging.getLogger('langchain_ollama').setLevel(logging.INFO)
    logging.getLogger('langchain_chroma').setLevel(logging.WARNING)
    
    # Ollama 클라이언트 로그 레벨
    logging.getLogger('ollama').setLevel(logging.WARNING)
    
    # ML 라이브러리 로그 레벨 조정
    logging.getLogger('catboost').setLevel(logging.WARNING)
    logging.getLogger('xgboost').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)
    
    # 데이터베이스 관련
    logging.getLogger('sqlalchemy').setLevel(logging.WARNING)
    logging.getLogger('oracledb').setLevel(logging.WARNING)
    
    logging.info(f"로깅 시스템이 초기화되었습니다. 레벨: {settings.logging.level}")


def setup_component_loggers() -> Dict[str, logging.Logger]:
    """컴포넌트별 전용 로거 설정"""
    loggers = {}
    
    # 컴포넌트별 로거 생성
    components = [
        'app.llm_service',
        'app.rag_manager', 
        'app.services',
        'app.ml.routes',
        'app.main.routes',
        'app.auth.routes'
    ]
    
    for component in components:
        logger = logging.getLogger(component)
        loggers[component.split('.')[-1]] = logger
        
        # 개별 파일 핸들러 (선택적)
        # if settings.logging.separate_files:
        #     file_handler = logging.FileHandler(f'logs/{component}.log')
        #     file_handler.setFormatter(formatter)
        #     logger.addHandler(file_handler)
    
    return loggers


def get_logger(name: str) -> logging.Logger:
    """네임스페이스별 로거 반환"""
    return logging.getLogger(name)


def log_system_info():
    """시스템 정보 로깅"""
    import sys
    import platform
    
    logger = get_logger('system')
    logger.info("=" * 50)
    logger.info("NEXTEP 시스템 시작")
    logger.info("=" * 50)
    logger.info(f"Python 버전: {sys.version}")
    logger.info(f"플랫폼: {platform.platform()}")
    logger.info(f"아키텍처: {platform.architecture()}")
    logger.info("=" * 50)


def log_langchain_info():
    """LangChain 관련 정보 로깅"""
    logger = get_logger('langchain')
    
    try:
        import langchain
        import langchain_core
        import langchain_ollama
        
        logger.info("LangChain 라이브러리 정보:")
        logger.info(f"  - langchain: {langchain.__version__}")
        logger.info(f"  - langchain-core: {langchain_core.__version__}")
        logger.info(f"  - langchain-ollama: {langchain_ollama.__version__}")
        
    except ImportError as e:
        logger.warning(f"LangChain 라이브러리 정보 수집 실패: {e}")


def log_model_loading_start(model_name: str, model_type: str = "LLM"):
    """모델 로딩 시작 로깅"""
    logger = get_logger('model_loader')
    logger.info(f"{model_type} 모델 로딩 시작: {model_name}")


def log_model_loading_end(model_name: str, duration: float, model_type: str = "LLM"):
    """모델 로딩 완료 로깅"""
    logger = get_logger('model_loader')
    logger.info(f"{model_type} 모델 로딩 완료: {model_name} ({duration:.2f}초)")