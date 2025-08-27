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
    
    logging.info(f"로깅 시스템이 초기화되었습니다. 레벨: {settings.logging.level}")


def get_logger(name: str) -> logging.Logger:
    """네임스페이스별 로거 반환"""
    return logging.getLogger(name)