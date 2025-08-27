# app/config/settings.py
"""
설정 관리 시스템 - 환경별 설정을 체계적으로 관리
"""

import os
import sys
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class DatabaseSettings:
    """데이터베이스 설정"""
    uri: str
    track_modifications: bool = False
    
    @property
    def is_oracle(self) -> bool:
        return 'oracle' in self.uri.lower()


@dataclass
class OllamaSettings:
    """Ollama LLM 설정"""
    url: str
    model: str
    embedding_model: str
    timeout: Optional[int]
    keep_alive: str
    
    # 성능 옵션 (기본값은 _load_ollama_settings에서 관리)
    num_ctx: int
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float
    num_predict: int
    num_thread: int
    num_gpu: int
    num_batch: int
    
    def get_default_options(self) -> dict:
        """Ollama API 요청에 사용할 기본 옵션 딕셔너리 반환"""
        return {
            "num_ctx": self.num_ctx,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "num_predict": self.num_predict,
            "num_thread": self.num_thread,
            "num_gpu": self.num_gpu,
            "num_batch": self.num_batch
        }


@dataclass
class SecuritySettings:
    """보안 설정"""
    secret_key: str
    session_cookie_secure: bool = False
    session_cookie_httponly: bool = True
    session_cookie_samesite: str = "Lax"
    session_timeout: int = 3600
    
    def __post_init__(self):
        if not self.secret_key:
            raise ValueError("SECRET_KEY is required")


@dataclass
class LoggingSettings:
    """로깅 설정"""
    to_stdout: bool = True
    file: str = "logs/app.log"
    level: str = "INFO"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class CacheSettings:
    """캐싱 설정"""
    embedding_cache_size: int = 500
    search_cache_size: int = 200
    embedding_ttl: int = 3600  # 1시간
    search_ttl: int = 1800     # 30분


class Settings:
    """메인 설정 클래스"""
    
    def __init__(self):
        self.database = self._load_database_settings()
        self.ollama = self._load_ollama_settings() 
        self.security = self._load_security_settings()
        self.logging = self._load_logging_settings()
        self.cache = self._load_cache_settings()
        
        # Flask 설정
        self.debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
        self.testing = os.environ.get('FLASK_TESTING', 'False').lower() == 'true'
        
        # Oracle Client
        self.oracle_client_lib_dir = os.environ.get('ORACLE_CLIENT_LIB_DIR')
        
        # 압축 설정
        self.compression = {
            'mimetypes': [
                'text/html', 'text/css', 'text/xml', 'application/json',
                'application/javascript', 'text/javascript', 'application/xml'
            ],
            'level': 6,
            'min_size': 500
        }
        
        self._validate_settings()
    
    def _load_database_settings(self) -> DatabaseSettings:
        """데이터베이스 설정 로드"""
        uri = os.environ.get(
            "DATABASE_URI",
            "oracle+oracledb://MIN:min@localhost:1521/XE"
        )
        return DatabaseSettings(uri=uri)
    
    def _load_ollama_settings(self) -> OllamaSettings:
        """Ollama 설정 로드"""
        return OllamaSettings(
            url=os.environ.get('OLLAMA_URL', 'http://localhost:11434'),
            model=os.environ.get('OLLAMA_MODEL', 'exaone3.5:7.8b'),
            embedding_model=os.environ.get('OLLAMA_EMBEDDING_MODEL', 'llama3'),
            timeout=self._get_optional_int('OLLAMA_TIMEOUT'),
            keep_alive=os.environ.get('OLLAMA_KEEP_ALIVE', '30m'),
            num_ctx=int(os.environ.get('OLLAMA_NUM_CTX', '2048')),
            temperature=float(os.environ.get('OLLAMA_TEMPERATURE', '0.7')),
            top_p=float(os.environ.get('OLLAMA_TOP_P', '0.9')),
            top_k=int(os.environ.get('OLLAMA_TOP_K', '40')),
            repeat_penalty=float(os.environ.get('OLLAMA_REPEAT_PENALTY', '1.1')),
            num_predict=int(os.environ.get('OLLAMA_NUM_PREDICT', '1024')),
            num_thread=int(os.environ.get('OLLAMA_NUM_THREAD', '-1')),
            num_gpu=int(os.environ.get('OLLAMA_NUM_GPU', '0')),
            num_batch=int(os.environ.get('OLLAMA_NUM_BATCH', '512'))
        )
    
    def _load_security_settings(self) -> SecuritySettings:
        """보안 설정 로드"""
        return SecuritySettings(
            secret_key=os.environ.get('SECRET_KEY', ''),
            session_cookie_secure=os.environ.get('SESSION_COOKIE_SECURE', 'False').lower() == 'true',
            session_timeout=int(os.environ.get('SESSION_TIMEOUT', '3600'))
        )
    
    def _load_logging_settings(self) -> LoggingSettings:
        """로깅 설정 로드"""
        return LoggingSettings(
            to_stdout=os.environ.get('LOG_TO_STDOUT', 'True').lower() == 'true',
            file=os.environ.get('LOG_FILE', 'logs/app.log'),
            level=os.environ.get('LOG_LEVEL', 'INFO').upper(),
            max_file_size=int(os.environ.get('LOG_MAX_FILE_SIZE', '10485760')),
            backup_count=int(os.environ.get('LOG_BACKUP_COUNT', '5'))
        )
    
    def _load_cache_settings(self) -> CacheSettings:
        """캐시 설정 로드"""
        return CacheSettings(
            embedding_cache_size=int(os.environ.get('CACHE_EMBEDDING_SIZE', '500')),
            search_cache_size=int(os.environ.get('CACHE_SEARCH_SIZE', '200')),
            embedding_ttl=int(os.environ.get('CACHE_EMBEDDING_TTL', '3600')),
            search_ttl=int(os.environ.get('CACHE_SEARCH_TTL', '1800'))
        )
    
    def _get_optional_int(self, key: str) -> Optional[int]:
        """환경변수에서 선택적 정수값 가져오기"""
        value = os.environ.get(key)
        if value is None or value == "" or value.lower() == "none":
            return None
        try:
            return int(value)
        except ValueError:
            return None
    
    def _validate_settings(self):
        """설정 검증"""
        if not self.security.secret_key:
            print("=" * 80, file=sys.stderr)
            print("FATAL: SECRET_KEY is not configured in the .env file or environment variables.", file=sys.stderr)
            print("Please set a strong, unique SECRET_KEY.", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            sys.exit(1)
    
    def to_flask_config(self) -> Dict[str, Any]:
        """Flask 앱에 사용할 설정 딕셔너리 반환"""
        return {
            # Flask 기본 설정
            'SECRET_KEY': self.security.secret_key,
            'DEBUG': self.debug,
            'TESTING': self.testing,
            
            # 데이터베이스
            'SQLALCHEMY_DATABASE_URI': self.database.uri,
            'SQLALCHEMY_TRACK_MODIFICATIONS': self.database.track_modifications,
            
            # 세션 보안
            'SESSION_COOKIE_SECURE': self.security.session_cookie_secure,
            'SESSION_COOKIE_HTTPONLY': self.security.session_cookie_httponly,
            'SESSION_COOKIE_SAMESITE': self.security.session_cookie_samesite,
            'PERMANENT_SESSION_LIFETIME': self.security.session_timeout,
            
            # 압축
            'COMPRESS_MIMETYPES': self.compression['mimetypes'],
            'COMPRESS_LEVEL': self.compression['level'],
            'COMPRESS_MIN_SIZE': self.compression['min_size'],
            
            # Ollama LLM 설정
            'OLLAMA_URL': self.ollama.url,
            'OLLAMA_MODEL': self.ollama.model,
            'OLLAMA_EMBEDDING_MODEL': self.ollama.embedding_model,
            'OLLAMA_TIMEOUT': self.ollama.timeout,
            'OLLAMA_KEEP_ALIVE': self.ollama.keep_alive,
        }


# 싱글톤 인스턴스
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """설정 인스턴스 반환 (싱글톤)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings