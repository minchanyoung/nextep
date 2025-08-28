# config.py
"""
Legacy config for backward compatibility
새로운 설정 시스템으로 점진적 마이그레이션
"""

from app.config import get_settings

# 새로운 설정 시스템 사용
_settings = get_settings()

class Config:
    """Flask 호환성을 위한 레거시 Config 클래스"""
    
    # 새로운 설정 시스템에서 값들을 가져옴
    def __init__(self):
        flask_config = _settings.to_flask_config()
        for key, value in flask_config.items():
            setattr(self, key, value)
    
    # 정적 속성들 (하위 호환성)
    SECRET_KEY = _settings.security.secret_key
    SQLALCHEMY_DATABASE_URI = _settings.database.uri
    SQLALCHEMY_TRACK_MODIFICATIONS = _settings.database.track_modifications
    
    SESSION_COOKIE_SECURE = _settings.security.session_cookie_secure
    SESSION_COOKIE_HTTPONLY = _settings.security.session_cookie_httponly
    SESSION_COOKIE_SAMESITE = _settings.security.session_cookie_samesite
    PERMANENT_SESSION_LIFETIME = _settings.security.session_timeout
    
    LOG_TO_STDOUT = _settings.logging.to_stdout
    LOG_FILE = _settings.logging.file
    LOG_LEVEL = _settings.logging.level
    
    ORACLE_CLIENT_LIB_DIR = _settings.oracle_client_lib_dir
    
    COMPRESS_MIMETYPES = _settings.compression['mimetypes']
    COMPRESS_LEVEL = _settings.compression['level']
    COMPRESS_MIN_SIZE = _settings.compression['min_size']
    
    # Ollama 설정 (하위 호환성)
    OLLAMA_URL = _settings.ollama.url
    OLLAMA_MODEL = _settings.ollama.model
    OLLAMA_EMBEDDING_MODEL = _settings.ollama.embedding_model
    OLLAMA_TIMEOUT = _settings.ollama.timeout
    OLLAMA_KEEP_ALIVE = _settings.ollama.keep_alive
    
    # LangChain 설정 (마이그레이션 후 추가)
    LANGCHAIN_VERBOSE = getattr(_settings, 'langchain_verbose', False)
    LANGCHAIN_CALLBACKS_ENABLED = getattr(_settings, 'langchain_callbacks_enabled', True)
    
    # RAG & 벡터 데이터베이스 설정
    CHROMA_PERSIST_DIR = getattr(_settings, 'chroma_persist_dir', 'instance/chroma_db')
    CHROMA_COLLECTION_NAME = getattr(_settings, 'chroma_collection_name', 'labor_market_docs')
    TEXT_CHUNK_SIZE = getattr(_settings, 'text_chunk_size', 1000)
    TEXT_CHUNK_OVERLAP = getattr(_settings, 'text_chunk_overlap', 200)
    SIMILARITY_SCORE_THRESHOLD = getattr(_settings, 'similarity_score_threshold', 0.7)
    SEARCH_TOP_K = getattr(_settings, 'search_top_k', 5)

