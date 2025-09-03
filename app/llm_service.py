"""
LangChain 기반 LLM 서비스 (완전 마이그레이션 버전)
기존 Ollama 직접 호출을 LangChain으로 완전 교체
"""

import os
import logging
from typing import List, Dict, Any, Iterator, Optional
from flask import current_app

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.callbacks.base import BaseCallbackHandler

from app.core.exceptions import LLMServiceError
from app.utils.error_handler import handle_service_exceptions

logger = logging.getLogger(__name__)


class StreamingCallbackHandler(BaseCallbackHandler):
    """스트리밍을 위한 콜백 핸들러"""
    
    def __init__(self):
        self.tokens = []
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """새 토큰이 생성될 때 호출"""
        self.tokens.append(token)
        
    def get_tokens(self) -> List[str]:
        """생성된 토큰들 반환"""
        return self.tokens.copy()
        
    def clear_tokens(self):
        """토큰 목록 초기화"""
        self.tokens.clear()


class LLMService:
    """LangChain 기반 완전 통합 LLM 서비스"""
    
    def __init__(self, app=None):
        # 기본값 설정
        self.ollama_url = "http://localhost:11434"
        self.ollama_model = "exaone3.5:7.8b"
        self.ollama_embedding_model = "llama3"
        self.ollama_timeout = None
        self.default_options = {}
        
        # LangChain 컴포넌트
        self.chat_model = None
        self.embedding_model = None
        self.output_parser = StrOutputParser()
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Flask 앱 초기화"""
        try:
            # 설정 로드
            from app.config.settings import get_settings
            settings = get_settings()
            
            self.ollama_url = settings.ollama.url
            self.ollama_model = settings.ollama.model
            self.ollama_embedding_model = settings.ollama.embedding_model
            self.ollama_timeout = settings.ollama.timeout
            self.default_options = settings.ollama.get_default_options()
            
            # LangChain 모델 초기화
            self._initialize_models()
            
            app.extensions["llm_service"] = self
            logger.info("LangChain 기반 LLM 서비스가 초기화되었습니다.")
            
        except Exception as e:
            logger.error(f"LLM 서비스 초기화 실패: {e}")
            raise e
    
    def _initialize_models(self):
        """LangChain 모델들 초기화"""
        try:
            # 채팅 모델 초기화 (새로운 langchain-ollama 사용)
            chat_params = {
                "base_url": self.ollama_url,
                "model": self.ollama_model,
                "temperature": self.default_options.get('temperature', 0.6),
                "num_ctx": self.default_options.get('num_ctx', 8192),
                "num_predict": self.default_options.get('num_predict', -1),
                "keep_alive": "30m"
            }
            
            # timeout이 설정되어 있을 때만 추가
            if self.ollama_timeout:
                chat_params["timeout"] = self.ollama_timeout
                
            self.chat_model = ChatOllama(**chat_params)
            
            # 임베딩 모델 초기화 (새로운 langchain-ollama 사용)
            embed_params = {
                "base_url": self.ollama_url,
                "model": self.ollama_embedding_model
            }
            
            self.embedding_model = OllamaEmbeddings(**embed_params)
            
            logger.info("LangChain 모델 초기화 완료")
            
        except Exception as e:
            logger.error(f"LangChain 모델 초기화 실패: {e}")
            raise LLMServiceError(f"모델 초기화 실패: {str(e)}")
    
    def _messages_to_langchain(self, messages: List[Dict[str, str]]) -> List:
        """기존 메시지 형식을 LangChain 메시지로 변환"""
        langchain_messages = []
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:
                # 기본적으로 user로 처리
                langchain_messages.append(HumanMessage(content=content))
        
        return langchain_messages
    
    @handle_service_exceptions("LLM")
    def chat_sync(self, messages: List[Dict[str, str]], options: Optional[Dict] = None) -> str:
        """동기 채팅 완성 (기존 API 호환)"""
        try:
            # 메시지 변환
            langchain_messages = self._messages_to_langchain(messages)
            
            # 옵션 적용
            if options:
                # 동적으로 모델 옵션 업데이트
                self.chat_model.temperature = options.get('temperature', self.chat_model.temperature)
                self.chat_model.num_ctx = options.get('num_ctx', self.chat_model.num_ctx)
                if 'num_predict' in options:
                    self.chat_model.num_predict = options['num_predict']
            
            # LangChain으로 응답 생성
            response = self.chat_model.invoke(langchain_messages)
            
            # 응답 파싱
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"LangChain 채팅 완성 오류: {e}")
            if "connection" in str(e).lower():
                raise LLMServiceError("Ollama 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.")
            elif "timeout" in str(e).lower():
                raise LLMServiceError(f"요청 시간 초과 (설정: {self.ollama_timeout}초)")
            elif "404" in str(e).lower():
                raise LLMServiceError(f"모델 '{self.ollama_model}'을 찾을 수 없습니다.")
            else:
                raise LLMServiceError(f"LLM 처리 중 오류: {str(e)}")
    
    def chat_stream(self, messages: List[Dict[str, str]], options: Optional[Dict] = None) -> Iterator[str]:
        """스트리밍 채팅 완성 (기존 API 호환)"""
        try:
            # 메시지 변환
            langchain_messages = self._messages_to_langchain(messages)
            
            # 옵션 적용
            if options:
                self.chat_model.temperature = options.get('temperature', self.chat_model.temperature)
                self.chat_model.num_ctx = options.get('num_ctx', self.chat_model.num_ctx)
                if 'num_predict' in options:
                    self.chat_model.num_predict = options['num_predict']
            
            # 스트리밍 콜백 핸들러
            streaming_handler = StreamingCallbackHandler()
            
            # 스트리밍으로 응답 생성
            for chunk in self.chat_model.stream(langchain_messages, callbacks=[streaming_handler]):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
                elif isinstance(chunk, str):
                    yield chunk
                    
        except Exception as e:
            logger.error(f"LangChain 스트리밍 오류: {e}")
            yield "AI 응답 생성 중 스트리밍 오류가 발생했습니다."
    
    def generate_embedding(self, text: str, model_name: Optional[str] = None) -> List[float]:
        """텍스트 임베딩 생성 (기존 API 호환)"""
        try:
            # 캐시 확인
            from app.utils.cache_manager import get_cache_manager
            cache_manager = get_cache_manager()
            
            model_to_use = model_name if model_name else self.ollama_embedding_model
            
            cached_embedding = cache_manager.get_embedding(text, model_to_use)
            if cached_embedding is not None:
                return cached_embedding
            
            # LangChain으로 임베딩 생성
            if model_name and model_name != self.ollama_embedding_model:
                # 다른 모델 사용시 임시 임베딩 모델 생성
                temp_params = {
                    "base_url": self.ollama_url,
                    "model": model_name
                }
                if self.ollama_timeout:
                    temp_params["timeout"] = self.ollama_timeout
                    
                temp_embedding_model = OllamaEmbeddings(**temp_params)
                embedding = temp_embedding_model.embed_query(text)
            else:
                embedding = self.embedding_model.embed_query(text)
            
            # 캐시에 저장
            cache_manager.cache_embedding(text, model_to_use, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"LangChain 임베딩 생성 오류: {e}")
            return []
    
    def embed_documents(self, texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
        """여러 문서 임베딩 생성 (배치 처리)"""
        try:
            model_to_use = model_name if model_name else self.ollama_embedding_model
            
            if model_name and model_name != self.ollama_embedding_model:
                temp_params = {
                    "base_url": self.ollama_url,
                    "model": model_name
                }
                if self.ollama_timeout:
                    temp_params["timeout"] = self.ollama_timeout
                    
                temp_embedding_model = OllamaEmbeddings(**temp_params)
                embeddings = temp_embedding_model.embed_documents(texts)
            else:
                embeddings = self.embedding_model.embed_documents(texts)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"LangChain 배치 임베딩 생성 오류: {e}")
            return [[] for _ in texts]
    
    def create_chain(self, prompt_template: str, **kwargs):
        """LangChain 체인 생성"""
        try:
            # 프롬프트 템플릿 생성
            prompt = ChatPromptTemplate.from_template(prompt_template)
            
            # 체인 구성
            chain = prompt | self.chat_model | self.output_parser
            
            return chain
            
        except Exception as e:
            logger.error(f"LangChain 체인 생성 오류: {e}")
            raise LLMServiceError(f"체인 생성 실패: {str(e)}")
    
    def create_conversational_chain(self):
        """대화형 체인 생성"""
        try:
            from app.prompt_templates import prompt_manager
            system_prompt = prompt_manager.get_conversational_system_prompt()
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            
            chain = prompt | self.chat_model | self.output_parser
            
            return chain
            
        except Exception as e:
            logger.error(f"대화형 체인 생성 오류: {e}")
            raise LLMServiceError(f"대화형 체인 생성 실패: {str(e)}")


# 기존 호환성을 위한 함수들
def clamp_options(temperature=0.6, num_ctx=8192, num_predict=None):
    """옵션 정규화 (기존 호환성)"""
    t = max(0.0, min(float(temperature), 2.0))
    opts = {"temperature": t, "num_ctx": int(num_ctx)}
    if num_predict is not None:
        opts["num_predict"] = int(num_predict)
    return opts


def get_llm_service() -> Optional[LLMService]:
    """LLM 서비스 인스턴스 반환 (기존 호환성)"""
    try:
        if current_app:
            return current_app.extensions.get("llm_service")
    except Exception as e:
        logger.warning(f"LLM 서비스 인스턴스 조회 실패: {e}")
    return None