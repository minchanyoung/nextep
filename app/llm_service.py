import os
import sys
import json
import requests
import logging
import numpy as np # numpy 임포트
from app.core.exceptions import LLMServiceError
from app.utils.error_handler import handle_service_exceptions

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self, app=None):
        # 기본값 설정 (설정 로드 전에 사용)
        self.ollama_url = "http://localhost:11434"
        self.ollama_model = "exaone3.5:7.8b"
        self.ollama_embedding_model = "llama3"
        self.ollama_timeout = None
        self.default_options = {}

        if app:
            self.init_app(app)

    def init_app(self, app):
        # 설정 시스템에서 Ollama 설정 가져오기
        from app.config.settings import get_settings
        settings = get_settings()
        
        self.ollama_url = settings.ollama.url
        self.ollama_model = settings.ollama.model
        self.ollama_embedding_model = settings.ollama.embedding_model
        self.ollama_timeout = settings.ollama.timeout
        self.default_options = settings.ollama.get_default_options()
        
        app.extensions["llm_service"] = self

    def _prepare_payload(self, messages, stream=False, options=None):
        # 기본 최적화 옵션을 먼저 적용하고, 사용자 옵션으로 덮어쓰기
        merged_options = self.default_options.copy()
        if options:
            merged_options.update(options)
            
        payload = {
            "model": self.ollama_model, 
            "messages": messages, 
            "stream": bool(stream),
            "options": merged_options,
            "keep_alive": "30m"  # 모델을 30분간 메모리에 유지
        }
        return payload

    @handle_service_exceptions("LLM")
    def chat_sync(self, messages, options=None):
        try:
            r = requests.post(
                f"{self.ollama_url}/api/chat",
                json=self._prepare_payload(messages, stream=False, options=options),
                timeout=self.ollama_timeout,
            )
            r.raise_for_status()
            data = r.json()
            # 공통 에러 처리
            if isinstance(data, dict) and data.get("error"):
                raise LLMServiceError(data["error"])
            # 응답 추출 (Ollama 형식 호환)
            msg = data.get("message") if isinstance(data, dict) else None
            if isinstance(msg, dict) and "content" in msg:
                return msg["content"]
            if isinstance(data, dict) and "response" in data:
                return data["response"]
            return json.dumps(data, ensure_ascii=False)
        except requests.exceptions.ConnectionError:
            raise LLMServiceError("Ollama 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.")
        except requests.exceptions.Timeout:
            raise LLMServiceError(f"요청 시간 초과 (설정: {self.ollama_timeout}초)")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise LLMServiceError(f"모델 '{self.ollama_model}'을 찾을 수 없습니다. 모델이 설치되어 있는지 확인해주세요.")
            raise LLMServiceError(f"HTTP 오류 (상태 코드: {e.response.status_code})")
        except LLMServiceError:
            raise  # 이미 처리된 서비스 에러는 다시 발생
        except Exception as e:
            raise LLMServiceError(f"예기치 못한 오류: {str(e)}")

    def chat_stream(self, messages, options=None):
        try:
            r = requests.post(
                f"{self.ollama_url}/api/chat",
                json=self._prepare_payload(messages, stream=True, options=options),
                timeout=self.ollama_timeout,
                stream=True,
            )
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict) and data.get("error"):
                    yield f"[stream-error] {data['error']}"
                    break
                if data.get("done"):
                    break
                msg = data.get("message") or {}
                if isinstance(msg, dict) and "content" in msg and msg["content"]:
                    yield msg["content"]
                elif "response" in data and data["response"]:
                    yield data["response"]
        except Exception as e:
            logger.error(f"Ollama stream error: {e}")
            yield "AI 응답 생성 중 스트리밍 오류가 발생했습니다."

    def generate_embedding(self, text: str, model_name: str = None) -> list:
        model_to_use = model_name if model_name else self.ollama_embedding_model
        
        # 캐시에서 먼저 확인
        from app.utils.cache_manager import get_cache_manager
        cache_manager = get_cache_manager()
        
        cached_embedding = cache_manager.get_embedding(text, model_to_use)
        if cached_embedding is not None:
            return cached_embedding
        
        try:
            r = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": model_to_use,
                    "prompt": text,
                    "options": {
                        "num_thread": -1,  # CPU 스레드 자동 설정
                    },
                    "keep_alive": "30m"  # 임베딩 모델도 메모리에 유지
                },
                timeout=self.ollama_timeout,
            )
            r.raise_for_status()
            data = r.json()
            if "embedding" in data:
                embedding = data["embedding"]
                # 캐시에 저장
                cache_manager.cache_embedding(text, model_to_use, embedding)
                return embedding
            raise RuntimeError(f"Embedding not found in response: {data}")
        except requests.exceptions.ConnectionError:
            logger.error("Ollama 서버에 연결할 수 없습니다 (임베딩)")
            return []
        except requests.exceptions.Timeout:
            logger.error(f"Ollama 임베딩 요청 타임아웃 ({self.ollama_timeout}초)")
            return []
        except requests.exceptions.HTTPError as e:
            logger.error(f"Ollama 임베딩 HTTP 오류: {e}")
            if e.response.status_code == 404:
                logger.error(f"임베딩 모델 '{model_to_use}'을 찾을 수 없습니다.")
            return []
        except Exception as e:
            logger.error(f"Ollama embedding error with model {model_to_use}: {e}")
            return []

def clamp_options(temperature=0.6, num_ctx=8192, num_predict=None):
    t = max(0.0, min(float(temperature), 2.0))
    opts = {"temperature": t, "num_ctx": int(num_ctx)}
    if num_predict is not None:
        opts["num_predict"] = int(num_predict)
    return opts