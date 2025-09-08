import logging
import os
import re
from typing import List, Dict, Iterator, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.core.exceptions import LLMServiceError
from app.utils.error_handler import handle_service_exceptions

logger = logging.getLogger(__name__)

_MIN_READ_TIMEOUT = int(os.getenv("LLM_MIN_READ_TIMEOUT", "300"))

_ROLE_START = ["### Assistant:", "assistant:", "Assistant:", "어시스턴트:"]
_ROLE_STOP  = ["### User:", "user:", "User:", "사용자:"]

def _strip_roles(s: str) -> str:
    if not s:
        return s
    cut = 0
    for m in _ROLE_START:
        i = s.rfind(m)
        if i != -1:
            cut = max(cut, i + len(m))
    if cut:
        s = s[cut:]
    stop = len(s)
    for m in _ROLE_STOP:
        j = s.find(m)
        if j != -1:
            stop = min(stop, j)
    return s[:stop].strip()

_LEAK_PATTERNS = re.compile(r"(규칙]|구조]|프롬프트|당신은 .*?입니다|시스템 지침)", re.S)
_REPEAT_FIX = re.compile(r"(\b[^\.!?]{3,}\b)([\.!?])(\s*\1\2)+")
_LIST_FIX = re.compile(r"(,\s*)+", re.S)
_BAN_PHRASE = re.compile(r"(4차 산업혁명,\s*AI,\s*IoT,\s*빅데이터,\s*클라우드,\s*블록체인,\s*사이버 보안)(?:\s*등)?", re.I)
_EN_KO_SPAM = re.compile(r"(?i)\b(NEXTE?P?T?\s+(Korea\s+)?(Job|Labor)\s+(Market|Trend|Forecast|Analysis)|Career\s+(Development|Framework))\b(\s*\1\b)+")

def _sanitize_output(s: str) -> str:
    s = _LEAK_PATTERNS.sub("", s)
    s = _BAN_PHRASE.sub("", s)
    s = _EN_KO_SPAM.sub(r"\1", s)
    s = _REPEAT_FIX.sub(r"\1\2", s)
    s = _LIST_FIX.sub(r", ", s)
    s = re.sub(r"(?m)^(.*)\n\1(\n|$)+", r"\1\2", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

class LLMService:
    def __init__(self, app=None):
        self.inference_server_url = ""
        self.connect_timeout = 10
        self.read_timeout = _MIN_READ_TIMEOUT
        self._session = self._build_session()
        if app:
            self.init_app(app)

    def _build_session(self) -> requests.Session:
        s = requests.Session()
        retry = Retry(total=3, backoff_factor=0.8, status_forcelist=[429, 502, 503, 504], allowed_methods=["GET", "POST"], raise_on_status=False)
        adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=64)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        return s

    def _load_timeouts_from_settings(self):
        ct_env = os.getenv("LLM_CONNECT_TIMEOUT")
        rt_env = os.getenv("LLM_READ_TIMEOUT")
        ct = int(ct_env) if ct_env and ct_env.isdigit() else None
        rt = int(rt_env) if rt_env and rt_env.isdigit() else None
        try:
            from app.config.settings import get_settings
            st = get_settings()
            base = (st.inference_server.url or "").strip()
            self.inference_server_url = base[:-1] if base.endswith("/") else base
            if rt is None:
                v = getattr(st.inference_server, "read_timeout", None)
                if isinstance(v, (int, float)):
                    rt = int(v)
            if ct is None:
                v = getattr(st.inference_server, "connect_timeout", None)
                if isinstance(v, (int, float)):
                    ct = int(v)
            legacy = getattr(st.inference_server, "timeout", None)
            if rt is None and isinstance(legacy, (int, float)):
                rt = int(legacy)
        except Exception as e:
            logger.warning(f"설정 로드 경고(기본값 사용): {e}")
        self.connect_timeout = max(1, int(ct if ct is not None else self.connect_timeout))
        self.read_timeout = int(rt if rt is not None else self.read_timeout)
        if self.read_timeout < _MIN_READ_TIMEOUT:
            logger.warning(f"read_timeout={self.read_timeout}s → 최소 {_MIN_READ_TIMEOUT}s로 상향")
            self.read_timeout = _MIN_READ_TIMEOUT

    def init_app(self, app):
        self._load_timeouts_from_settings()
        app.extensions["llm_service"] = self
        logger.info(f"LLM 서비스 초기화: {self.inference_server_url} (timeout=(connect:{self.connect_timeout}s, read:{self.read_timeout}s))")

    def _post_json(self, path: str, payload: dict) -> requests.Response:
        url = f"{self.inference_server_url}{path if path.startswith('/') else '/' + path}"
        t = (self.connect_timeout, self.read_timeout)
        resp = self._session.post(url, json=payload, timeout=t)
        if resp.status_code >= 400:
            logger.error(f"[LLM] POST {url} -> {resp.status_code} {resp.text[:200]}")
        resp.raise_for_status()
        return resp

    def _post_stream(self, path: str, payload: dict) -> Iterator[str]:
        url = f"{self.inference_server_url}{path if path.startswith('/') else '/' + path}"
        t = (self.connect_timeout, self.read_timeout)
        with self._session.post(url, json=payload, timeout=t, stream=True) as resp:
            if resp.status_code >= 400:
                first = ""
                try:
                    first = next(resp.iter_content(1024)).decode("utf-8", "ignore")
                except Exception:
                    pass
                logger.error(f"[LLM] STREAM {url} -> {resp.status_code} {first[:200]}")
                resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk:
                    yield chunk

    @handle_service_exceptions("LLM")
    def chat_sync(self, messages: List[Dict[str, str]], options: Optional[Dict] = None) -> str:
        try:
            fixed: List[Dict[str, str]] = []
            for i, m in enumerate(messages or []):
                role = (m.get("role") or "").strip()
                content = (m.get("content") or "").strip()
                if role not in {"system", "user", "assistant"} or not content:
                    raise LLMServiceError(f"messages[{i}] 형식 오류")
                fixed.append({"role": role, "content": content})
            if not fixed:
                raise LLMServiceError("messages 가 비었습니다.")
            
            opts = options or {}
            payload = {
                "messages": fixed,
                "max_new_tokens": int(opts.get("max_new_tokens", 1024)),
                "temperature": float(opts.get("temperature", 0.7)),
                "top_p": float(opts.get("top_p", 0.85)),
            }

            resp = self._post_json("/generate", payload)
            data = resp.json()
            return _sanitize_output(_strip_roles(data.get("result", "")))
        except Exception as e:
            raise LLMServiceError(f"LLM 처리 중 오류: {str(e)}")

    def chat_stream(self, messages: List[Dict[str, str]], options: Optional[Dict] = None) -> Iterator[str]:
        fixed = [{"role": (m.get("role") or "").strip(), "content": (m.get("content") or "").strip()} for m in (messages or [])]
        if not fixed:
            raise LLMServiceError("messages 가 비었습니다.")
        opts = options or {}
        payload = {
            "messages": fixed,
            "max_new_tokens": int(opts.get("max_new_tokens", 1024)),
            "temperature": float(opts.get("temperature", 0.7)),
            "top_p": float(opts.get("top_p", 0.85)),
        }
        for chunk in self._post_stream("/generate_stream", payload):
            yield _sanitize_output(chunk)

    def generate_embedding(self, text: str, model_name: Optional[str] = None) -> List[float]:
        vecs = self.embed_documents([text])
        return vecs[0] if vecs else []

    def embed_query(self, text: str) -> List[float]:
        return self.generate_embedding(text)

    def embed_documents(self, texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
        try:
            resp = self._post_json("/embed", {"texts": texts, "normalize": True})
            data = resp.json()
            return data.get("embeddings", [])
        except Exception as e:
            raise LLMServiceError(f"배치 임베딩 처리 중 오류: {str(e)}")

def get_llm_service():
    try:
        from flask import current_app
        if current_app:
            return current_app.extensions.get("llm_service")
    except Exception as e:
        logger.warning(f"LLM 서비스 인스턴스 조회 실패: {e}")
    return None
