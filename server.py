# server.py
import os
from typing import List, Dict, Optional
from threading import Thread

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
from sentence_transformers import SentenceTransformer, CrossEncoder
from starlette.responses import StreamingResponse
from starlette.background import BackgroundTask
import uvicorn

# ===================== 설정 =====================
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "MLP-KTLim/llama-3-Korean-Bllossom-8B")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "jhgan/ko-sroberta-multitask")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "bongsoo/kpf-cross-encoder-v1")

# 토큰 길이 기본/상한 (원하셨던 1024~2048 권장범위 반영)
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", "1536"))
MAX_NEW_TOKENS_LIMIT = int(os.getenv("MAX_NEW_TOKENS_LIMIT", "2048"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[BOOT] Using device: {DEVICE}")

# 일부 GPU에서 연산 최적화(선택)
try:
    torch.set_float32_matmul_precision("high")  # 안전하면 성능에 도움
except Exception:
    pass

# ===================== 모델 로딩 =====================
try:
    print(f"[BOOT] LLM 토크나이저 로딩: {LLM_MODEL_NAME}")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    print(f"[BOOT] LLM 모델 로딩: {LLM_MODEL_NAME}")
    if DEVICE == "cuda":
        # 8bit 로딩으로 VRAM 절약 (지원 안 하면 fp16로 fallback)
        try:
            llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                load_in_8bit=True,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        except Exception:
            llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
    else:
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            trust_remote_code=True,
        ).to(DEVICE)
    llm_model.eval()

    print(f"[BOOT] 임베딩 모델 로딩: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

    print(f"[BOOT] Reranker 모델 로딩: {RERANKER_MODEL_NAME}")
    reranker_model = CrossEncoder(RERANKER_MODEL_NAME, device=DEVICE)

    print("[BOOT] 모든 모델 로딩 완료.")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}") from e

# ===================== FastAPI 앱 =====================
app = FastAPI(title="LLM Inference Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요 시 도메인 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== 데이터 모델 =====================
class GenerationRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_new_tokens: Optional[int] = DEFAULT_MAX_NEW_TOKENS
    temperature: float = 0.7
    top_p: float = 0.9

class GenerationResponse(BaseModel):
    result: str

class EmbeddingRequest(BaseModel):
    texts: List[str]
    normalize: bool = True

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]

class RerankRequest(BaseModel):
    query: str
    documents: List[str]

class RerankResponse(BaseModel):
    reranked_documents: List[str]

# ===================== 유틸 =====================
def _cap_tokens(v: Optional[int]) -> int:
    """요청 토큰 길이를 안전 상한으로 캡."""
    if v is None:
        return DEFAULT_MAX_NEW_TOKENS
    return max(1, min(int(v), MAX_NEW_TOKENS_LIMIT))

def _terminators():
    # eos / eot 토큰 ID 안전 수집 (None 제거)
    ids = [
        llm_tokenizer.eos_token_id,
        llm_tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    return [t for t in ids if t is not None]

# ===================== 엔드포인트 =====================
@app.get("/")
def read_root():
    return {
        "status": "Inference server is running",
        "device": DEVICE,
        "llm_model": LLM_MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranker_model": RERANKER_MODEL_NAME,
        "default_max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "max_new_tokens_limit": MAX_NEW_TOKENS_LIMIT,
    }

@app.get("/readyz")
def readyz():
    try:
        _ = llm_tokenizer.eos_token_id
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/generate", response_model=GenerationResponse)
def generate(request: GenerationRequest):
    try:
        input_ids = llm_tokenizer.apply_chat_template(
            request.messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(DEVICE)

        term = _terminators()
        max_new = _cap_tokens(request.max_new_tokens)

        with torch.inference_mode():
            output_tokens = llm_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new,
                eos_token_id=term if term else None,
                do_sample=True,
                temperature=float(request.temperature),
                top_p=float(request.top_p),
                use_cache=True,
            )

        response = output_tokens[0][input_ids.shape[-1]:]
        result = llm_tokenizer.decode(response, skip_special_tokens=True).strip()
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GENERATE_ERROR: {e}")

@app.post("/generate_stream")
def generate_stream(request: GenerationRequest):
    """
    텍스트 청크 스트리밍 (프록시 524 회피에 유리).
    """
    try:
        input_ids = llm_tokenizer.apply_chat_template(
            request.messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(DEVICE)

        # 토큰이 생성되는 대로 전달
        streamer = TextIteratorStreamer(
            llm_tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        term = _terminators()
        max_new = _cap_tokens(request.max_new_tokens)

        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=max_new,
            eos_token_id=term if term else None,
            do_sample=True,
            temperature=float(request.temperature),
            top_p=float(request.top_p),
            use_cache=True,
        )

        thread = Thread(target=llm_model.generate, kwargs=generation_kwargs, daemon=True)
        thread.start()

        # 프록시/브라우저 버퍼링 방지 헤더 + 종료 시 thread join
        return StreamingResponse(
            (chunk for chunk in streamer),
            media_type="text/plain",
            headers={
                "X-Accel-Buffering": "no",   # nginx 계열 버퍼링 off
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
            background=BackgroundTask(thread.join),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GENERATE_STREAM_ERROR: {e}")

@app.post("/generate_sse")
def generate_sse(request: GenerationRequest):
    """
    Server-Sent Events 스트리밍 (클라이언트가 EventSource로 받기 편함).
    """
    try:
        input_ids = llm_tokenizer.apply_chat_template(
            request.messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(DEVICE)

        streamer = TextIteratorStreamer(
            llm_tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        term = _terminators()
        max_new = _cap_tokens(request.max_new_tokens)

        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=max_new,
            eos_token_id=term if term else None,
            do_sample=True,
            temperature=float(request.temperature),
            top_p=float(request.top_p),
            use_cache=True,
        )

        thread = Thread(target=llm_model.generate, kwargs=generation_kwargs, daemon=True)
        thread.start()

        def event_source():
            for text in streamer:
                yield f"data: {text}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
            background=BackgroundTask(thread.join),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GENERATE_SSE_ERROR: {e}")

@app.post("/embed", response_model=EmbeddingResponse)
def embed(request: EmbeddingRequest):
    try:
        embeddings = embedding_model.encode(
            request.texts,
            convert_to_numpy=True,
            normalize_embeddings=bool(request.normalize),
        ).tolist()
        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EMBED_ERROR: {e}")

@app.post("/rerank", response_model=RerankResponse)
def rerank(request: RerankRequest):
    try:
        pairs = [[request.query, doc] for doc in request.documents]
        if not pairs:
            return {"reranked_documents": []}

        scores = reranker_model.predict(pairs)
        scored_docs = sorted(zip(scores, request.documents), key=lambda x: x[0], reverse=True)
        reranked_docs = [doc for score, doc in scored_docs]
        return {"reranked_documents": reranked_docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RERANK_ERROR: {e}")

# ===================== 엔트리 포인트 =====================
if __name__ == "__main__":
    # uvicorn 옵션으로 keep-alive 약간 늘려 524 가능성 더 낮춤
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=bool(int(os.getenv("RELOAD", "0"))),
        timeout_keep_alive=int(os.getenv("TIMEOUT_KEEP_ALIVE", "120")),
        # workers는 GPU/모델 공유 고려해서 환경에 맞게 조정 (기본 1)
    )
