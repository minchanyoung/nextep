import os
from typing import List, Dict
from threading import Thread

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from sentence_transformers import SentenceTransformer, CrossEncoder
import uvicorn
from starlette.responses import StreamingResponse

LLM_MODEL_NAME = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"
RERANKER_MODEL_NAME = "bongsoo/kpf-cross-encoder-v1" # Reranker 모델
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

try:
    print(f"LLM 모델 로딩 중: {LLM_MODEL_NAME}")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    if DEVICE == "cuda":
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            load_in_8bit=True,
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

    print(f"임베딩 모델 로딩 중: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

    print(f"Reranker 모델 로딩 중: {RERANKER_MODEL_NAME}")
    reranker_model = CrossEncoder(RERANKER_MODEL_NAME, device=DEVICE)

    print("모든 모델 로딩 완료.")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}") from e

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class GenerationRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_new_tokens: int = 1024
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



@app.get("/")
def read_root():
    return {
        "status": "Inference server is running",
        "device": DEVICE,
        "llm_model": LLM_MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranker_model": RERANKER_MODEL_NAME,
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
            return_tensors="pt"
        ).to(DEVICE)

        terminators = [
            llm_tokenizer.eos_token_id,
            llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        with torch.inference_mode():
            output_tokens = llm_model.generate(
                input_ids,
                max_new_tokens=int(request.max_new_tokens),
                eos_token_id=terminators,
                do_sample=True,
                temperature=float(request.temperature),
                top_p=float(request.top_p),
            )

        response = output_tokens[0][input_ids.shape[-1]:]
        result = llm_tokenizer.decode(response, skip_special_tokens=True)
        return {"result": result.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GENERATE_ERROR: {e}")


@app.post("/generate_stream")
async def generate_stream(request: GenerationRequest):
    try:
        input_ids = llm_tokenizer.apply_chat_template(
            request.messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(DEVICE)

        streamer = TextIteratorStreamer(llm_tokenizer, skip_prompt=True, skip_special_tokens=True)

        terminators = [
            llm_tokenizer.eos_token_id,
            llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=int(request.max_new_tokens),
            eos_token_id=terminators,
            do_sample=True,
            temperature=float(request.temperature),
            top_p=float(request.top_p),
        )

        thread = Thread(target=llm_model.generate, kwargs=generation_kwargs)
        thread.start()

        return StreamingResponse(streamer, media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GENERATE_STREAM_ERROR: {e}")


@app.post("/embed", response_model=EmbeddingResponse)
def embed(request: EmbeddingRequest):
    try:
        embeddings = embedding_model.encode(
            request.texts,
            convert_to_numpy=True,
            normalize_embeddings=request.normalize,
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
        
        # 점수와 문서를 묶어서 점수 기준으로 내림차순 정렬
        scored_docs = sorted(zip(scores, request.documents), key=lambda x: x[0], reverse=True)
        
        # 정렬된 문서만 추출
        reranked_docs = [doc for score, doc in scored_docs]
        
        return {"reranked_documents": reranked_docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RERANK_ERROR: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
