import os
import logging
from typing import List, Dict, Optional
from flask import current_app

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.document_processor import DocumentProcessor
from app.core.exceptions import RAGError

logger = logging.getLogger(__name__)

class RAGManager:
    """새로운 LLMService와 연동되는 RAG 시스템"""
    
    def __init__(self, app=None):
        self.vector_store = None
        self.text_splitter = None
        self.retriever = None
        self.document_processor = None
        self.llm_service = None # LLM 서비스 저장
        self._initialized = False
        
        if app:
            self.init_app(app)
    
    def init_app(self, app, llm_service):
        """Flask 앱과 연결"""
        try:
            self.llm_service = llm_service # llm_service 인스턴스 저장

            # 텍스트 분할기 초기화
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # 벡터 데이터베이스 초기화
            persist_dir = os.path.join(app.instance_path, 'chroma_db')
            os.makedirs(persist_dir, exist_ok=True)
            
            self.vector_store = Chroma(
                persist_directory=persist_dir,
                # 임베딩 생성을 llm_service에 위임
                embedding_function=self.llm_service, 
                collection_name="labor_market_docs"
            )
            
            # 리트리버 초기화
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    'k': 5,
                    'fetch_k': 20
                }
            )
            
            # 문서 프로세서 초기화
            self.document_processor = DocumentProcessor()
            
            app.extensions['rag_manager'] = self
            self._initialized = True
            
            logger.info("RAG 관리자가 새로운 LLM 서비스와 함께 초기화되었습니다.")
            
        except Exception as e:
            logger.error(f"RAG 관리자 초기화 실패: {e}")
            raise RAGError(f"RAG 시스템 초기화 실패: {str(e)}")

    def _format_docs(self, docs: List[Document]) -> str:
        """문서 포맷팅"""
        return "\n\n".join(doc.page_content for doc in docs)

    def ingest_pdf_document(self, pdf_path: str) -> bool:
        """PDF 문서를 처리하여 벡터 데이터베이스에 저장"""
        try:
            logger.info(f"PDF 문서 인제스트 시작: {pdf_path}")
            chunks = self.document_processor.process_pdf_file(pdf_path)
            if not chunks:
                logger.error("PDF 처리 결과가 없습니다.")
                return False
            
            documents = [Document(page_content=chunk.get('content', ''), metadata={'source': pdf_path}) for chunk in chunks if chunk.get('content', '').strip()]
            
            if not documents:
                logger.warning("변환할 문서가 없습니다.")
                return False
            
            self.vector_store.add_documents(documents)
            logger.info(f"PDF 문서 인제스트 완료: {len(documents)}개 문서")
            return True
            
        except Exception as e:
            logger.error(f"PDF 인제스트 실패: {e}")
            return False

    def get_career_advice(self, query_text: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """종합적인 커리어 조언 검색 (수동 RAG 로직, 대화 기록 포함)"""
        try:
            if not self.llm_service:
                logger.warning("LLM 서비스가 초기화되지 않았습니다.")
                return "LLM 서비스가 준비되지 않아 조언을 생성할 수 없습니다."

            history = history or []

            # 1. 문서 검색 (Retrieval)
            retrieved_docs = self.retriever.invoke(query_text)
            if not retrieved_docs:
                logger.warning(f"'{query_text}'에 대한 관련 문서를 찾지 못했습니다.")
                context_str = "관련 정보 없음"
            else:
                # 1.5 문서 재정렬 (Reranking)
                doc_contents = [doc.page_content for doc in retrieved_docs]
                reranked_docs = self.llm_service.rerank_documents(query=query_text, documents=doc_contents)
                # 재정렬된 문서 중 상위 3개만 선택
                top_docs = reranked_docs[:3]
                logger.info(f"Rerank 후 상위 {len(top_docs)}개 문서를 컨텍스트로 사용합니다.")
                context_str = "\n\n".join(top_docs)

            # 2. 프롬프트 생성 (Augmentation)
            system_prompt = f"""
            당신은 사용자의 커리어 고민에 대해 조언해주는 전문가입니다.
            아래의 [관련 문서]를 최우선으로 참고하여 사용자의 질문에 대해 상세하고 친절하게 답변해주세요.
            문서에 내용이 없다면, 당신의 전문 지식을 활용해 답변해도 좋습니다.
            
            [관련 문서]
            {context_str}
            """
            
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            # 대화 기록을 메시지에 추가
            messages.extend(history)
            # 현재 사용자 질문을 메시지에 추가
            messages.append({"role": "user", "content": query_text})

            # 3. LLM 호출 (Generation)
            response = self.llm_service.chat_sync(messages)
            return response
            
        except Exception as e:
            logger.error(f"커리어 조언 생성 실패: {e}")
            return "커리어 조언을 생성하는 중 오류가 발생했습니다."

    def get_advice_with_sources(self, query_text: str, history: Optional[List[Dict[str, str]]] = None) -> Dict:
        """커리어 조언과 소스 문서를 함께 반환합니다."""
        try:
            if not self.llm_service:
                raise RAGError("LLM 서비스가 초기화되지 않았습니다.")

            history = history or []

            # 1. 문서 검색 (Retrieval)
            retrieved_docs = self.retriever.invoke(query_text)
            
            source_documents = []
            if not retrieved_docs:
                logger.warning(f"'{query_text}'에 대한 관련 문서를 찾지 못했습니다.")
                context_str = "관련 정보 없음"
            else:
                # 1.5 문서 재정렬 (Reranking)
                doc_contents = [doc.page_content for doc in retrieved_docs]
                reranked_contents = self.llm_service.rerank_documents(query=query_text, documents=doc_contents)

                # 재정렬된 내용에 맞춰 원본 Document 객체 순서 맞추기
                content_to_doc_map = {doc.page_content: doc for doc in retrieved_docs}
                reranked_docs_full = [content_to_doc_map[content] for content in reranked_contents if content in content_to_doc_map]

                top_docs = reranked_docs_full[:3]
                logger.info(f"Rerank 후 상위 {len(top_docs)}개 문서를 컨텍스트로 사용합니다.")
                
                context_str = "\n\n".join([d.page_content for d in top_docs])
                # 소스 정보를 포함한 객체 리스트 생성
                source_documents = [{
                    "content": d.page_content,
                    "source": d.metadata.get('source', '출처 정보 없음')
                } for d in top_docs]

            # 2. 프롬프트 생성 (Augmentation)
            system_prompt = f"""
            당신은 사용자의 커리어 고민에 대해 조언해주는 전문가입니다.
            아래의 [관련 문서]를 최우선으로 참고하여 사용자의 질문에 대해 상세하고 친절하게 답변해주세요.
            문서에 내용이 없다면, 당신의 전문 지식을 활용해 답변해도 좋습니다.
            
            [관련 문서]
            {context_str}
            """
            
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            messages.extend(history)
            messages.append({"role": "user", "content": query_text})

            # 3. LLM 호출 (Generation)
            answer = self.llm_service.chat_sync(messages)
            
            return {
                "answer": answer,
                "sources": source_documents
            }
            
        except Exception as e:
            logger.error(f"소스와 함께 조언 생성 실패: {e}")
            raise RAGError(f"소스와 함께 조언 생성 중 오류 발생: {str(e)}")

    def search_documents(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """범용 문서 검색 인터페이스"""
        try:
            docs = self.vector_store.similarity_search_with_score(
                query, 
                k=top_k,
                filter=filters
            )
            results = [{'content': doc.page_content, 'metadata': doc.metadata, 'score': score} for doc, score in docs]
            return results
        except Exception as e:
            logger.error(f"문서 검색 실패: {e}")
            return []

# ... (get_collection_stats, reset_database 등 나머지 메소드는 변경 없음) ...

def get_rag_manager() -> Optional[RAGManager]:
    """RAG 관리자 인스턴스 반환 (하위 호환성)"""
    try:
        if current_app:
            return current_app.extensions.get('rag_manager')
    except Exception as e:
        logger.warning(f"RAG 관리자 인스턴스 조회 실패: {e}")
    return None