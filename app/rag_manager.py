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
                search_type="similarity_score_threshold",
                search_kwargs={
                    "score_threshold": 0.3,
                    "k": 5
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

    def get_career_advice(self, query_text: str, top_k: int = 5) -> str:
        """종합적인 커리어 조언 검색 (수동 RAG 로직)"""
        try:
            if not self.llm_service:
                logger.warning("LLM 서비스가 초기화되지 않았습니다.")
                return "LLM 서비스가 준비되지 않아 조언을 생성할 수 없습니다."

            # 1. 문서 검색 (Retrieval)
            retrieved_docs = self.retriever.invoke(query_text)
            if not retrieved_docs:
                logger.warning(f"'{query_text}'에 대한 관련 문서를 찾지 못했습니다.")
                # 관련 문서가 없어도 LLM에 직접 질문
                context_str = "관련 정보 없음"
            else:
                context_str = self._format_docs(retrieved_docs)

            # 2. 프롬프트 생성 (Augmentation)
            prompt = f"""
            당신은 사용자의 커리어 고민에 대해 조언해주는 전문가입니다.
            아래의 관련 문서를 참고하여 사용자의 질문에 대해 상세하고 친절하게 답변해주세요.
            
            [관련 문서]
            {context_str}
            
            [사용자 질문]
            {query_text}
            
            [답변]
            """

            # 3. LLM 호출 (Generation)
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_service.chat_sync(messages)
            return response
            
        except Exception as e:
            logger.error(f"커리어 조언 생성 실패: {e}")
            return "커리어 조언을 생성하는 중 오류가 발생했습니다."

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