"""
LangChain 기반 RAG 시스템 (완전 마이그레이션 버전)
기존 ChromaDB 직접 사용을 LangChain으로 완전 통합
"""

import os
import logging
from typing import List, Dict, Optional, Union
from flask import current_app

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from app.document_processor import DocumentProcessor
from app.rag_data import LABOR_MARKET_TRENDS, LEARNING_RECOMMENDATIONS
from app.core.exceptions import RAGError

logger = logging.getLogger(__name__)


class RAGManager:
    """LangChain 기반 완전 통합 RAG 시스템"""
    
    def __init__(self, app=None):
        """RAG 관리자 초기화"""
        self.vector_store = None
        self.embedding_model = None
        self.text_splitter = None
        self.retriever = None
        self.rag_chain = None
        self.document_processor = None
        self._initialized = False
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Flask 앱과 연결"""
        try:
            # 설정 로드
            from app.config.settings import get_settings
            settings = get_settings()
            
            # 임베딩 모델 초기화
            embed_params = {
                "base_url": settings.ollama.url,
                "model": settings.ollama.embedding_model
            }
            
            # timeout이 설정되어 있을 때만 추가
            if settings.ollama.timeout:
                embed_params["timeout"] = settings.ollama.timeout
                
            self.embedding_model = OllamaEmbeddings(**embed_params)
            
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
                embedding_function=self.embedding_model,
                collection_name="labor_market_docs"
            )
            
            # 리트리버 초기화
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "score_threshold": 0.7,
                    "k": 5
                }
            )
            
            # 문서 프로세서 초기화
            self.document_processor = DocumentProcessor()
            
            # RAG 체인 초기화
            self._initialize_rag_chain()
            
            app.extensions['rag_manager'] = self
            self._initialized = True
            
            logger.info("LangChain 기반 RAG 관리자가 초기화되었습니다.")
            
        except Exception as e:
            logger.error(f"RAG 관리자 초기화 실패: {e}")
            raise RAGError(f"RAG 시스템 초기화 실패: {str(e)}")
    
    def _initialize_rag_chain(self):
        """RAG 체인 초기화"""
        try:
            from app.llm_service import get_llm_service
            
            llm_service = get_llm_service()
            if not llm_service:
                logger.warning("LLM 서비스를 사용할 수 없어 RAG 체인 초기화를 건너뜁니다.")
                return
            
            # RAG 프롬프트 템플릿
            rag_prompt = ChatPromptTemplate.from_template("""
            당신은 한국의 노동시장 및 커리어 전문가입니다. 
            제공된 문서 정보를 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공해주세요.

            관련 문서:
            {context}

            질문: {question}

            답변:
            """)
            
            # RAG 체인 생성
            self.rag_chain = (
                {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
                | rag_prompt
                | llm_service.chat_model
                | StrOutputParser()
            )
            
        except Exception as e:
            logger.error(f"RAG 체인 초기화 실패: {e}")
    
    def _format_docs(self, docs):
        """문서 포맷팅"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def ingest_pdf_document(self, pdf_path: str, llm_service=None) -> bool:
        """PDF 문서를 처리하여 벡터 데이터베이스에 저장"""
        try:
            logger.info(f"PDF 문서 인제스트 시작: {pdf_path}")
            
            # PDF 처리 및 청킹
            chunks = self.document_processor.process_pdf_file(pdf_path)
            if not chunks:
                logger.error("PDF 처리 결과가 없습니다.")
                return False
            
            # LangChain Document 객체로 변환
            documents = []
            for chunk in chunks:
                content = chunk.get('content', '')
                if not content.strip():
                    continue
                
                metadata = {
                    'source': chunk.get('source', pdf_path),
                    'page': chunk.get('page', 0),
                    'chunk_type': chunk.get('chunk_type', 'text'),
                    'keywords': ','.join(chunk.get('keywords', [])),
                    'content_length': len(content)
                }
                
                documents.append(Document(page_content=content, metadata=metadata))
            
            if not documents:
                logger.warning("변환할 문서가 없습니다.")
                return False
            
            # 벡터 스토어에 추가
            self.vector_store.add_documents(documents)
            
            logger.info(f"PDF 문서 인제스트 완료: {len(documents)}개 문서")
            return True
            
        except Exception as e:
            logger.error(f"PDF 인제스트 실패: {e}")
            return False
    
    def ingest_legacy_data(self, llm_service=None) -> bool:
        """기존 하드코딩된 RAG 데이터를 벡터 데이터베이스로 이전"""
        try:
            logger.info("기존 RAG 데이터 이전 시작...")
            
            documents = []
            
            # 노동시장 트렌드 데이터 변환
            for idx, item in enumerate(LABOR_MARKET_TRENDS):
                content = item.get("content", "")
                if content.strip():
                    metadata = {
                        'source': 'legacy_labor_market',
                        'chunk_type': 'labor_trend',
                        'keywords': ','.join(item.get('keywords', [])),
                        'id': f"labor_trend_{idx}",
                        'content_length': len(content)
                    }
                    documents.append(Document(page_content=content, metadata=metadata))
            
            # 학습 추천 데이터 변환
            for idx, item in enumerate(LEARNING_RECOMMENDATIONS):
                content = item.get("description", "")
                if content.strip():
                    metadata = {
                        'source': 'legacy_learning',
                        'chunk_type': 'learning_rec',
                        'keywords': ','.join(item.get('keywords', [])),
                        'category': item.get('category', ''),
                        'id': f"learning_rec_{idx}",
                        'content_length': len(content)
                    }
                    documents.append(Document(page_content=content, metadata=metadata))
            
            if documents:
                # 벡터 스토어에 추가
                self.vector_store.add_documents(documents)
                logger.info(f"기존 RAG 데이터 이전 완료: {len(documents)}개 문서")
                return True
            else:
                logger.warning("이전할 기존 데이터가 없습니다.")
                return False
                
        except Exception as e:
            logger.error(f"기존 데이터 이전 실패: {e}")
            return False
    
    def get_labor_market_info(self, query_text: str, top_k: int = 3, llm_service=None) -> str:
        """노동시장 정보 검색 (LangChain 기반)"""
        try:
            # 노동시장 관련 문서만 필터링하여 검색
            filter_query = {"chunk_type": "labor_trend"}
            
            # 유사도 검색
            docs = self.vector_store.similarity_search_with_score(
                query_text, 
                k=top_k,
                filter=filter_query
            )
            
            if not docs:
                return ""
            
            # 결과 포맷팅
            context_parts = []
            for doc, score in docs:
                if score < 0.8:  # 유사도 임계값
                    content = doc.page_content
                    keywords = doc.metadata.get('keywords', '')
                    
                    context_parts.append(f"관련도: {1-score:.1%}")
                    if keywords:
                        context_parts.append(f"키워드: {keywords}")
                    context_parts.append(f"내용: {content}")
                    context_parts.append("---")
            
            return "\n".join(context_parts) if context_parts else ""
            
        except Exception as e:
            logger.error(f"노동시장 정보 검색 실패: {e}")
            return ""
    
    def get_learning_recommendations(self, query_text: str, top_k: int = 3, llm_service=None) -> str:
        """학습 추천 정보 검색 (LangChain 기반)"""
        try:
            # 학습 추천 관련 문서만 필터링하여 검색
            filter_query = {"chunk_type": "learning_rec"}
            
            # 유사도 검색
            docs = self.vector_store.similarity_search_with_score(
                query_text, 
                k=top_k,
                filter=filter_query
            )
            
            if not docs:
                return ""
            
            # 결과 포맷팅
            context_parts = []
            for doc, score in docs:
                if score < 0.8:  # 유사도 임계값
                    content = doc.page_content
                    keywords = doc.metadata.get('keywords', '')
                    category = doc.metadata.get('category', '')
                    
                    context_parts.append(f"관련도: {1-score:.1%}")
                    if category:
                        context_parts.append(f"분야: {category}")
                    if keywords:
                        context_parts.append(f"키워드: {keywords}")
                    context_parts.append(f"내용: {content}")
                    context_parts.append("---")
            
            return "\n".join(context_parts) if context_parts else ""
            
        except Exception as e:
            logger.error(f"학습 추천 검색 실패: {e}")
            return ""
    
    def get_career_advice(self, query_text: str, top_k: int = 5) -> str:
        """종합적인 커리어 조언 검색 (RAG 체인 사용)"""
        try:
            if not self.rag_chain:
                logger.warning("RAG 체인이 초기화되지 않았습니다.")
                return ""
            
            # RAG 체인을 통한 답변 생성
            response = self.rag_chain.invoke(query_text)
            return response
            
        except Exception as e:
            logger.error(f"커리어 조언 생성 실패: {e}")
            return ""
    
    def search_documents(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """범용 문서 검색 인터페이스"""
        try:
            # 필터링된 유사도 검색
            docs = self.vector_store.similarity_search_with_score(
                query, 
                k=top_k,
                filter=filters
            )
            
            results = []
            for doc, score in docs:
                result = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"문서 검색 실패: {e}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """컬렉션 통계 정보"""
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                'total_documents': count,
                'collection_name': collection.name,
                'initialized': self._initialized
            }
            
        except Exception as e:
            logger.error(f"통계 정보 조회 실패: {e}")
            return {'error': str(e)}
    
    def delete_documents(self, filters: Optional[Dict] = None) -> bool:
        """문서 삭제"""
        try:
            if filters:
                # 필터 조건에 맞는 문서들 삭제
                docs = self.vector_store.similarity_search("", k=1000, filter=filters)
                if docs:
                    ids = [doc.metadata.get('id') for doc in docs if doc.metadata.get('id')]
                    if ids:
                        self.vector_store._collection.delete(ids=ids)
                        logger.info(f"{len(ids)}개 문서가 삭제되었습니다.")
                        return True
            return False
            
        except Exception as e:
            logger.error(f"문서 삭제 실패: {e}")
            return False


def get_rag_manager() -> Optional[RAGManager]:
    """RAG 관리자 인스턴스 반환 (기존 호환성)"""
    try:
        if current_app:
            return current_app.extensions.get('rag_manager')
    except Exception as e:
        logger.warning(f"RAG 관리자 인스턴스 조회 실패: {e}")
    return None