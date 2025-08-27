# app/rag_manager.py
"""
RAG 시스템 통합 관리자
기존 하드코딩된 RAG 데이터와 벡터 데이터베이스를 통합 관리
"""

import os
import logging
from typing import List, Dict, Optional, Union
from flask import current_app
from app.vector_db import VectorDatabase
from app.document_processor import DocumentProcessor
from app.rag_data import LABOR_MARKET_TRENDS, LEARNING_RECOMMENDATIONS
import numpy as np

logger = logging.getLogger(__name__)

class RAGManager:
    def __init__(self, app=None):
        """RAG 관리자 초기화"""
        self.vector_db = None
        self.document_processor = None
        self.legacy_embeddings = {}  # 기존 임베딩 캐시
        self._initialized = False  # 초기화 상태 플래그
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Flask 앱과 연결"""
        try:
            # 벡터 데이터베이스 초기화
            persist_dir = os.path.join(app.instance_path, 'chroma_db')
            os.makedirs(persist_dir, exist_ok=True)
            
            self.vector_db = VectorDatabase(persist_directory=persist_dir)
            self.document_processor = DocumentProcessor()
            
            app.extensions['rag_manager'] = self
            logger.info("RAG 관리자가 초기화되었습니다.")
            
        except Exception as e:
            logger.error(f"RAG 관리자 초기화 실패: {e}")
            raise e
    
    def ingest_pdf_document(self, pdf_path: str, llm_service=None) -> bool:
        """
        PDF 문서를 처리하여 벡터 데이터베이스에 저장
        Args:
            pdf_path: PDF 파일 경로
            llm_service: 임베딩 생성용 LLM 서비스
        Returns:
            bool: 성공 여부
        """
        try:
            logger.info(f"PDF 문서 인제스트 시작: {pdf_path}")
            
            # 1. PDF 처리 및 청킹
            chunks = self.document_processor.process_pdf_file(pdf_path)
            if not chunks:
                logger.error("PDF 처리 결과가 없습니다.")
                return False
            
            # 2. 벡터 데이터베이스에 저장
            success = self.vector_db.add_document_chunks(chunks, llm_service)
            
            if success:
                logger.info(f"PDF 문서 인제스트 완료: {len(chunks)}개 청크")
                return True
            else:
                logger.error("벡터 데이터베이스 저장 실패")
                return False
                
        except Exception as e:
            logger.error(f"PDF 인제스트 실패: {e}")
            return False
    
    def ingest_legacy_data(self, llm_service=None) -> bool:
        """
        기존 하드코딩된 RAG 데이터를 벡터 데이터베이스로 이전
        Args:
            llm_service: 임베딩 생성용 LLM 서비스
        Returns:
            bool: 성공 여부
        """
        try:
            logger.info("기존 RAG 데이터 이전 시작")
            
            chunks = []
            
            # 노동시장 트렌드 데이터 변환 (처음 5개만 테스트)
            for item in LABOR_MARKET_TRENDS[:5]:  # 빠른 테스트를 위해 제한
                chunk = {
                    'content': f"제목: {item['title']}\n\n내용: {item['content']}",
                    'source': 'legacy_labor_market_trends',
                    'page': 0,
                    'chunk_type': 'labor_trend',
                    'keywords': item['keywords'],
                    'legacy_id': item['id']
                }
                chunks.append(chunk)
            
            # 학습 추천 데이터 변환 (처음 3개만 테스트)
            for item in LEARNING_RECOMMENDATIONS[:3]:  # 빠른 테스트를 위해 제한
                resources_text = ", ".join([f"{r['name']} ({r['type']})" for r in item['learning_resources']])
                content = f"기술명: {item['skill_name']}\n\n설명: {item['description']}\n\n학습자료: {resources_text}"
                
                chunk = {
                    'content': content,
                    'source': 'legacy_learning_recommendations',
                    'page': 0,
                    'chunk_type': 'learning_recommendation',
                    'keywords': item['keywords'],  # 리스트는 vector_db.py에서 문자열로 변환됨
                    'legacy_id': item['id']
                    # related_job_categories는 Chroma 메타데이터에서 제외 (리스트 타입)
                }
                chunks.append(chunk)
            
            # 벡터 데이터베이스에 저장 (임베딩 없이)
            success = self.vector_db.add_document_chunks(chunks, None)  # 임베딩 생성 건너뛰기
            
            if success:
                logger.info(f"기존 RAG 데이터 이전 완료: {len(chunks)}개 항목")
                return True
            else:
                logger.error("기존 RAG 데이터 이전 실패")
                return False
                
        except Exception as e:
            logger.error(f"기존 RAG 데이터 이전 실패: {e}")
            return False
    
    def _ensure_initialized(self, llm_service=None):
        """RAG 시스템이 초기화되었는지 확인하고 필요시 초기화"""
        if not self._initialized:
            try:
                logger.info("RAG 시스템 지연 초기화 시작...")
                
                # 기존 데이터가 있는지 확인
                stats = self.vector_db.get_collection_stats()
                if stats.get('total_documents', 0) == 0:
                    # 기존 데이터 이전
                    self.ingest_legacy_data(llm_service)
                
                self._initialized = True
                logger.info("RAG 시스템 지연 초기화 완료")
            except Exception as e:
                logger.error(f"RAG 시스템 초기화 실패: {e}")
    
    def search_documents(self, query: str, n_results: int = 3, llm_service=None, 
                        source_filter: Optional[str] = None) -> List[Dict]:
        """
        통합 문서 검색
        Args:
            query: 검색 쿼리
            n_results: 반환할 결과 수
            llm_service: 임베딩 생성용 LLM 서비스
            source_filter: 소스 필터 (예: 'pdf', 'legacy')
        Returns:
            List[Dict]: 검색 결과
        """
        try:
            # 필요시 초기화
            self._ensure_initialized(llm_service)
            
            # 벡터 데이터베이스에서 검색
            vector_results = self.vector_db.search_similar_documents(
                query=query, 
                n_results=n_results * 2,  # 필터링을 위해 더 많이 가져옴
                llm_service=llm_service
            )
            
            # 소스 필터링 적용
            if source_filter:
                if source_filter == 'pdf':
                    vector_results = [r for r in vector_results 
                                    if r['metadata']['source'].endswith('.pdf')]
                elif source_filter == 'legacy':
                    vector_results = [r for r in vector_results 
                                    if r['metadata']['source'].startswith('legacy_')]
            
            # 결과 수 제한
            return vector_results[:n_results]
            
        except Exception as e:
            logger.error(f"문서 검색 실패: {e}")
            return []
    
    def get_labor_market_info(self, query: str, top_n: int = 2, llm_service=None) -> str:
        """
        노동시장 정보 검색 (기존 함수 호환성 유지)
        """
        try:
            # 필요시 초기화
            self._ensure_initialized(llm_service)
            
            results = self.search_documents(
                query=query, 
                n_results=top_n,
                llm_service=llm_service,
                source_filter=None  # 모든 소스에서 검색
            )
            
            if not results:
                return ""
            
            # 관련 정보를 텍스트로 결합
            info_texts = []
            for result in results:
                content = result['content']
                source = result['metadata'].get('source', 'unknown')
                
                # PDF 소스인 경우 페이지 정보 추가
                if source.endswith('.pdf'):
                    page = result['metadata'].get('page', 0)
                    info_texts.append(f"[{source} p.{page}] {content}")
                else:
                    info_texts.append(content)
            
            return "\n\n".join(info_texts)
            
        except Exception as e:
            logger.error(f"노동시장 정보 검색 실패: {e}")
            return ""
    
    def get_learning_recommendations(self, query: str, top_n: int = 2, llm_service=None) -> str:
        """
        학습 추천 정보 검색 (기존 함수 호환성 유지)
        """
        try:
            # 필요시 초기화
            self._ensure_initialized(llm_service)
            
            # 학습 추천에 특화된 검색
            learning_query = f"학습 교육 추천 역량 강화 {query}"
            results = self.search_documents(
                query=learning_query,
                n_results=top_n,
                llm_service=llm_service
            )
            
            if not results:
                return ""
            
            # 학습 추천 형식으로 정리
            recommendations = []
            for result in results:
                content = result['content']
                metadata = result['metadata']
                
                # 학습 추천 데이터인 경우 특별 처리
                if metadata.get('chunk_type') == 'learning_recommendation':
                    recommendations.append(f"- {content}")
                elif 'learning' in content.lower() or '교육' in content or '훈련' in content:
                    recommendations.append(f"- {content}")
            
            return "\n".join(recommendations) if recommendations else ""
            
        except Exception as e:
            logger.error(f"학습 추천 검색 실패: {e}")
            return ""
    
    def get_database_stats(self) -> Dict:
        """데이터베이스 통계 정보"""
        try:
            stats = self.vector_db.get_collection_stats()
            
            # 추가 통계 계산
            if stats.get('total_documents', 0) > 0:
                try:
                    all_docs = self.vector_db.collection.get()
                    chunk_types = {}
                    sources = {}
                    
                    for metadata in all_docs.get('metadatas', []):
                        chunk_type = metadata.get('chunk_type', 'unknown')
                        source = metadata.get('source', 'unknown')
                        
                        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                        sources[source] = sources.get(source, 0) + 1
                    
                    stats['chunk_types'] = chunk_types
                    stats['source_distribution'] = sources
                except Exception as e:
                    logger.warning(f"통계 상세 정보 조회 실패: {e}")
            
            return stats
            
        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            return {}
    
    def refresh_database(self, llm_service=None) -> bool:
        """
        데이터베이스 초기화 및 재구성
        """
        try:
            logger.info("RAG 데이터베이스 새로고침 시작")
            
            # 데이터베이스 초기화
            self.vector_db.reset_collection()
            
            # 기존 데이터 재이전
            legacy_success = self.ingest_legacy_data(llm_service)
            
            # PDF 파일들 재처리
            data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
            pdf_success = True
            
            if os.path.exists(data_dir):
                for filename in os.listdir(data_dir):
                    if filename.endswith('.pdf'):
                        pdf_path = os.path.join(data_dir, filename)
                        if not self.ingest_pdf_document(pdf_path, llm_service):
                            pdf_success = False
                            logger.warning(f"PDF 재처리 실패: {filename}")
            
            success = legacy_success and pdf_success
            
            if success:
                logger.info("RAG 데이터베이스 새로고침 완료")
            else:
                logger.warning("RAG 데이터베이스 새로고침 부분 실패")
            
            return success
            
        except Exception as e:
            logger.error(f"데이터베이스 새로고침 실패: {e}")
            return False

# 중복 함수 제거 - utils.flask_utils로 이동됨
# 하위 호환성을 위해 import 유지
from app.utils.flask_utils import get_rag_manager