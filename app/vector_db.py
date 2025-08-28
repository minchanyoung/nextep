# app/vector_db.py
"""
벡터 데이터베이스 관리 시스템 (LangChain 통합됨)
이 파일은 호환성을 위해 유지되며, 실제 구현은 rag_manager.py에서 처리됩니다.
"""

import logging
import hashlib
from typing import List, Dict
import chromadb
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self, persist_directory: str = "chroma_db"):
        """
        벡터 데이터베이스 초기화
        Args:
            persist_directory: 데이터베이스 저장 경로
        """
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Chroma 데이터베이스 초기화"""
        try:
            # Chroma 클라이언트 생성 (로컬 저장)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # 컬렉션 생성 또는 가져오기
            try:
                self.collection = self.client.get_collection("labor_market_docs")
                logger.info("기존 벡터 데이터베이스 컬렉션을 로드했습니다.")
            except:
                self.collection = self.client.create_collection(
                    name="labor_market_docs",
                    metadata={"description": "노동시장 관련 문서 벡터 저장소"}
                )
                logger.info("새로운 벡터 데이터베이스 컬렉션을 생성했습니다.")
                
        except Exception as e:
            logger.error(f"벡터 데이터베이스 초기화 실패: {e}")
            raise e
    
    def _generate_document_id(self, content: str, source: str, page: int = None) -> str:
        """문서 청크의 고유 ID 생성"""
        base_string = f"{source}_{page}_{content[:100]}" if page else f"{source}_{content[:100]}"
        return hashlib.md5(base_string.encode()).hexdigest()
    
    def add_document_chunks(self, chunks: List[Dict], llm_service=None) -> bool:
        """
        문서 청크들을 벡터 데이터베이스에 추가
        Args:
            chunks: 문서 청크 리스트 
            llm_service: 임베딩 생성용 LLM 서비스
        Returns:
            bool: 성공 여부
        """
        try:
            if not chunks:
                logger.warning("추가할 문서 청크가 없습니다.")
                return False
            
            documents = []
            metadatas = []
            ids = []
            embeddings = []
            
            for chunk in chunks:
                # 텍스트 내용
                content = chunk.get('content', '')
                if not content.strip():
                    continue
                    
                # 메타데이터 (Chroma는 리스트를 지원하지 않으므로 문자열로 변환)
                keywords = chunk.get('keywords', [])
                keywords_str = ','.join(keywords) if isinstance(keywords, list) else str(keywords)
                
                metadata = {
                    'source': chunk.get('source', 'unknown'),
                    'page': chunk.get('page', 0),
                    'chunk_type': chunk.get('chunk_type', 'text'),
                    'keywords': keywords_str,
                    'created_at': datetime.now().isoformat(),
                    'content_length': len(content)
                }
                
                # 고유 ID 생성
                doc_id = self._generate_document_id(content, metadata['source'], metadata['page'])
                
                # 중복 체크
                try:
                    existing = self.collection.get(ids=[doc_id])
                    if existing['ids']:
                        logger.debug(f"문서 청크 {doc_id}는 이미 존재합니다.")
                        continue
                except:
                    pass
                
                documents.append(content)
                metadatas.append(metadata)
                ids.append(doc_id)
            
            if not documents:
                logger.warning("유효한 문서 청크가 없습니다.")
                return False
            
            # ChromaDB의 기본 임베딩 모델 사용 (차원 호환성을 위해 Ollama 임베딩 사용하지 않음)
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"{len(documents)}개 문서 청크를 벡터 데이터베이스에 추가했습니다.")
            return True
            
        except Exception as e:
            logger.error(f"문서 청크 추가 실패: {e}")
            return False
    
    def search_similar_documents(self, query: str, n_results: int = 3, llm_service=None) -> List[Dict]:
        """
        유사한 문서 검색
        Args:
            query: 검색 쿼리
            n_results: 반환할 결과 수
            llm_service: 임베딩 생성용 LLM 서비스 (현재 ChromaDB 기본 임베딩 사용으로 무시됨)
        Returns:
            List[Dict]: 검색 결과
        """
        try:
            if not self.collection.count():
                logger.warning("벡터 데이터베이스가 비어있습니다.")
                return []
            
            results = []
            
            # ChromaDB의 기본 임베딩 모델 사용 (차원 호환성을 위해)
            # Ollama 임베딩 대신 ChromaDB 내장 임베딩 사용
            search_results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count())
            )
            
            # 결과 정리
            if search_results and search_results['documents']:
                for i, doc in enumerate(search_results['documents'][0]):
                    metadata = search_results['metadatas'][0][i].copy()
                    
                    # 키워드 문자열을 다시 리스트로 변환
                    if 'keywords' in metadata and isinstance(metadata['keywords'], str):
                        metadata['keywords'] = [k.strip() for k in metadata['keywords'].split(',') if k.strip()]
                    
                    result = {
                        'content': doc,
                        'metadata': metadata,
                        'score': search_results['distances'][0][i] if search_results.get('distances') else 0,
                        'id': search_results['ids'][0][i]
                    }
                    results.append(result)
            
            logger.info(f"검색 완료: {len(results)}개 결과 반환")
            return results
            
        except Exception as e:
            logger.error(f"문서 검색 실패: {e}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """컬렉션 통계 정보 반환"""
        try:
            count = self.collection.count()
            
            # 메타데이터 통계
            all_docs = self.collection.get()
            sources = set()
            pages = set()
            
            for metadata in all_docs.get('metadatas', []):
                sources.add(metadata.get('source', 'unknown'))
                pages.add(metadata.get('page', 0))
            
            return {
                'total_documents': count,
                'unique_sources': len(sources),
                'sources': list(sources),
                'total_pages': len(pages),
                'collection_name': self.collection.name
            }
        except Exception as e:
            logger.error(f"통계 정보 조회 실패: {e}")
            return {}
    
    def delete_by_source(self, source: str) -> bool:
        """특정 소스의 모든 문서 삭제"""
        try:
            # 해당 소스의 문서 ID 조회
            results = self.collection.get(
                where={"source": source}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"소스 '{source}'의 {len(results['ids'])}개 문서를 삭제했습니다.")
                return True
            else:
                logger.info(f"소스 '{source}'에 해당하는 문서가 없습니다.")
                return False
                
        except Exception as e:
            logger.error(f"문서 삭제 실패: {e}")
            return False
    
    def reset_collection(self) -> bool:
        """전체 컬렉션 초기화"""
        try:
            self.client.delete_collection("labor_market_docs")
            self.collection = self.client.create_collection(
                name="labor_market_docs",
                metadata={"description": "노동시장 관련 문서 벡터 저장소"}
            )
            logger.info("벡터 데이터베이스를 초기화했습니다.")
            return True
        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {e}")
            return False