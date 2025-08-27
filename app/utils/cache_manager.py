# app/utils/cache_manager.py
"""
캐싱 관리자 - RAG 성능 최적화를 위한 다층 캐싱 시스템
"""

import json
import hashlib
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)


class LRUCache:
    """메모리 기반 LRU 캐시"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.RLock()
    
    def _is_expired(self, key: str) -> bool:
        """키가 만료되었는지 확인"""
        if key not in self.timestamps:
            return True
        return time.time() - self.timestamps[key] > self.ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회"""
        with self.lock:
            if key not in self.cache or self._is_expired(key):
                self._remove_if_exists(key)
                return None
            
            # LRU: 최근 사용한 항목을 끝으로 이동
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
    
    def put(self, key: str, value: Any):
        """캐시에 값 저장"""
        with self.lock:
            # 기존 키 제거
            self._remove_if_exists(key)
            
            # 용량 초과시 가장 오래된 항목 제거
            while len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                self._remove_if_exists(oldest_key)
            
            # 새 값 저장
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def _remove_if_exists(self, key: str):
        """키가 존재하면 제거"""
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]
    
    def clear(self):
        """캐시 비우기"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def size(self) -> int:
        """현재 캐시 크기"""
        return len(self.cache)


class RAGCacheManager:
    """RAG 시스템 전용 다층 캐싱 관리자"""
    
    def __init__(self, embedding_cache_size: int = 500, search_cache_size: int = 200):
        # 임베딩 캐시 (TTL: 1시간)
        self.embedding_cache = LRUCache(embedding_cache_size, 3600)
        
        # 검색 결과 캐시 (TTL: 30분)
        self.search_cache = LRUCache(search_cache_size, 1800)
        
        # 통계
        self.stats = {
            'embedding_hits': 0,
            'embedding_misses': 0,
            'search_hits': 0,
            'search_misses': 0,
            'total_requests': 0
        }
    
    def _generate_key(self, text: str, prefix: str = "") -> str:
        """텍스트를 기반으로 캐시 키 생성"""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:16]
        return f"{prefix}_{text_hash}" if prefix else text_hash
    
    def get_embedding(self, text: str, model_name: str) -> Optional[List[float]]:
        """임베딩 캐시에서 조회"""
        key = self._generate_key(f"{model_name}:{text}", "emb")
        result = self.embedding_cache.get(key)
        
        if result is not None:
            self.stats['embedding_hits'] += 1
            logger.debug(f"임베딩 캐시 히트: {text[:50]}...")
            return result
        else:
            self.stats['embedding_misses'] += 1
            return None
    
    def cache_embedding(self, text: str, model_name: str, embedding: List[float]):
        """임베딩을 캐시에 저장"""
        key = self._generate_key(f"{model_name}:{text}", "emb")
        self.embedding_cache.put(key, embedding)
        logger.debug(f"임베딩 캐시 저장: {text[:50]}...")
    
    def get_search_results(self, query: str, n_results: int, source_filter: Optional[str] = None) -> Optional[List[Dict]]:
        """검색 결과 캐시에서 조회"""
        cache_key = f"{query}:{n_results}:{source_filter or 'all'}"
        key = self._generate_key(cache_key, "search")
        result = self.search_cache.get(key)
        
        if result is not None:
            self.stats['search_hits'] += 1
            logger.debug(f"검색 캐시 히트: {query[:50]}...")
            return result
        else:
            self.stats['search_misses'] += 1
            return None
    
    def cache_search_results(self, query: str, n_results: int, results: List[Dict], source_filter: Optional[str] = None):
        """검색 결과를 캐시에 저장"""
        cache_key = f"{query}:{n_results}:{source_filter or 'all'}"
        key = self._generate_key(cache_key, "search")
        self.search_cache.put(key, results)
        logger.debug(f"검색 결과 캐시 저장: {query[:50]}...")
    
    def clear_all(self):
        """모든 캐시 비우기"""
        self.embedding_cache.clear()
        self.search_cache.clear()
        logger.info("RAG 캐시가 모두 비워졌습니다.")
    
    def get_stats(self) -> Dict:
        """캐시 통계 조회"""
        total_embedding_requests = self.stats['embedding_hits'] + self.stats['embedding_misses']
        total_search_requests = self.stats['search_hits'] + self.stats['search_misses']
        
        return {
            'embedding_cache': {
                'size': self.embedding_cache.size(),
                'max_size': self.embedding_cache.max_size,
                'hits': self.stats['embedding_hits'],
                'misses': self.stats['embedding_misses'],
                'hit_rate': self.stats['embedding_hits'] / max(total_embedding_requests, 1) * 100
            },
            'search_cache': {
                'size': self.search_cache.size(),
                'max_size': self.search_cache.max_size,
                'hits': self.stats['search_hits'],
                'misses': self.stats['search_misses'],
                'hit_rate': self.stats['search_hits'] / max(total_search_requests, 1) * 100
            },
            'total_requests': self.stats['total_requests']
        }


# 글로벌 캐시 매니저 인스턴스
_cache_manager = None


def get_cache_manager() -> RAGCacheManager:
    """RAG 캐시 매니저 인스턴스 반환 (싱글톤)"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = RAGCacheManager()
        logger.info("RAG 캐시 매니저가 초기화되었습니다.")
    return _cache_manager