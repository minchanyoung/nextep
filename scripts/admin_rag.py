#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 시스템 관리 도구
벡터 데이터베이스 관리, PDF 추가, 통계 조회 등의 관리 기능
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from app import create_app
from config import Config
import argparse
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGAdmin:
    def __init__(self):
        """RAG 관리자 초기화"""
        self.app = create_app(Config)
        
    def show_stats(self):
        """데이터베이스 통계 표시"""
        with self.app.app_context():
            from app.rag_manager import get_rag_manager
            
            rag_manager = get_rag_manager()
            if not rag_manager:
                print("RAG 매니저를 찾을 수 없습니다.")
                return
            
            stats = rag_manager.get_database_stats()
            
            print("\n" + "=" * 50)
            print("RAG 데이터베이스 통계")
            print("=" * 50)
            
            print(f"총 문서 수: {stats.get('total_documents', 0)}")
            print(f"고유 소스 수: {stats.get('unique_sources', 0)}")
            print(f"총 페이지 수: {stats.get('total_pages', 0)}")
            print(f"컬렉션 명: {stats.get('collection_name', 'N/A')}")
            
            if stats.get('sources'):
                print(f"\n소스 목록:")
                for source in stats['sources']:
                    print(f"  - {source}")
            
            if stats.get('chunk_types'):
                print(f"\n청크 유형별 분포:")
                for chunk_type, count in stats['chunk_types'].items():
                    print(f"  - {chunk_type}: {count}개")
            
            if stats.get('source_distribution'):
                print(f"\n소스별 문서 분포:")
                for source, count in stats['source_distribution'].items():
                    print(f"  - {source}: {count}개")
    
    def add_pdf(self, pdf_path):
        """PDF 파일 추가"""
        with self.app.app_context():
            from app.rag_manager import get_rag_manager
            
            rag_manager = get_rag_manager()
            llm_service = self.app.extensions.get("llm_service")
            
            if not rag_manager:
                print("RAG 매니저를 찾을 수 없습니다.")
                return
            
            if not os.path.exists(pdf_path):
                print(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
                return
            
            print(f"PDF 파일 처리 중: {pdf_path}")
            
            success = rag_manager.ingest_pdf_document(pdf_path, llm_service)
            
            if success:
                print("PDF 파일이 성공적으로 추가되었습니다.")
                self.show_stats()
            else:
                print("PDF 파일 추가 실패")
    
    def search_documents(self, query, n_results=5):
        """문서 검색 테스트"""
        with self.app.app_context():
            from app.rag_manager import get_rag_manager
            
            rag_manager = get_rag_manager()
            llm_service = self.app.extensions.get("llm_service")
            
            if not rag_manager:
                print("RAG 매니저를 찾을 수 없습니다.")
                return
            
            print(f"\n 검색어: '{query}'")
            print("-" * 50)
            
            results = rag_manager.search_documents(
                query=query,
                n_results=n_results,
                llm_service=llm_service
            )
            
            if not results:
                print("검색 결과가 없습니다.")
                return
            
            for i, result in enumerate(results, 1):
                metadata = result['metadata']
                content = result['content']
                score = result.get('score', 0)
                
                print(f"\n[결과 {i}] (유사도: {score:.3f})")
                print(f"소스: {metadata.get('source', 'N/A')}")
                print(f"페이지: {metadata.get('page', 'N/A')}")
                print(f"청크 타입: {metadata.get('chunk_type', 'N/A')}")
                print(f"키워드: {', '.join(metadata.get('keywords', []))}")
                print(f"내용: {content[:300]}...")
                print("-" * 30)
    
    def refresh_database(self):
        """데이터베이스 새로고침"""
        with self.app.app_context():
            from app.rag_manager import get_rag_manager
            
            rag_manager = get_rag_manager()
            llm_service = self.app.extensions.get("llm_service")
            
            if not rag_manager:
                print("RAG 매니저를 찾을 수 없습니다.")
                return
            
            print("데이터베이스 새로고침 중...")
            
            success = rag_manager.refresh_database(llm_service)
            
            if success:
                print("데이터베이스 새로고침 완료")
                self.show_stats()
            else:
                print("데이터베이스 새로고침 실패")
    
    def test_legacy_compatibility(self):
        """기존 함수 호환성 테스트"""
        with self.app.app_context():
            from app.services import retrieve_labor_market_info, retrieve_learning_recommendations
            
            print("\n🧪 기존 함수 호환성 테스트")
            print("=" * 50)
            
            test_queries = [
                "청년층 고용 문제",
                "제조업 전망",
                "정보통신업 성장"
            ]
            
            for query in test_queries:
                print(f"\n테스트 쿼리: '{query}'")
                
                # 노동시장 정보
                labor_info = retrieve_labor_market_info(query, top_n=2)
                if labor_info:
                    print(f"노동시장 정보: {len(labor_info)} 문자")
                    print(f"미리보기: {labor_info[:150]}...")
                else:
                    print("노동시장 정보 없음")
                
                # 학습 추천
                learning_info = retrieve_learning_recommendations(query, top_n=2)
                if learning_info:
                    print(f"학습 추천: {len(learning_info)} 문자")
                    print(f"미리보기: {learning_info[:150]}...")
                else:
                    print("학습 추천 없음")
    
    def delete_source(self, source_name):
        """특정 소스의 문서 삭제"""
        with self.app.app_context():
            from app.rag_manager import get_rag_manager
            
            rag_manager = get_rag_manager()
            
            if not rag_manager:
                print("RAG 매니저를 찾을 수 없습니다.")
                return
            
            print(f"소스 삭제 중: {source_name}")
            
            success = rag_manager.vector_db.delete_by_source(source_name)
            
            if success:
                print("소스 삭제 완료")
                self.show_stats()
            else:
                print("소스 삭제 실패")

def main():
    parser = argparse.ArgumentParser(description="NEXTEP RAG 시스템 관리 도구")
    subparsers = parser.add_subparsers(dest='command', help='사용 가능한 명령어')
    
    # stats 명령어
    subparsers.add_parser('stats', help='데이터베이스 통계 표시')
    
    # add-pdf 명령어
    add_pdf_parser = subparsers.add_parser('add-pdf', help='PDF 파일 추가')
    add_pdf_parser.add_argument('pdf_path', help='추가할 PDF 파일 경로')
    
    # search 명령어
    search_parser = subparsers.add_parser('search', help='문서 검색')
    search_parser.add_argument('query', help='검색어')
    search_parser.add_argument('-n', '--n_results', type=int, default=5, help='반환할 결과 수')
    
    # refresh 명령어
    subparsers.add_parser('refresh', help='데이터베이스 새로고침')
    
    # test 명령어
    subparsers.add_parser('test', help='기존 함수 호환성 테스트')
    
    # delete-source 명령어
    delete_parser = subparsers.add_parser('delete-source', help='특정 소스 삭제')
    delete_parser.add_argument('source_name', help='삭제할 소스 이름')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    admin = RAGAdmin()
    
    try:
        if args.command == 'stats':
            admin.show_stats()
        elif args.command == 'add-pdf':
            admin.add_pdf(args.pdf_path)
        elif args.command == 'search':
            admin.search_documents(args.query, args.n_results)
        elif args.command == 'refresh':
            admin.refresh_database()
        elif args.command == 'test':
            admin.test_legacy_compatibility()
        elif args.command == 'delete-source':
            admin.delete_source(args.source_name)
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n 작업이 중단되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()