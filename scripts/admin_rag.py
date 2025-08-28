#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 시스템 관리 도구
벡터 데이터베이스 관리, PDF 추가, 통계 조회 등의 관리 기능
"""

import os
import sys
import glob
sys.path.insert(0, os.path.abspath('.'))

from app import create_app
from app.config.settings import get_settings
import argparse
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGAdmin:
    def __init__(self):
        """RAG 관리자 초기화"""
        settings = get_settings()
        config = settings.to_flask_config()
        self.app = create_app(type('Config', (), config))
        
    def show_stats(self):
        """데이터베이스 통계 표시"""
        with self.app.app_context():
            from app.rag_manager import get_rag_manager
            
            rag_manager = get_rag_manager()
            if not rag_manager:
                print("RAG 매니저를 찾을 수 없습니다.")
                return
            
            stats = rag_manager.get_collection_stats()
            
            print("\n" + "=" * 50)
            print("RAG 데이터베이스 통계")
            print("=" * 50)
            
            print(f"총 문서 수: {stats.get('total_documents', 0)}")
            print(f"컬렉션 명: {stats.get('collection_name', 'N/A')}")
            
            # 상세 통계가 필요하면 rag_manager에 추가 구현 필요
            # 예: 소스별, 청크 유형별 분포 등

    def add_pdf(self, pdf_path):
        """PDF 파일 추가"""
        with self.app.app_context():
            from app.rag_manager import get_rag_manager
            
            rag_manager = get_rag_manager()
            
            if not rag_manager:
                print("RAG 매니저를 찾을 수 없습니다.")
                return False
            
            if not os.path.exists(pdf_path):
                print(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
                return False
            
            print(f"PDF 파일 처리 중: {pdf_path}")
            
            success = rag_manager.ingest_pdf_document(pdf_path)
            
            if success:
                print(f"PDF 파일이 성공적으로 추가되었습니다: {os.path.basename(pdf_path)}")
                return True
            else:
                print(f"PDF 파일 추가 실패: {os.path.basename(pdf_path)}")
                return False

    def ingest_all_pdfs(self):
        """'data' 디렉터리의 모든 PDF 파일을 처리"""
        print("'data' 디렉터리에서 모든 PDF 파일 인제스트를 시작합니다...")
        
        # 프로젝트 루트를 기준으로 'data' 디렉터리 경로 설정
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        data_dir = os.path.join(project_root, 'rag_data')
        
        pdf_files = glob.glob(os.path.join(data_dir, '*.pdf'))
        
        if not pdf_files:
            print("'rag_data' 디렉터리에서 PDF 파일을 찾을 수 없습니다.")
            return

        print(f"총 {len(pdf_files)}개의 PDF 파일을 찾았습니다.")
        
        success_count = 0
        fail_count = 0
        
        for pdf_path in pdf_files:
            if self.add_pdf(pdf_path):
                success_count += 1
            else:
                fail_count += 1
            print("-" * 60)

        print("\n" + "=" * 60)
        print("모든 PDF 파일 처리 완료!")
        print(f"성공: {success_count}개, 실패: {fail_count}개")
        print("=" * 60)
        
        # 최종 통계 표시
        self.show_stats()

    def search_documents(self, query, n_results=5):
        """문서 검색 테스트"""
        with self.app.app_context():
            from app.rag_manager import get_rag_manager
            
            rag_manager = get_rag_manager()
            
            if not rag_manager:
                print("RAG 매니저를 찾을 수 없습니다.")
                return
            
            print(f"\n 검색어: '{query}'")
            print("-" * 50)
            
            results = rag_manager.search_documents(
                query=query, 
                top_k=n_results
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
                print(f"키워드: {metadata.get('keywords', 'N/A')}")
                print(f"내용: {content[:300]}...")
                print("-" * 30)
    
    def delete_source(self, source_name):
        """특정 소스의 문서 삭제"""
        with self.app.app_context():
            from app.rag_manager import get_rag_manager
            
            rag_manager = get_rag_manager()
            
            if not rag_manager:
                print("RAG 매니저를 찾을 수 없습니다.")
                return
            
            print(f"소스 삭제 중: {source_name}")
            
            # RAGManager에 소스 삭제 기능이 필요. ChromaDB 직접 호출은 지양.
            # success = rag_manager.delete_documents_by_source(source_name) 
            # 아래는 임시 구현. RAGManager에 위임하는 것이 좋음.
            try:
                rag_manager.vector_store._collection.delete(where={"source": source_name})
                success = True
            except Exception as e:
                print(f"소스 삭제 중 오류: {e}")
                success = False

            if success:
                print("소스 삭제 완료")
                self.show_stats()
            else:
                print("소스 삭제 실패")

def main():
    parser = argparse.ArgumentParser(description="NEXTEP RAG 시스템 관리 도구")
    subparsers = parser.add_subparsers(dest='command', help='사용 가능한 명령어', required=True)
    
    # stats 명령어
    subparsers.add_parser('stats', help='데이터베이스 통계 표시')
    
    # add-pdf 명령어
    add_pdf_parser = subparsers.add_parser('add-pdf', help='단일 PDF 파일 추가')
    add_pdf_parser.add_argument('pdf_path', help='추가할 PDF 파일 경로')

    # ingest-all 명령어
    subparsers.add_parser('ingest-all', help="'data' 디렉터리의 모든 PDF 파일을 DB에 추가")
    
    # search 명령어
    search_parser = subparsers.add_parser('search', help='문서 검색')
    search_parser.add_argument('query', help='검색어')
    search_parser.add_argument('-n', '--n_results', type=int, default=5, help='반환할 결과 수')
    
    # delete-source 명령어
    delete_parser = subparsers.add_parser('delete-source', help='특정 소스(파일명)의 모든 문서 삭제')
    delete_parser.add_argument('source_name', help='삭제할 소스 파일 이름 (예: my_document.pdf)')
    
    args = parser.parse_args()
    
    admin = RAGAdmin()
    
    try:
        if args.command == 'stats':
            admin.show_stats()
        elif args.command == 'add-pdf':
            admin.add_pdf(args.pdf_path)
        elif args.command == 'ingest-all':
            admin.ingest_all_pdfs()
        elif args.command == 'search':
            admin.search_documents(args.query, args.n_results)
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