#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
임시 RAG 데이터베이스 리셋 및 재입력 스크립트
"""

import os
import sys
import glob
import logging

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.abspath('.'))

from app import create_app
from config import Config

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_tasks():
    """DB 리셋 및 재입력 작업 수행"""
    app = create_app(Config)
    with app.app_context():
        logging.info("애플리케이션 컨텍스트 생성 완료.")
        from app.rag_manager import get_rag_manager
        rag_manager = get_rag_manager()

        if not rag_manager:
            logging.error("RAG 매니저를 가져오는 데 실패했습니다. 중단합니다.")
            return

        # 1. 데이터베이스 리셋
        logging.info("데이터베이스 초기화를 시도합니다...")
        if rag_manager.reset_database():
            logging.info("데이터베이스가 성공적으로 초기화되었습니다.")
        else:
            logging.error("데이터베이스 초기화에 실패했습니다. 중단합니다.")
            return

        # 2. 모든 PDF 파일 재입력
        logging.info("모든 PDF 파일의 재입력을 시작합니다...")
        project_root = os.path.abspath('.')
        data_dir = os.path.join(project_root, 'rag_data')
        pdf_files = glob.glob(os.path.join(data_dir, '*.pdf'))

        if not pdf_files:
            logging.warning("'rag_data' 디렉터리에서 PDF 파일을 찾지 못했습니다.")
        else:
            logging.info(f"{len(pdf_files)}개의 PDF 파일을 찾았습니다.")
            success_count = 0
            for pdf_path in pdf_files:
                logging.info(f"처리 중: {pdf_path}")
                if rag_manager.ingest_pdf_document(pdf_path):
                    success_count += 1
                    logging.info(f"성공적으로 추가됨: {os.path.basename(pdf_path)}")
                else:
                    logging.error(f"추가 실패: {os.path.basename(pdf_path)}")
            logging.info(f"입력 완료. 성공: {success_count}/{len(pdf_files)}.")

        # 3. 최종 통계 확인
        stats = rag_manager.get_collection_stats()
        logging.info(f"최종 데이터베이스 상태: {stats}")

if __name__ == "__main__":
    run_tasks()
