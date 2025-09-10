
import os
import sys
import logging
from dotenv import load_dotenv

# .env 파일에서 환경변수를 로드합니다.
load_dotenv()

# 프로젝트 루트를 시스템 경로에 추가하여 'app' 모듈을 임포트할 수 있도록 함
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app
from app.rag_manager import get_rag_manager
from config import Config

# 기본 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ingest_data():
    """
    rag_data 디렉토리의 모든 PDF 파일을 스캔하여 벡터 데이터베이스에 인제스트(저장)합니다.
    """
    logging.info("데이터 인제스트 스크립트를 시작합니다.")
    
    # Flask 앱 컨텍스트 생성
    # create_app 함수는 config 객체를 인자로 받습니다.
    app = create_app(Config)
    with app.app_context():
        logging.info("Flask 애플리케이션 컨텍스트를 성공적으로 생성했습니다.")
        
        rag_manager = get_rag_manager()
        if not rag_manager or not rag_manager._initialized:
            logging.error("RAG Manager가 초기화되지 않았습니다. app/__init__.py 설정을 확인하세요.")
            return

        # rag_data 디렉토리 경로 설정
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        rag_data_dir = os.path.join(project_root, 'rag_data')

        if not os.path.isdir(rag_data_dir):
            logging.error(f"'{rag_data_dir}' 디렉토리를 찾을 수 없습니다. 디렉토리를 생성하고 PDF 파일을 넣어주세요.")
            return

        logging.info(f"'{rag_data_dir}' 디렉토리에서 PDF 파일을 스캔합니다...")

        pdf_files = []
        for root, _, files in os.walk(rag_data_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))

        if not pdf_files:
            logging.warning(f"'{rag_data_dir}' 디렉토리에서 인제스트할 PDF 파일을 찾지 못했습니다.")
            return

        logging.info(f"총 {len(pdf_files)}개의 PDF 파일을 인제스트 대상으로 찾았습니다.")

        successful_ingests = 0
        for pdf_path in pdf_files:
            try:
                logging.info(f"-> 처리 중: '{os.path.basename(pdf_path)}'")
                success = rag_manager.ingest_pdf_document(pdf_path)
                if success:
                    successful_ingests += 1
                else:
                    logging.warning(f"'{os.path.basename(pdf_path)}' 파일 처리 중 문서를 추가하지 못했습니다.")
            except Exception as e:
                logging.error(f"'{os.path.basename(pdf_path)}' 파일 처리 중 심각한 오류 발생: {e}", exc_info=True)

        logging.info("=" * 60)
        logging.info("데이터 인제스트 프로세스 완료.")
        logging.info(f"총 {len(pdf_files)}개의 PDF 파일 중 {successful_ingests}개가 성공적으로 처리되었습니다.")
        logging.info("이제 챗봇이 PDF 내용을 기반으로 답변할 수 있습니다.")
        logging.info("=" * 60)

if __name__ == '__main__':
    ingest_data()
