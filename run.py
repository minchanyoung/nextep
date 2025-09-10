import os
import sys
import oracledb
from app import create_app, db
from app.models import User # User 모델을 임포트하여 Flask-Migrate가 인식하도록 함
from config import Config

# Oracle Instant Client 초기화
skip_oracle = os.environ.get('SKIP_ORACLE_INIT', '').lower() in ('1', 'true', 'yes')
if not skip_oracle:
    try:
        # 환경 변수에서 Oracle Client 경로 가져오기
        lib_dir = Config.ORACLE_CLIENT_LIB_DIR
        if lib_dir:
            oracledb.init_oracle_client(lib_dir=lib_dir)
        else:
            # 환경변수가 설정되지 않은 경우 기본값 사용 (개발환경용)
            oracledb.init_oracle_client(lib_dir=r"C:\oraclexe\instantclient_19_25")
            print("WARNING: ORACLE_CLIENT_LIB_DIR 환경변수가 설정되지 않아 기본 경로를 사용합니다.")
    except Exception as e:
        print("Oracle Instant Client 초기화 실패:", e)
        print("ORACLE_CLIENT_LIB_DIR 환경변수를 확인하거나 Oracle Client가 설치되어 있는지 확인해주세요.")
        print("Oracle 없이 서버를 실행하려면 SKIP_ORACLE_INIT=1 환경변수를 설정하세요.")
        exit(1)
else:
    print("Oracle Client 초기화를 건너뜁니다. (테스트 모드)")

app = create_app(Config)

# Flask CLI에서 db 객체와 User 모델을 사용할 수 있도록 노출
@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'User': User}

if __name__ == '__main__':
    # 빠른 시작 모드 확인
    fast_start = os.environ.get('FAST_START', '').lower() in ('1', 'true', 'yes')
    skip_preload = os.environ.get('SKIP_MODEL_PRELOAD', '').lower() in ('1', 'true', 'yes')
    
    print("=" * 60)
    print("NEXTEP Flask 서버가 시작됩니다!")
    print("URL: http://localhost:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)