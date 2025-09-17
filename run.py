import os
import sys
import oracledb
from app import create_app, db
from app.models import User # User 모델을 임포트하여 Flask-Migrate가 인식하도록 함
from config import Config

# Oracle Instant Client 초기화 (안전 모드)
skip_oracle = os.environ.get('SKIP_ORACLE_INIT', '').lower() in ('1', 'true', 'yes')
if not skip_oracle:
    try:
        # Oracle 클라이언트가 이미 초기화되었는지 확인
        if not hasattr(oracledb, '_initialized'):
            lib_dir = Config.ORACLE_CLIENT_LIB_DIR
            # Oracle 11g는 thick mode만 지원
            if lib_dir and os.path.exists(lib_dir):
                oracledb.init_oracle_client(lib_dir=lib_dir)
                print(f"Oracle Client 초기화 완료 (thick mode for Oracle 11g): {lib_dir}")
            elif os.path.exists(r"C:\oraclexe\instantclient_19_25"):
                oracledb.init_oracle_client(lib_dir=r"C:\oraclexe\instantclient_19_25")
                print("Oracle Client 초기화 완료 (thick mode for Oracle 11g): 기본 경로 사용")
            elif os.path.exists(r"C:\app\User\product\21c\dbhomeXE\bin"):
                oracledb.init_oracle_client(lib_dir=r"C:\app\User\product\21c\dbhomeXE\bin")
                print("Oracle Client 초기화 완료 (thick mode for Oracle 11g): XE 경로 사용")
            elif os.path.exists(r"C:\oraclexe\app\oracle\product\11.2.0\server\bin"):
                oracledb.init_oracle_client(lib_dir=r"C:\oraclexe\app\oracle\product\11.2.0\server\bin")
                print("Oracle Client 초기화 완료 (thick mode for Oracle 11g): Oracle 11g 경로 사용")
            else:
                print("Oracle Instant Client 경로를 찾을 수 없어 초기화를 건너뜁니다.")
                print("주의: Oracle 11g는 thin mode를 지원하지 않습니다.")
            oracledb._initialized = True
    except Exception as e:
        print(f"Oracle Instant Client 초기화 중 오류 (무시하고 계속): {e}")
        # 오류가 발생해도 서버 종료하지 않음
else:
    print("Oracle Client 초기화를 건너뜁니다.")

app = create_app(Config)

# Flask CLI에서 db 객체와 User 모델을 사용할 수 있도록 노출
@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'User': User}

if __name__ == '__main__':
    # 빠른 시작 모드 확인
    fast_start = os.environ.get('FAST_START', '').lower() in ('1', 'true', 'yes')
    skip_preload = os.environ.get('SKIP_MODEL_PRELOAD', '').lower() in ('1', 'true', 'yes')

    # 메모리 최적화 설정
    import gc
    gc.set_threshold(700, 10, 10)  # 가비지 컬렉션 임계값 조정

    print("=" * 60)
    print("NEXTEP Flask 서버가 시작됩니다!")
    print("URL: http://localhost:5000")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)