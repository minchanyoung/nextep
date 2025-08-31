import os
import sys
import oracledb
from app import create_app, db
from app.models import User # User 모델을 임포트하여 Flask-Migrate가 인식하도록 함
from config import Config

def preload_models():
    """모델 사전 로딩 함수"""
    print("LLM 모델 사전 로딩을 시작합니다...")
    try:
        from scripts.preload_models import preload_all_models
        success = preload_all_models()
        if success:
            print("[SUCCESS] 모든 모델이 성공적으로 사전 로딩되었습니다!")
            return True
        else:
            print("[WARNING] 일부 모델 로딩에 실패했지만 서버를 시작합니다.")
            return False
    except ImportError:
        print("[ERROR] scripts/preload_models.py를 찾을 수 없습니다. 모델 사전 로딩을 건너뜁니다.")
        return False
    except Exception as e:
        print(f"[ERROR] 모델 사전 로딩 중 오류 발생: {e}")
        print("서버를 시작하지만 첫 번째 요청이 느릴 수 있습니다.")
        return False

# Oracle Instant Client 초기화
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
    exit(1)

app = create_app(Config)

# Flask CLI에서 db 객체와 User 모델을 사용할 수 있도록 노출
@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'User': User}

if __name__ == '__main__':
    # 빠른 시작 모드 확인
    fast_start = os.environ.get('FAST_START', '').lower() in ('1', 'true', 'yes')
    skip_preload = os.environ.get('SKIP_MODEL_PRELOAD', '').lower() in ('1', 'true', 'yes')
    
    if fast_start:
        print("빠른 시작 모드: 모든 사전 로딩을 건너뜁니다.")
        print("첫 번째 AI 요청 시 초기화가 진행되어 응답이 느릴 수 있습니다.")
    elif not skip_preload:
        # 개발 서버 시작 전 모델 사전 로딩
        preload_success = preload_models()
        if preload_success:
            print("모델 사전 로딩 완료! Flask 서버를 시작합니다.")
        else:
            print("모델 사전 로딩 없이 Flask 서버를 시작합니다.")
    else:
        print("환경변수 설정으로 모델 사전 로딩을 건너뜁니다.")
    
    print("=" * 60)
    print("NEXTEP Flask 서버가 시작됩니다!")
    print("URL: http://localhost:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)