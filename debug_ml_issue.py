#!/usr/bin/env python
"""
ML 예측이 작동하지 않는 문제를 디버깅하는 스크립트
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Flask 설정
from flask import Flask

def check_ml_setup():
    """ML 설정 상태 확인"""
    
    # Flask 앱 생성
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'debug-key'
    
    with app.app_context():
        from app.ml.routes import init_app
        
        print("=== ML 초기화 시도 ===")
        try:
            init_app(app)
            print("✓ ML 초기화 성공")
        except Exception as e:
            print(f"✗ ML 초기화 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # ML 리소스 확인
        ml_resources = app.extensions.get('ml_resources', {})
        print(f"\n=== ML 리소스 상태 ===")
        print(f"lgb_income: {'✓' if ml_resources.get('lgb_income') else '✗'}")
        print(f"cat_satis: {'✓' if ml_resources.get('cat_satis') else '✗'}")
        print(f"job_category_stats: {'✓' if ml_resources.get('job_category_stats') is not None else '✗'}")
        print(f"income_features: {len(ml_resources.get('income_features', []))} 개")
        print(f"satis_features: {len(ml_resources.get('satis_features', []))} 개")
        
        # 예측 함수 직접 테스트
        print(f"\n=== 예측 함수 직접 테스트 ===")
        user_input = {
            'age': 30,
            'gender': 1,
            'education': 3,
            'monthly_income': 300,
            'job_satisfaction': 4,
            'current_job_category': '2',
            'job_A_category': '1',
            'job_B_category': '3',
            'satis_wage': 4,
            'satis_stability': 3,
            'satis_growth': 4,
            'satis_task_content': 4,
            'satis_work_env': 3,
            'satis_work_time': 4,
            'satis_communication': 3,
            'satis_fair_eval': 4,
            'satis_welfare': 3
        }
        
        try:
            from app.ml.routes import run_prediction_with_proper_features
            results = run_prediction_with_proper_features(user_input)
            print(f"✓ 예측 성공: {results}")
            
            # 결과 분석
            for i, result in enumerate(results):
                scenario = result.get('scenario', f'시나리오{i}')
                income = result.get('income_change_rate', 0)
                satis = result.get('satisfaction_change_score', 0)
                print(f"  {scenario}: 소득={income}, 만족도={satis}")
                
        except Exception as e:
            print(f"✗ 예측 실패: {e}")
            import traceback
            traceback.print_exc()
        
        return True

if __name__ == "__main__":
    check_ml_setup()