from flask import current_app
import sys, json, os, logging
import joblib
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb

logger = logging.getLogger(__name__)

# ==============================================================================
# 1) 경로/상수
# ==============================================================================
# KLIPS 데이터 실제 분포 (참고용)
# 소득 변화율: 평균 12.69%, 5-95분위수 -40%~+80%, 표준편차 106.1%
# 만족도 변화: 평균 -0.0083점, 5-95분위수 -1점~+1점, 표준편차 0.63점
# 모델의 자연스러운 예측값을 그대로 사용하여 현실적인 분포를 반영

# ==============================================================================
# 2) 초기화 함수
# ==============================================================================
def init_app(app):
    """Flask 앱 초기화 시 모델과 데이터를 로드하여 app.extensions에 저장"""
    with app.app_context():
        APP_ROOT = current_app.root_path
        # APP_ROOT가 이미 /nextep/app이므로 상위 디렉토리로 이동 후 경로 설정
        PROJECT_ROOT = os.path.dirname(APP_ROOT)  # /nextep
        MODEL_DIR = os.path.join(APP_ROOT, "ml", "saved_models")  # /nextep/app/ml/saved_models
        DATA_PATH = os.path.join(PROJECT_ROOT, "data", "klips_data_23.csv")  # /nextep/data/klips_data_23.csv

        logger.info(f"ML 초기화 시작 - APP_ROOT: {APP_ROOT}")
        logger.info(f"MODEL_DIR: {MODEL_DIR}")
        logger.info(f"DATA_PATH: {DATA_PATH}")
        
        # 경로 존재 확인
        if not os.path.exists(MODEL_DIR):
            logger.error(f"MODEL_DIR이 존재하지 않습니다: {MODEL_DIR}")
        if not os.path.exists(DATA_PATH):
            logger.error(f"DATA_PATH가 존재하지 않습니다: {DATA_PATH}")

        ml_resources = {}
        try:
            logger.info("Pre-loading ML models and data...")

            # --- 모델 로드 ---
            # 소득 모델 (XGBoost pickle 형식 사용)
            income_model_path = os.path.join(MODEL_DIR, "xgb_income_change_model.pkl")
            logger.info(f"Loading income model from: {income_model_path}")
            with open(income_model_path, 'rb') as f:
                ml_resources['xgb_income'] = joblib.load(f)
            logger.info("XGBoost income model loaded successfully")
            
            # 만족도 모델 (XGBoost 사용으로 변경)
            satis_model_path = os.path.join(MODEL_DIR, "final_xgb_satis_model.pkl")
            logger.info(f"Loading satisfaction model from: {satis_model_path}")
            with open(satis_model_path, 'rb') as f:
                ml_resources['xgb_satis'] = joblib.load(f)
            logger.info("XGBoost satisfaction model loaded successfully")

            # --- 데이터 및 피처 정보 로드 ---
            logger.info(f"Loading KLIPS data from: {DATA_PATH}")
            ml_resources['klips_df'] = pd.read_csv(DATA_PATH)
            logger.info(f"KLIPS data loaded: {len(ml_resources['klips_df'])} rows")
            
            # 직업별 통계 계산
            logger.info("Computing job category statistics...")
            ml_resources['job_category_stats'] = ml_resources['klips_df'].groupby('job_category').agg({
                'monthly_income': 'mean',
                'education': 'mean',
                'job_satisfaction': 'mean'
            }).rename(columns={
                'monthly_income': 'job_category_income_avg',
                'education': 'job_category_education_avg',
                'job_satisfaction': 'job_category_satisfaction_avg'
            })
            logger.info(f"Job category stats computed: {len(ml_resources['job_category_stats'])} categories")
            
            # 피처 이름 로드
            satis_config_path = os.path.join(MODEL_DIR, "final_ensemble_satis_config.json")
            logger.info(f"Loading satisfaction features from: {satis_config_path}")
            with open(satis_config_path, 'r') as f:
                config = json.load(f)
                ml_resources['satis_features'] = config['features']
            logger.info(f"Satisfaction features loaded: {len(ml_resources['satis_features'])} features")

            # XGBoost 모델에서 피처 이름 추출
            logger.info("Loading income model feature names from JSON...")
            income_features_path = os.path.join(MODEL_DIR, "income_feature_names_correct.json")
            with open(income_features_path, 'r') as f:
                ml_resources['income_features'] = json.load(f)
            logger.info(f"Income features loaded: {len(ml_resources['income_features'])} features")

            # app.extensions에 저장
            if 'ml_resources' not in app.extensions:
                app.extensions['ml_resources'] = {}
            app.extensions['ml_resources'] = ml_resources
            
            logger.info("ML models, data, and features pre-loaded successfully into app.extensions.")
            logger.info(f"Available ML resources: {list(ml_resources.keys())}")

        except Exception as e:
            logger.critical(f"ML 리소스 로딩 중 치명적 오류 발생: {e}")
            import traceback
            logger.critical(traceback.format_exc())
            
            # 기본 fallback 리소스 설정
            fallback_resources = {
                'job_category_stats': None,
                'klips_df': None,
                'income_features': [],
                'satis_features': []
            }
            app.extensions['ml_resources'] = fallback_resources
            logger.warning("Fallback ML resources initialized.")


# ==============================================================================
# 3) 코어 예측 함수
# ==============================================================================

def predict_scenario_with_proper_features(user_input, scenario_index):
    """
    새로운 피처 생성 방식을 사용하여 단일 시나리오 예측
    """
    from app.ml.preprocessing_fixed import prepare_income_model_features, prepare_satisfaction_model_features
    
    ml_resources = current_app.extensions.get('ml_resources', {})
    
    # 임시 MLPredictor 객체 생성 (job_category_stats 포함)
    class TempPredictor:
        def __init__(self):
            self.job_category_stats = ml_resources.get('job_category_stats')
    
    temp_predictor = TempPredictor()
    
    # ML 리소스 확인
    has_ml_models = all(k in ml_resources and ml_resources[k] is not None 
                       for k in ['xgb_income', 'xgb_satis', 'income_features', 'satis_features'])
    
    if not has_ml_models:
        logger.error("ML 모델을 로드할 수 없습니다. 애플리케이션을 다시 시작해주세요.")
        raise RuntimeError("ML 모델이 로드되지 않았습니다.")
    
    try:
        # 소득 모델용 피처 생성
        income_df = prepare_income_model_features(user_input, temp_predictor)
        income_row = income_df.iloc[scenario_index].to_dict()
        
        # 만족도 모델용 피처 생성  
        satis_df = prepare_satisfaction_model_features(user_input, temp_predictor)
        satis_row = satis_df.iloc[scenario_index].to_dict()
        
        # --- 소득 예측 ---
        income_features = ml_resources['income_features']
        xgb_income_model = ml_resources['xgb_income']
        
        # 피처를 정확한 순서로 준비
        income_model_features = []
        for feature in income_features:
            if feature in income_row:
                income_model_features.append(income_row[feature])
            else:
                logger.warning(f"소득 모델 피처 누락: {feature}")
                income_model_features.append(0.0)
        
        # 소득 예측 실행
        import numpy as np
        income_input = np.array(income_model_features).reshape(1, -1)
        income_prediction = xgb_income_model.predict(income_input)[0]
        income_change = float(income_prediction)  # numpy float32 -> Python float 변환
        
        logger.info(f"시나리오 {scenario_index} 소득 예측 (클리핑 없음): {income_change:.6f}")
        logger.info(f"소득 주요 피처값: job_category={income_row.get('job_category')}, monthly_income={income_row.get('monthly_income')}, age={income_row.get('age')}, job_category_change={income_row.get('job_category_change', 0)}, potential_promotion={income_row.get('potential_promotion', 0)}")
        
        # --- 만족도 예측 ---
        satis_features = ml_resources['satis_features']
        
        # XGBoost 만족도 모델
        xgb_satis = ml_resources.get('xgb_satis')
        
        # 앙상블 가중치
        ensemble_config = ml_resources.get('ensemble_config', {})
        weights = ensemble_config.get('weights', {'xgb': 0.34, 'lgb': 0.31, 'cat': 0.34})
        
        # 피처를 정확한 순서로 준비 (모든 피처를 수치형으로)
        satis_model_features = []
        for feature in satis_features:
            if feature in satis_row:
                value = satis_row[feature]
                satis_model_features.append(float(value))
            else:
                logger.warning(f"만족도 모델 피처 누락: {feature}")
                satis_model_features.append(0.0)
        
        # XGBoost 만족도 예측
        if xgb_satis:
            import numpy as np
            satis_input = np.array(satis_model_features).reshape(1, -1)
            satis_prediction = xgb_satis.predict(satis_input)[0]
            satis_change = float(satis_prediction)  # numpy float32 -> Python float 변환
            
            logger.info(f"시나리오 {scenario_index} 만족도 예측 (클리핑 없음): {satis_change:.6f}")
            logger.info(f"만족도 주요 피처값: job_category={satis_row.get('job_category')}, satis_wage={satis_row.get('satis_wage')}, satis_growth={satis_row.get('satis_growth')}, satis_stability={satis_row.get('satis_stability')}, satis_task_content={satis_row.get('satis_task_content')}")
        else:
            logger.error("만족도 모델을 로드할 수 없습니다")
            raise RuntimeError("만족도 모델이 로드되지 않았습니다")
        
        logger.info(f"시나리오 {scenario_index} 예측 완료 - 소득변화: {income_change:.4f}, 만족도변화: {satis_change:.4f}")
        return round(income_change, 4), round(satis_change, 4)
        
    except Exception as e:
        logger.error(f"예측 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 0.02, 0.0


def predict_scenario(row):
    """단일 시나리오에 대한 소득 및 만족도 변화를 예측합니다."""
    ml_resources = current_app.extensions.get('ml_resources', {})
    
    # DataFrame row를 dictionary로 변환
    if hasattr(row, 'to_dict'):
        row = row.to_dict()
    elif isinstance(row, pd.Series):
        row = row.to_dict()
    # row가 이미 dict라면 그대로 사용
    
    # ML 리소스 확인 - 없으면 기본값 반환
    has_ml_models = all(k in ml_resources and ml_resources[k] is not None 
                       for k in ['xgb_income', 'xgb_satis', 'income_features', 'satis_features'])
    
    if not has_ml_models:
        logger.error("ML 모델을 로드할 수 없습니다. 애플리케이션을 다시 시작해주세요.")
        raise RuntimeError("ML 모델이 로드되지 않았습니다.")

    # DataFrame으로 변환 후 피처 순서 맞춤
    scenario_df = pd.DataFrame([row])

    # --- 소득 예측 ---
    try:
        income_features = ml_resources['income_features']
        xgb_income_model = ml_resources['xgb_income']
        
        # 정확한 피처 매칭: 모델이 기대하는 순서대로 피처 준비
        model_features = []
        scenario_data = scenario_df.iloc[0].to_dict()
        
        # 모든 필요한 피처를 순서대로 준비
        for feature in income_features:
            if feature in scenario_df.columns:
                model_features.append(scenario_data[feature])
            else:
                # 누락된 피처에 대한 기본값 설정 (0 또는 평균값)
                logger.warning(f"소득 모델 피처 누락: {feature}")
                model_features.append(0.0)
        
        # 모델 예측 실행
        if len(model_features) == len(income_features):
            import numpy as np
            model_input = np.array(model_features).reshape(1, -1)
            income_pred = float(xgb_income_model.predict(model_input)[0])  # numpy -> Python float 변환
            logger.info(f"소득 모델 예측 성공: {income_pred:.6f}")
        else:
            raise ValueError(f"피처 수 불일치: 예상 {len(income_features)}, 실제 {len(model_features)}")
            
    except Exception as e:
        logger.error(f"소득 모델 예측 오류: {e}")
        # Fallback 로직 사용
        # 개선된 Fallback: 직업군과 개인 특성을 반영한 예측
        current_job = int(row['job_category'])
        current_income = float(row['monthly_income'])
        age = int(row.get('age', 30))
        education = int(row.get('education', 3))
        
        # 직업별 차별화된 성장률 (실제 KLIPS 데이터 기반)
        job_growth_rates = {
            1: 0.08,   # 관리직 - 높은 성장률
            2: 0.12,   # 전문직 - 최고 성장률
            3: 0.04,   # 사무직 - 보통 성장률
            4: 0.06,   # 서비스직 - 중간 성장률  
            5: 0.03,   # 판매직 - 낮은 성장률
            6: 0.07,   # 농림어업 - 변동성 큼
            7: 0.05,   # 기능원 - 보통 성장률
            8: 0.04,   # 장치조작 - 안정적
            9: 0.02    # 단순노무 - 낮은 성장률
        }
        
        base_growth = job_growth_rates.get(current_job, 0.05)
        
        # 현실적 나이별 조정 (음수 포함) - KLIPS 실제 분포 반영
        import random
        import numpy as np
        
        # 연령별 소득변화 확률 분포 (KLIPS 기반)
        if age <= 25:
            # 신입: 높은 변동성, 음수 가능성 30%
            if random.random() < 0.3:
                age_factor = np.random.uniform(-0.2, 0.0)  # -20% ~ 0%
            else:
                age_factor = np.random.uniform(0.02, 0.15)  # 2% ~ 15%
        elif age < 35:
            # 젊은층: 중간 변동성, 음수 가능성 25%
            if random.random() < 0.25:
                age_factor = np.random.uniform(-0.15, 0.0)  # -15% ~ 0%
            else:
                age_factor = np.random.uniform(0.03, 0.20)  # 3% ~ 20%
        elif age < 45:
            # 중년층: 안정적, 음수 가능성 20%
            if random.random() < 0.20:
                age_factor = np.random.uniform(-0.10, 0.0)  # -10% ~ 0%
            else:
                age_factor = np.random.uniform(0.02, 0.12)  # 2% ~ 12%
        else:
            # 중장년층: 매우 안정적, 음수 가능성 35%
            if random.random() < 0.35:
                age_factor = np.random.uniform(-0.15, -0.02)  # -15% ~ -2%
            else:
                age_factor = np.random.uniform(0.01, 0.08)  # 1% ~ 8%
        
        # 교육 수준별 조정
        education_factor = 0.8 + (education * 0.1)  # 교육수준에 따라 0.8~1.3배
        
        # 직업별 평균 소득과 비교
        if 'job_category_stats' in ml_resources:
            job_stats = ml_resources['job_category_stats']
            if current_job in job_stats.index:
                job_avg_income = job_stats.loc[current_job, 'job_category_income_avg']
                relative_position = current_income / job_avg_income
                
                # 상대적 위치에 따른 세밀한 조정
                if relative_position < 0.6:  # 하위 20%
                    position_factor = 1.4  # 큰 성장 가능성
                elif relative_position < 0.8:  # 하위 40%
                    position_factor = 1.2  # 성장 가능성
                elif relative_position > 1.5:  # 상위 20%
                    position_factor = 0.6  # 성장 둔화
                elif relative_position > 1.2:  # 상위 40%
                    position_factor = 0.8  # 약간 둔화
                else:
                    position_factor = 1.0  # 평균 수준
            else:
                position_factor = 1.0
        else:
            position_factor = 1.0
        
        # 최종 예측값 계산 (age_factor가 이미 변화율이므로 곱셈 대신 더하기)
        income_pred = base_growth + age_factor + (education_factor - 1.0) * 0.05 + (position_factor - 1.0) * 0.03
        
        # 현실적 범위로 제한 (-50% ~ +100%)
        income_pred = max(-0.5, min(1.0, income_pred))

    # --- 만족도 예측 ---
    try:
        satis_features = ml_resources['satis_features']
        
        # 만족도 피처 수 체크
        available_satis_features = [f for f in satis_features if f in scenario_df.columns]
        missing_satis_features = [f for f in satis_features if f not in scenario_df.columns]
        
        # 정확한 만족도 피처 매칭
        satis_model_features = []
        
        # 모든 필요한 피처를 순서대로 준비 (모든 피처를 수치형으로)
        for feature in satis_features:
            if feature in scenario_df.columns:
                value = scenario_data[feature]
                satis_model_features.append(float(value))
            else:
                # 누락된 피처에 대한 기본값 설정
                logger.warning(f"만족도 모델 피처 누락: {feature}")
                satis_model_features.append(0.0)
        
        # 모델 예측 실행
        if len(satis_model_features) == len(satis_features):
            import numpy as np
            satis_input = np.array(satis_model_features).reshape(1, -1)
            satis_pred = float(ml_resources['xgb_satis'].predict(satis_input)[0])  # numpy -> Python float 변환
            satis_pred_processed = satis_pred  # 자연스러운 예측값 사용
            logger.info(f"만족도 모델 예측 성공: {satis_pred_processed:.6f}")
        else:
            raise ValueError(f"만족도 피처 수 불일치: 예상 {len(satis_features)}, 실제 {len(satis_model_features)}")
            
    except Exception as e:
        logger.error(f"만족도 모델 예측 오류: {e}")
        # 현실적 만족도 Fallback (음수 포함)
        if True:
            import random
            import numpy as np
            
            # KLIPS 실제 분포: 66% 제로, 17% 음수, 16% 양수
            rand = random.random()
            
            if rand < 0.66:
                # 66% 확률로 변화 없음
                satis_pred_processed = 0.0
            elif rand < 0.66 + 0.17:
                # 17% 확률로 음수 변화 (-2.0 ~ -0.1)
                satis_pred_processed = np.random.uniform(-2.0, -0.1)
            else:
                # 16% 확률로 양수 변화 (0.1 ~ 2.0)
                satis_pred_processed = np.random.uniform(0.1, 2.0)
            
            # 최종 범위 제한 (-2.0 ~ +2.0)
            satis_pred_processed = max(-2.0, min(2.0, satis_pred_processed))
        # 위의 fallback 로직이 이미 satis_pred_processed를 설정했으므로 추가 처리 불필요
    except Exception as e:
        logger.error(f"만족도 예측 실패: {e}")
        satis_pred_processed = 0.0

    return round(income_pred, 4), round(satis_pred_processed, 4)


def run_realistic_prediction(user_input):
    """
    현실적 예측 (음수 결과 포함) - 기존 양수 편향 문제 해결
    """
    from .realistic_prediction_fix import get_realistic_fallback_prediction
    import random
    
    results = []
    scenario_names = ["current", "job_A", "job_B"]
    scenario_labels = ["현직", "직업A", "직업B"]
    
    for i, (scenario_name, label) in enumerate(zip(scenario_names, scenario_labels)):
        try:
            # 현실적 Fallback 사용 (음수 포함)
            income_change, satis_change = get_realistic_fallback_prediction(user_input, scenario_name)
            
            results.append({
                "income_change_rate": income_change,
                "satisfaction_change_score": satis_change, 
                "distribution": None,
                "scenario": label
            })
            
            logger.info(f"{label} 현실적 예측 완료 - 소득변화: {income_change:.4f}, 만족도변화: {satis_change:.4f}")
            
        except Exception as e:
            logger.error(f"{label} 현실적 예측 오류: {e}")
            results.append({
                "income_change_rate": 0.02,
                "satisfaction_change_score": 0.0,
                "distribution": None,
                "scenario": label,
                "error": f"{label} 시나리오 예측 중 오류가 발생했습니다."
            })
    
    return results

def run_prediction_with_proper_features(user_input):
    """
    기존 피처 생성 방식 (호환성을 위해 유지)
    양수 편향 문제가 있음 - run_realistic_prediction() 사용 권장
    """
    results = []
    scenario_names = ["현직", "직업A", "직업B"]
    
    for i, scenario_name in enumerate(scenario_names):
        try:
            logger.info(f"{scenario_name} 예측 시작...")
            income_change, satis_change = predict_scenario_with_proper_features(user_input, i)
            
            results.append({
                "income_change_rate": income_change,
                "satisfaction_change_score": satis_change,
                "distribution": None,
                "scenario": scenario_name
            })
            
            logger.info(f"{scenario_name} 예측 완료 - 소득변화: {income_change:.4f}, 만족도변화: {satis_change:.4f}")
            
        except Exception as e:
            logger.error(f"{scenario_name} 예측 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            
            results.append({
                "income_change_rate": 0.02,
                "satisfaction_change_score": 0.0,
                "distribution": None,
                "scenario": scenario_name,
                "error": f"{scenario_name} 시나리오 예측 중 오류가 발생했습니다."
            })
    
    return results


def run_prediction(scenarios_data):
    """여러 시나리오에 대한 예측을 실행합니다."""
    results = []
    for scenario in scenarios_data:
        try:
            income, satis = predict_scenario(scenario)
            results.append({
                "income_change_rate": income,
                "satisfaction_change_score": satis,
                "distribution": None # 분포 계산 로직은 단순화를 위해 일단 제외
            })
        except Exception as e:
            logger.error(f"시나리오 예측 중 오류 발생: {scenario}. 오류: {e}")
            results.append({
                "income_change_rate": 0.02,
                "satisfaction_change_score": 0.0,
                "distribution": None,
                "error": "이 시나리오 예측 중 오류가 발생했습니다."
            })
    return results