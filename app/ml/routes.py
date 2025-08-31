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
# 실제 KLIPS 데이터 기준으로 현실적 범위 조정
MIN_SATISFACTION_CHANGE = -4  # 실제 최소값
MAX_SATISFACTION_CHANGE = 3   # 실제 최대값

# 실제 KLIPS 데이터 기반 현실적 범위 (클리핑 제거를 위한 참고값)
# 실제 5-95분위수: -40% ~ +80%, 평균 12.69%
# 클리핑 대신 피처 엔지니어링으로 현실적 예측 유도

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
            # 소득 모델
            income_model_path = os.path.join(MODEL_DIR, "lgb_income_change_model.txt")
            logger.info(f"Loading income model from: {income_model_path}")
            ml_resources['lgb_income'] = lgb.Booster(model_file=income_model_path)
            logger.info("LightGBM income model loaded successfully")
            
            # 만족도 모델 (최적 단일 모델: CatBoost)
            satis_model_path = os.path.join(MODEL_DIR, "final_cat_satis_model.cbm")
            logger.info(f"Loading satisfaction model from: {satis_model_path}")
            cat_satis_model = CatBoostRegressor()
            cat_satis_model.load_model(satis_model_path)
            ml_resources['cat_satis'] = cat_satis_model
            logger.info("CatBoost satisfaction model loaded successfully")

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

            # 실제 모델에서 피처 이름을 직접 추출
            logger.info("Extracting feature names from loaded income model...")
            ml_resources['income_features'] = ml_resources['lgb_income'].feature_name()
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
                       for k in ['lgb_income', 'cat_satis', 'income_features', 'satis_features'])
    
    if not has_ml_models:
        logger.warning(f"ML models not available, using fallback")
        job_category = int(user_input.get("current_job_category" if scenario_index == 0 else 
                                         f"job_{'A' if scenario_index == 1 else 'B'}_category", 1))
        base_income_change = 0.02  # 2% 기본값
        base_satis_change = 0.0
        
        # 직업별 조정
        if job_category == 2:
            base_income_change = 0.025
            base_satis_change = 0.1
        elif job_category == 3:
            base_income_change = 0.015
            base_satis_change = 0.05
        
        return round(base_income_change, 4), round(base_satis_change, 4)
    
    try:
        # 소득 모델용 피처 생성
        income_df = prepare_income_model_features(user_input, temp_predictor)
        income_row = income_df.iloc[scenario_index].to_dict()
        
        # 만족도 모델용 피처 생성  
        satis_df = prepare_satisfaction_model_features(user_input, temp_predictor)
        satis_row = satis_df.iloc[scenario_index].to_dict()
        
        # --- 소득 예측 ---
        income_features = ml_resources['income_features']
        lgb_income_model = ml_resources['lgb_income']
        
        # 피처를 정확한 순서로 준비
        income_model_features = []
        for feature in income_features:
            if feature in income_row:
                income_model_features.append(income_row[feature])
            else:
                logger.warning(f"소득 모델 피처 누락: {feature}")
                income_model_features.append(0.0)
        
        # 소득 예측 실행 (클리핑 제거)
        income_prediction = lgb_income_model.predict([income_model_features])[0]
        income_change = income_prediction  # 자연스러운 예측값 사용
        
        logger.info(f"시나리오 {scenario_index} 소득 예측: 원본값={income_prediction:.6f}, 클리핑후={income_change:.6f}")
        logger.info(f"소득 주요 피처값: job_category={income_row.get('job_category')}, monthly_income={income_row.get('monthly_income')}, age={income_row.get('age')}, job_category_change={income_row.get('job_category_change', 0)}, potential_promotion={income_row.get('potential_promotion', 0)}")
        
        # --- 만족도 예측 ---
        satis_features = ml_resources['satis_features']
        
        # CatBoost 모델들
        xgb_satis = ml_resources.get('xgb_satis')
        lgb_satis = ml_resources.get('lgb_satis')  
        cat_satis = ml_resources.get('cat_satis')
        
        # 앙상블 가중치
        ensemble_config = ml_resources.get('ensemble_config', {})
        weights = ensemble_config.get('weights', {'xgb': 0.34, 'lgb': 0.31, 'cat': 0.34})
        
        # 피처를 정확한 순서로 준비
        satis_model_features = []
        for feature in satis_features:
            if feature in satis_row:
                value = satis_row[feature]
                # CatBoost categorical 피처를 위해 일부 피처는 문자열로 변환
                if feature in ['gender', 'education', 'job_category', 'career_stage']:
                    satis_model_features.append(str(int(float(value))))
                else:
                    satis_model_features.append(float(value))
            else:
                logger.warning(f"만족도 모델 피처 누락: {feature}")
                # categorical 피처면 문자열 "0", 수치 피처면 0.0
                if feature in ['gender', 'education', 'job_category', 'career_stage']:
                    satis_model_features.append("0")
                else:
                    satis_model_features.append(0.0)
        
        # 앙상블 예측
        predictions = []
        if xgb_satis:
            pred = xgb_satis.predict([satis_model_features])[0]
            predictions.append(('xgb', pred, weights['xgb']))
        if lgb_satis:
            pred = lgb_satis.predict([satis_model_features])[0] 
            predictions.append(('lgb', pred, weights['lgb']))
        if cat_satis:
            pred = cat_satis.predict([satis_model_features])[0]
            predictions.append(('cat', pred, weights['cat']))
        
        if predictions:
            satis_prediction = sum(pred * weight for _, pred, weight in predictions) / sum(w for _, _, w in predictions)
            satis_change = satis_prediction  # 자연스러운 예측값 사용
            
            logger.info(f"시나리오 {scenario_index} 만족도 예측: 원본값={satis_prediction:.6f}, 클리핑후={satis_change:.6f}")
            logger.info(f"앙상블 세부결과: {[(name, pred, weight) for name, pred, weight in predictions]}")
            logger.info(f"만족도 주요 피처값: job_category={satis_row.get('job_category')}, satis_wage={satis_row.get('satis_wage')}, satis_growth={satis_row.get('satis_growth')}, satis_stability={satis_row.get('satis_stability')}, satis_task_content={satis_row.get('satis_task_content')}")
        else:
            logger.warning("만족도 모델이 없어 기본값 사용")
            satis_change = 0.0
        
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
                       for k in ['lgb_income', 'cat_satis', 'income_features', 'satis_features'])
    
    if not has_ml_models:
        logger.warning(f"ML models not available, using fallback for scenario: {row.get('job_category', 'unknown')}")
        # 기본 예측값 반환 (직업별로 약간의 차이 적용)
        job_category = int(row.get('job_category', 1))
        base_income_change = 0.05
        base_satis_change = 0.0
        
        # 직업별 간단한 휴리스틱 적용
        if job_category == 2:  # IT 관련직
            base_income_change = 0.06
            base_satis_change = 0.1
        elif job_category == 3:  # 교육/연구
            base_income_change = 0.03
            base_satis_change = 0.05
        elif job_category >= 7:  # 서비스업 등
            base_income_change = 0.04
            base_satis_change = -0.05
            
        return round(base_income_change, 4), round(base_satis_change, 4)

    # DataFrame으로 변환 후 피처 순서 맞춤
    scenario_df = pd.DataFrame([row])

    # --- 소득 예측 ---
    try:
        income_features = ml_resources['income_features']
        lgb_income_model = ml_resources['lgb_income']
        
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
            income_pred = lgb_income_model.predict(model_input)[0]
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
        
        # 나이별 조정 (경력 곡선)
        if age < 30:
            age_factor = 1.3  # 젊은 층 높은 성장
        elif age < 45:
            age_factor = 1.1  # 중간층 약간 높음
        elif age < 55:
            age_factor = 0.9  # 중년층 둔화
        else:
            age_factor = 0.7  # 고령층 낮은 성장
        
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
        
        # 최종 예측값 계산
        income_pred = base_growth * age_factor * education_factor * position_factor
        
        # 자연스러운 예측값 사용 (클리핑 제거)

    # --- 만족도 예측 ---
    try:
        satis_features = ml_resources['satis_features']
        
        # 만족도 피처 수 체크
        available_satis_features = [f for f in satis_features if f in scenario_df.columns]
        missing_satis_features = [f for f in satis_features if f not in scenario_df.columns]
        
        # 정확한 만족도 피처 매칭
        satis_model_features = []
        
        # 모든 필요한 피처를 순서대로 준비
        for feature in satis_features:
            if feature in scenario_df.columns:
                value = scenario_data[feature]
                # CatBoost categorical 피처를 위해 일부 피처는 문자열로 변환
                if feature in ['gender', 'education', 'job_category', 'career_stage']:
                    satis_model_features.append(str(int(float(value))))
                else:
                    satis_model_features.append(float(value))
            else:
                # 누락된 피처에 대한 기본값 설정
                logger.warning(f"만족도 모델 피처 누락: {feature}")
                if feature in ['gender', 'education', 'job_category', 'career_stage']:
                    satis_model_features.append("0")
                else:
                    satis_model_features.append(0.0)
        
        # 모델 예측 실행
        if len(satis_model_features) == len(satis_features):
            import numpy as np
            satis_input = np.array(satis_model_features).reshape(1, -1)
            satis_pred = ml_resources['cat_satis'].predict(satis_input)[0]
            satis_pred_processed = satis_pred  # 자연스러운 예측값 사용
            logger.info(f"만족도 모델 예측 성공: {satis_pred_processed:.6f}")
        else:
            raise ValueError(f"만족도 피처 수 불일치: 예상 {len(satis_features)}, 실제 {len(satis_model_features)}")
            
    except Exception as e:
        logger.error(f"만족도 모델 예측 오류: {e}")
        # Fallback 로직 사용
        if True:
            # 개선된 만족도 fallback: 9개 만족도 요인 기반 예측
            current_job = int(row['job_category'])
            current_satisfaction = int(row.get('job_satisfaction', 3))
            
            # 9개 만족도 요인 점수 수집
            satis_factors = [
                int(row.get('satis_wage', 3)),
                int(row.get('satis_stability', 3)),
                int(row.get('satis_growth', 3)),
                int(row.get('satis_task_content', 3)),
                int(row.get('satis_work_env', 3)),
                int(row.get('satis_work_time', 3)),
                int(row.get('satis_communication', 3)),
                int(row.get('satis_fair_eval', 3)),
                int(row.get('satis_welfare', 3))
            ]
            
            # 평균 만족도와 현재 만족도 비교
            avg_factor_score = sum(satis_factors) / len(satis_factors)
            satisfaction_gap = avg_factor_score - current_satisfaction
            
            # 직업군별 만족도 변화 경향 (실제 데이터 기반)
            job_satis_tendency = {
                1: 0.1,    # 관리직 - 약간 상승
                2: 0.2,    # 전문직 - 상승 경향
                3: -0.1,   # 사무직 - 약간 하락
                4: -0.2,   # 서비스직 - 하락 경향
                5: -0.1,   # 판매직 - 약간 하락
                6: 0.0,    # 농림어업 - 변화 적음
                7: 0.0,    # 기능원 - 안정적
                8: -0.1,   # 장치조작 - 약간 하락
                9: -0.3    # 단순노무 - 하락 경향
            }
            
            base_change = job_satis_tendency.get(current_job, 0.0)
            
            # 만족도 갭에 따른 조정
            gap_adjustment = satisfaction_gap * 0.3  # 갭의 30% 반영
            
            satis_pred_processed = base_change + gap_adjustment
        else:
            # 정확한 피처 순서로 선택
            satis_df = scenario_df[satis_features].copy()
            logger.info(f"만족도 모델 피처 개수: {len(satis_df.columns)}, 기대: {len(satis_features)}")
            logger.info(f"만족도 모델 입력 피처 통계: min={satis_df.min().min():.4f}, max={satis_df.max().max():.4f}")
            
            satis_pred = ml_resources['cat_satis'].predict(satis_df)
            logger.info(f"만족도 모델 원본 예측값: {satis_pred[0]:.4f}")
            
            satis_pred_processed = np.clip(satis_pred, MIN_SATISFACTION_CHANGE, MAX_SATISFACTION_CHANGE)[0]
            logger.info(f"만족도 모델 최종 예측값: {satis_pred_processed:.4f}")
    except Exception as e:
        logger.error(f"만족도 예측 실패: {e}")
        satis_pred_processed = 0.0

    return round(income_pred, 4), round(satis_pred_processed, 4)


def run_prediction_with_proper_features(user_input):
    """
    새로운 피처 생성 방식을 사용하여 3개 시나리오 예측을 실행합니다.
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