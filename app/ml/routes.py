from flask import current_app, Blueprint
import sys, json, os, logging, random
import joblib
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
from typing import Dict, Tuple

# Blueprint 생성
bp = Blueprint('ml', __name__)
logger = logging.getLogger(__name__)

# realistic_prediction_fix.py에서 가져온 상수 및 헬퍼 함수
REAL_INCOME_DISTRIBUTION = {
    'mean': 0.126943, 'std': 1.060963, 'negative_ratio': 0.2795,
    'percentiles': {5: -0.4, 25: -0.048, 75: 0.2, 95: 0.8}
}
REAL_SATISFACTION_DISTRIBUTION = {
    'mean': -0.008254, 'std': 0.629151, 'negative_ratio': 0.1735,
    'zero_ratio': 0.6605, 'positive_ratio': 0.1660
}
AGE_RISK_FACTORS = {
    (18, 25): {'income_volatility': 1.3, 'success_bonus': 0.1},
    (26, 35): {'income_volatility': 1.1, 'success_bonus': 0.2},
    (36, 45): {'income_volatility': 0.9, 'success_bonus': 0.0},
    (46, 55): {'income_volatility': 0.7, 'success_bonus': -0.1},
    (56, 100): {'income_volatility': 0.5, 'success_bonus': -0.2}
}

def get_age_risk_factor(age: int) -> Dict:
    for (min_age, max_age), factors in AGE_RISK_FACTORS.items():
        if min_age <= age <= max_age:
            return factors
    return AGE_RISK_FACTORS[(26, 35)]

def get_realistic_fallback_prediction(user_profile: Dict, scenario_type: str) -> Tuple[float, float]:
    age = int(user_profile['age'])
    current_job = int(user_profile['current_job_category'])
    
    if scenario_type == 'current':
        income_change = np.random.normal(0.05, 0.15)
        satisfaction_change = np.random.choice([0.0, -0.1, 0.1], p=[0.8, 0.1, 0.1])
    else:
        target_job = int(user_profile['job_A_category'] if scenario_type == 'jobA' else user_profile['job_B_category'])
        
        # 직업별 기본 소득 변화 프로필 (fallback용)
        job_income_profiles = {
            1: {'base': 0.18, 'std': 0.12},  # 관리자 - 높은 소득 증가 가능성
            2: {'base': 0.25, 'std': 0.15},  # 전문가 - 최고 소득 증가 가능성
            3: {'base': 0.08, 'std': 0.10},  # 사무직 - 안정적이지만 낮은 증가
            4: {'base': -0.05, 'std': 0.20}, # 서비스직 - 소득 감소 위험
            5: {'base': 0.12, 'std': 0.25}   # 판매직 - 높은 변동성
        }
        
        job_profile = job_income_profiles.get(target_job, {'base': 0.05, 'std': 0.15})
        income_change = np.random.normal(job_profile['base'], job_profile['std'])
        
        # 직업별 만족도 변화 프로필
        job_satis_profiles = {
            1: {'positive_prob': 0.65, 'base': 0.3},  # 관리자 - 높은 만족도
            2: {'positive_prob': 0.70, 'base': 0.4},  # 전문가 - 최고 만족도
            3: {'positive_prob': 0.55, 'base': 0.2},  # 사무직 - 보통 만족도
            4: {'positive_prob': 0.45, 'base': 0.1},  # 서비스직 - 낮은 만족도
            5: {'positive_prob': 0.50, 'base': 0.15}  # 판매직 - 변동적 만족도
        }
        
        satis_profile = job_satis_profiles.get(target_job, {'positive_prob': 0.5, 'base': 0.2})
        if random.random() < satis_profile['positive_prob']:
            satisfaction_change = np.random.uniform(0.1, satis_profile['base'] + 1.0)
        else:
            satisfaction_change = np.random.uniform(-1.5, -0.1)
            
    return round(income_change, 4), round(satisfaction_change, 4)

def generate_distribution_data(user_input, scenario_type, income_change, satis_change):
    base_income_std, base_satis_std = 0.15, 0.8
    income_std = base_income_std * (0.8 if scenario_type == "current" else 1.3)
    satis_std = base_satis_std * (0.7 if scenario_type == "current" else 1.1)
    n_samples = random.randint(80, 150)
    income_samples = np.clip(np.random.normal(income_change, income_std, n_samples), -0.5, 1.0)
    satis_samples = np.clip(np.random.normal(satis_change, satis_std, n_samples), -2.5, 2.5)
    
    def create_histogram(data, n_bins=8):
        hist, bin_edges = np.histogram(data, bins=n_bins)
        return hist.tolist(), bin_edges.tolist()
        
    income_counts, income_bins = create_histogram(income_samples)
    satis_counts, satis_bins = create_histogram(satis_samples)
    return {"income": {"counts": income_counts, "bins": income_bins}, "satisfaction": {"counts": satis_counts, "bins": satis_bins}}

def init_app(app):
    with app.app_context():
        APP_ROOT = current_app.root_path
        PROJECT_ROOT = os.path.dirname(APP_ROOT) if os.path.basename(APP_ROOT) == 'app' else APP_ROOT
        MODEL_DIR = os.path.join(PROJECT_ROOT, "app", "ml", "saved_models")
        DATA_PATH = os.path.join(PROJECT_ROOT, "data", "klips_data_23.csv")
        
        ml_resources = {}
        try:
            logger.info("ML 모델 및 데이터 사전 로딩 시작...")
            ml_resources['lgb_income'] = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lgb_income_change_model.txt"))
            with open(os.path.join(MODEL_DIR, "final_xgb_satis_model.pkl"), 'rb') as f:
                ml_resources['xgb_satis'] = joblib.load(f)
            ml_resources['lgb_satis'] = lgb.Booster(model_file=os.path.join(MODEL_DIR, "final_lgb_satis_model.txt"))
            ml_resources['cat_satis'] = CatBoostRegressor().load_model(os.path.join(MODEL_DIR, "final_cat_satis_model.cbm"))
            ml_resources['klips_df'] = pd.read_csv(DATA_PATH)
            ml_resources['job_category_stats'] = ml_resources['klips_df'].groupby('job_category').agg(
                {'monthly_income': 'mean', 'education': 'mean', 'job_satisfaction': 'mean'}
            ).rename(columns={'monthly_income': 'job_category_income_avg', 'education': 'job_category_education_avg', 'job_satisfaction': 'job_category_satisfaction_avg'})
            with open(os.path.join(MODEL_DIR, "final_ensemble_satis_config.json"), 'r') as f:
                ml_resources['ensemble_config'] = json.load(f)
                ml_resources['satis_features'] = ml_resources['ensemble_config']['features']
            with open(os.path.join(MODEL_DIR, "income_feature_names_correct.json"), 'r') as f:
                ml_resources['income_features'] = json.load(f)
            app.extensions['ml_resources'] = ml_resources
            logger.info("ML 리소스 사전 로딩 완료.")
        except Exception as e:
            logger.critical(f"ML 리소스 로딩 실패: {e}", exc_info=True)
            app.extensions['ml_resources'] = {}

@bp.route('/predict', methods=['POST'])
def run_prediction(user_input):
    from app.ml.preprocessing import prepare_income_model_features, prepare_satisfaction_model_features

    def _get_change_class(value):
        if not isinstance(value, (int, float)): return 'no-change'
        if value > 0.001: return 'positive-change'
        if value < -0.001: return 'negative-change'
        return 'no-change'

    results = []
    scenario_names = ["현직", "직업A", "직업B"]
    scenario_types = ["current", "jobA", "jobB"]
    
    try:
        ml_resources = current_app.extensions.get('ml_resources', {})
        if not ml_resources:
            raise RuntimeError("ML 리소스가 로드되지 않았습니다.")

        class TempPredictor:
            def __init__(self):
                self.job_category_stats = ml_resources.get('job_category_stats')
        temp_predictor = TempPredictor()

        income_df = prepare_income_model_features(user_input, temp_predictor)
        satis_df = prepare_satisfaction_model_features(user_input, temp_predictor)
        
        # 디버깅: 피처 데이터 로깅
        logger.info(f"소득 모델 입력 피처 (첫 3행):")
        for i, row in income_df.head(3).iterrows():
            logger.info(f"  시나리오 {i}: job_category={row.get('job_category', 'N/A')}, "
                       f"monthly_income={row.get('monthly_income', 'N/A')}, "
                       f"job_category_change={row.get('job_category_change', 'N/A')}, "
                       f"income_vs_peers={row.get('income_vs_peers', 'N/A')}")
            logger.info(f"    education_roi={row.get('education_roi', 'N/A')}, "
                       f"income_age_ratio={row.get('income_age_ratio', 'N/A')}, "
                       f"potential_promotion={row.get('potential_promotion', 'N/A')}")
        
        income_features = ml_resources['income_features']
        satis_features = ml_resources['satis_features']
        lgb_income_model = ml_resources['lgb_income']
        ensemble_models = {
            'xgb': ml_resources.get('xgb_satis'),
            'lgb': ml_resources.get('lgb_satis'),
            'cat': ml_resources.get('cat_satis')
        }
        weights = ml_resources.get('ensemble_config', {}).get('weights', {'xgb': 0.34, 'lgb': 0.31, 'cat': 0.34})

        for i, (scenario_name, scenario_type) in enumerate(zip(scenario_names, scenario_types)):
            income_row = income_df.iloc[i].to_dict()
            satis_row = satis_df.iloc[i].to_dict()

            income_model_features = [income_row.get(f, 0.0) for f in income_features]
            income_input = np.array(income_model_features).reshape(1, -1)
            base_income_change = float(lgb_income_model.predict(income_input)[0])
            
            # 직업별 소득 변화율 보정
            current_job = int(user_input["current_job_category"])
            target_job = income_row.get('job_category', current_job)
            
            if target_job != current_job:  # 이직 시나리오
                # 직업별 기본 소득 프리미엄/디스카운트 적용
                job_income_multiplier = {
                    1: 1.20,  # 관리자 - 20% 소득 프리미엄
                    2: 1.35,  # 전문가 - 35% 소득 프리미엄 (최고)
                    3: 1.05,  # 사무직 - 5% 소득 프리미엄 (안정적)
                    4: 0.90,  # 서비스직 - 10% 소득 감소 위험
                    5: 1.15   # 판매직 - 15% 소득 변동성 (성과 기반)
                }.get(target_job, 1.0)
                
                # 현재 직업에서의 이직 난이도 보정
                job_transition_difficulty = {
                    1: 0.15,  # 관리자로 이직 - 어려움 (소득 증가 제한)
                    2: 0.25,  # 전문가로 이직 - 매우 어려움 (큰 소득 증가 가능)
                    3: 0.05,  # 사무직으로 이직 - 쉬움 (소득 변화 적음)
                    4: -0.10, # 서비스직으로 이직 - 소득 감소 위험
                    5: 0.10   # 판매직으로 이직 - 변동성 큼
                }.get(target_job, 0.0)
                
                income_change = base_income_change * job_income_multiplier + job_transition_difficulty
            else:
                # 현직 유지 - 기본 예측값 사용 (소폭 보정)
                income_change = base_income_change * 1.02  # 2% 안정성 보너스
            
            # 현실적 범위로 제한
            income_change = max(-0.50, min(1.00, income_change))
            
            # 디버깅: 예측 결과 로깅
            logger.info(f"{scenario_name} 시나리오: 기본예측={base_income_change:.4f}, "
                       f"최종예측={income_change:.4f}, 대상직업={target_job}")

            satis_model_features = [float(satis_row.get(f, 0.0)) for f in satis_features]
            satis_input = np.array(satis_model_features).reshape(1, -1)
            
            predictions = []
            for name, model in ensemble_models.items():
                if model:
                    try:
                        if name == 'cat':
                            # CatBoost용 데이터프레임 생성
                            cat_df = pd.DataFrame([satis_model_features], columns=satis_features)
                            
                            # 카테고리형 피처들을 정수로 변환 
                            categorical_features = ['age', 'gender', 'education', 'job_category', 'career_stage']
                            for cat_col in categorical_features:
                                if cat_col in cat_df.columns:
                                    cat_df[cat_col] = cat_df[cat_col].astype(int)
                            
                            pred = float(model.predict(cat_df)[0])
                        else:
                            pred = float(model.predict(satis_input)[0])
                        predictions.append((pred, weights.get(name, 0.33)))
                    except Exception as e:
                        logger.warning(f"{name} 만족도 모델 예측 실패: {e}")
            
            if predictions:
                weighted_sum = sum(pred * w for pred, w in predictions)
                total_weight = sum(w for _, w in predictions)
                satis_change = weighted_sum / total_weight if total_weight > 0 else 0.0
            else:
                satis_change = 0.0

            distribution = generate_distribution_data(user_input, scenario_type, income_change, satis_change)
            
            results.append({
                "income_change_rate": round(income_change, 4),
                "satisfaction_change_score": round(satis_change, 4),
                "income_class": _get_change_class(income_change),
                "satisfaction_class": _get_change_class(satis_change),
                "distribution": distribution,
                "scenario": scenario_name
            })

    except Exception as e:
        logger.error(f"모델 예측 실패: {e}", exc_info=True)
        # 모델 예측 실패 시 현실적인 Fallback 로직 사용
        results = []
        for scenario_name, scenario_type in zip(scenario_names, scenario_types):
            income_change, satis_change = get_realistic_fallback_prediction(user_input, scenario_type)
            distribution = generate_distribution_data(user_input, scenario_type, income_change, satis_change)
            results.append({
                "income_change_rate": income_change,
                "satisfaction_change_score": satis_change,
                "income_class": _get_change_class(income_change),
                "satisfaction_class": _get_change_class(satis_change),
                "distribution": distribution,
                "scenario": scenario_name,
                "error": "모델 예측에 실패하여 현실적 추정치를 제공합니다."
            })
        
    return results
