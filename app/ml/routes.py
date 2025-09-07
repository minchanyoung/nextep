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
    age = user_profile['age']
    current_job = user_profile['current_job_category']
    if scenario_type == 'current':
        income_change = np.random.normal(0.05, 0.15)
        satisfaction_change = np.random.choice([0.0, -0.1, 0.1], p=[0.8, 0.1, 0.1])
    else:
        target_job = user_profile['job_A_category'] if scenario_type == 'job_A' else user_profile['job_B_category']
        # (구현 간소화) realistic_prediction_fix의 복잡한 로직 대신 기본 분포 사용
        if random.random() < REAL_INCOME_DISTRIBUTION['negative_ratio']:
            income_change = np.random.uniform(REAL_INCOME_DISTRIBUTION['percentiles'][5], REAL_INCOME_DISTRIBUTION['percentiles'][25])
        else:
            income_change = np.random.uniform(REAL_INCOME_DISTRIBUTION['percentiles'][25], REAL_INCOME_DISTRIBUTION['percentiles'][95])
        
        rand_satis = random.random()
        if rand_satis < REAL_SATISFACTION_DISTRIBUTION['zero_ratio']:
            satisfaction_change = 0.0
        elif rand_satis < REAL_SATISFACTION_DISTRIBUTION['zero_ratio'] + REAL_SATISFACTION_DISTRIBUTION['negative_ratio']:
            satisfaction_change = np.random.uniform(-2.0, -0.1)
        else:
            satisfaction_change = np.random.uniform(0.1, 2.0)
            
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
            income_change = float(lgb_income_model.predict(income_input)[0])

            satis_model_features = [float(satis_row.get(f, 0.0)) for f in satis_features]
            satis_input = np.array(satis_model_features).reshape(1, -1)
            
            predictions = []
            for name, model in ensemble_models.items():
                if model:
                    try:
                        if name == 'cat':
                            pred = float(model.predict(pd.DataFrame([satis_model_features], columns=satis_features))[0])
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
                satis_change = 0.0 # 모든 만족도 모델 실패 시

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
