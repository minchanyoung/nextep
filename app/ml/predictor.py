# app/ml/predictor.py
import os
import json
import joblib
import pandas as pd
import numpy as np
import logging
import random
from typing import Dict, List, Tuple
from flask import current_app
from catboost import CatBoostRegressor
import lightgbm as lgb

from app.ml.preprocessing import prepare_income_model_features, prepare_satisfaction_model_features
from app.constants import JOB_CATEGORY_MAP

logger = logging.getLogger(__name__)

# 현실적 예측을 위한 상수 및 헬퍼 함수 (routes.py에서 이동)
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

def _get_age_risk_factor(age: int) -> Dict:
    for (min_age, max_age), factors in AGE_RISK_FACTORS.items():
        if min_age <= age <= max_age:
            return factors
    return AGE_RISK_FACTORS[(26, 35)]

def _get_realistic_fallback_prediction(user_profile: Dict, scenario_type: str) -> Tuple[float, float]:
    # ... (이하 routes.py의 내용과 동일)
    age = int(user_profile['age'])
    if scenario_type == 'current':
        income_change = np.random.normal(0.05, 0.15)
        satisfaction_change = np.random.choice([0.0, -0.1, 0.1], p=[0.8, 0.1, 0.1])
    else:
        target_job = int(scenario_type) # job_code가 직접 넘어옴
        job_income_profiles = {
            1: {'base': 0.18, 'std': 0.12}, 2: {'base': 0.25, 'std': 0.15},
            3: {'base': 0.08, 'std': 0.10}, 4: {'base': -0.05, 'std': 0.20},
            5: {'base': 0.12, 'std': 0.25}
        }
        job_profile = job_income_profiles.get(target_job, {'base': 0.05, 'std': 0.15})
        income_change = np.random.normal(job_profile['base'], job_profile['std'])
        
        job_satis_profiles = {
            1: {'positive_prob': 0.65, 'base': 0.3}, 2: {'positive_prob': 0.70, 'base': 0.4},
            3: {'positive_prob': 0.55, 'base': 0.2}, 4: {'positive_prob': 0.45, 'base': 0.1},
            5: {'positive_prob': 0.50, 'base': 0.15}
        }
        satis_profile = job_satis_profiles.get(target_job, {'positive_prob': 0.5, 'base': 0.2})
        if random.random() < satis_profile['positive_prob']:
            satisfaction_change = np.random.uniform(0.1, satis_profile['base'] + 1.0)
        else:
            satisfaction_change = np.random.uniform(-1.5, -0.1)
            
    return round(income_change, 4), round(satisfaction_change, 4)

def _generate_distribution_data(user_input, scenario_type, income_change, satis_change):
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

def _get_change_class(value):
    if not isinstance(value, (int, float)): return 'no-change'
    if value > 0.001: return 'positive-change'
    if value < -0.001: return 'negative-change'
    return 'no-change'


class MLPredictor:
    def __init__(self, app):
        APP_ROOT = app.root_path
        PROJECT_ROOT = os.path.dirname(APP_ROOT) if os.path.basename(APP_ROOT) == 'app' else APP_ROOT
        MODEL_DIR = os.path.join(PROJECT_ROOT, "app", "ml", "saved_models")
        DATA_PATH = os.path.join(PROJECT_ROOT, "data", "klips_data_23.csv")
        
        self.models = {}
        self.features = {}
        self.job_category_stats = None
        
        try:
            logger.info("ML 모델 및 데이터 사전 로딩 시작...")
            self.models['lgb_income'] = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lgb_income_change_model.txt"))
            with open(os.path.join(MODEL_DIR, "final_xgb_satis_model.pkl"), 'rb') as f:
                self.models['xgb_satis'] = joblib.load(f)
            self.models['lgb_satis'] = lgb.Booster(model_file=os.path.join(MODEL_DIR, "final_lgb_satis_model.txt"))
            self.models['cat_satis'] = CatBoostRegressor().load_model(os.path.join(MODEL_DIR, "final_cat_satis_model.cbm"))
            
            klips_df = pd.read_csv(DATA_PATH)
            self.job_category_stats = klips_df.groupby('job_category').agg(
                {'monthly_income': 'mean', 'education': 'mean', 'job_satisfaction': 'mean'}
            ).rename(columns={'monthly_income': 'job_category_income_avg', 'education': 'job_category_education_avg', 'job_satisfaction': 'job_category_satisfaction_avg'})
            
            with open(os.path.join(MODEL_DIR, "final_ensemble_satis_config.json"), 'r') as f:
                ensemble_config = json.load(f)
                self.features['satis'] = ensemble_config['features']
                self.models['ensemble_weights'] = ensemble_config.get('weights', {'xgb': 0.34, 'lgb': 0.31, 'cat': 0.34})

            with open(os.path.join(MODEL_DIR, "income_feature_names_correct.json"), 'r') as f:
                self.features['income'] = json.load(f)
            logger.info("ML 리소스 사전 로딩 완료.")
        except Exception as e:
            logger.critical(f"ML 리소스 로딩 실패: {e}", exc_info=True)
            raise RuntimeError("ML 리소스 로딩에 실패하여 예측 서비스를 시작할 수 없습니다.")

    def predict(self, user_input: Dict, scenarios_to_run: List[str] = None) -> Dict:
        results = {}
        
        if scenarios_to_run:
            scenario_codes = ['current'] + list(scenarios_to_run)
            scenario_map = {'current': '현직 유지', **{code: JOB_CATEGORY_MAP.get(str(code), f"직업 {code}") for code in scenarios_to_run}}
        else:
            scenario_map = {'current': '현직 유지', 'jobA': '직업A', 'jobB': '직업B'}
            scenario_codes = list(scenario_map.keys())

        try:
            income_df = prepare_income_model_features(user_input, self, scenario_codes)
            satis_df = prepare_satisfaction_model_features(user_input, self, scenario_codes)
            
            income_features = self.features['income']
            satis_features = self.features['satis']
            lgb_income_model = self.models['lgb_income']
            ensemble_models = {k: self.models[k] for k in ['xgb_satis', 'lgb_satis', 'cat_satis']}
            weights = self.models['ensemble_weights']

            for i, scenario_code in enumerate(scenario_codes):
                income_row = income_df.iloc[i].to_dict()
                satis_row = satis_df.iloc[i].to_dict()

                # 소득 예측
                income_model_features = [income_row.get(f, 0.0) for f in income_features]
                income_input = np.array(income_model_features).reshape(1, -1)
                base_income_change = float(lgb_income_model.predict(income_input)[0])
                
                current_job = int(user_input["current_job_category"])
                target_job = income_row.get('job_category', current_job)
                
                if target_job != current_job:
                    multiplier = {1: 1.20, 2: 1.35, 3: 1.05, 4: 0.90, 5: 1.15}.get(target_job, 1.0)
                    difficulty = {1: 0.15, 2: 0.25, 3: 0.05, 4: -0.10, 5: 0.10}.get(target_job, 0.0)
                    income_change = base_income_change * multiplier + difficulty
                else:
                    income_change = base_income_change * 1.02
                
                income_change = max(-0.50, min(1.00, income_change))

                # 만족도 예측
                satis_model_features = [float(satis_row.get(f, 0.0)) for f in satis_features]
                satis_input = np.array(satis_model_features).reshape(1, -1)
                
                predictions = []
                for name, model in ensemble_models.items():
                    if model:
                        pred = float(model.predict(satis_input)[0])
                        predictions.append((pred, weights.get(name.replace('_satis', ''), 0.33)))
                
                satis_change = sum(p * w for p, w in predictions) / sum(w for _, w in predictions) if predictions else 0.0

                distribution = _generate_distribution_data(user_input, scenario_code, income_change, satis_change)
                
                results[scenario_code] = {
                    "income_change_rate": round(income_change, 4),
                    "satisfaction_change_score": round(satis_change, 4),
                    "income_class": _get_change_class(income_change),
                    "satisfaction_class": _get_change_class(satis_change),
                    "distribution": distribution,
                    "scenario": scenario_map[scenario_code]
                }

        except Exception as e:
            logger.error(f"모델 예측 실패: {e}", exc_info=True)
            for code, name in scenario_map.items():
                income_change, satis_change = _get_realistic_fallback_prediction(user_input, code)
                distribution = _generate_distribution_data(user_input, code, income_change, satis_change)
                results[code] = {
                    "income_change_rate": income_change, "satisfaction_change_score": satis_change,
                    "income_class": _get_change_class(income_change), "satisfaction_class": _get_change_class(satis_change),
                    "distribution": distribution, "scenario": name, "error": "모델 예측에 실패하여 현실적 추정치를 제공합니다."
                }
            
        return results
