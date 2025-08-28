from flask import current_app
import sys, json, os, logging
import joblib
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

logger = logging.getLogger(__name__)

# ==============================================================================
# 1) 경로/상수
# ==============================================================================
def get_paths():
    APP_ROOT = current_app.root_path
    MODEL_DIR = os.path.join(APP_ROOT, "ml", "saved_models")
    DATA_PATH = os.path.join(APP_ROOT, "..", "data", "klips_data_23.csv")
    return {
        "CAT_INCOME_MODEL_PATH": os.path.join(MODEL_DIR, "cat_income_change_model.cbm"),
        "XGB_SATIS_MODEL_PATH": os.path.join(MODEL_DIR, "xgb_satisfaction_change_model.pkl"),
        "CAT_SATIS_MODEL_PATH": os.path.join(MODEL_DIR, "cat_satisfaction_change_model.cbm"),
        "DATA_PATH": DATA_PATH
    }

ALPHA = 0.3
MIN_SATISFACTION_CHANGE = -4
MAX_SATISFACTION_CHANGE = 3

SATIS_FEATURES = [
    "age",
    "gender",
    "education",
    "monthly_income",
    "job_category",
    "satis_wage",
    "satis_stability",
    "satis_growth",
    "satis_task_content",
    "satis_work_env",
    "satis_work_time",
    "satis_communication",
    "satis_fair_eval",
    "satis_welfare",
    "prev_job_satisfaction",
    "prev_monthly_income",
    "job_category_income_avg",
    "income_relative_to_job",
    "job_category_education_avg",
    "education_relative_to_job",
    "job_category_satisfaction_avg",
    "age_x_job_category",
    "monthly_income_x_job_category",
    "education_x_job_category",
    "income_relative_to_job_x_job_category",
    "satisfaction_mean",
    "satisfaction_std",
    "satisfaction_min",
    "satisfaction_max",
    "satisfaction_range"
]
INCOME_FEATURES = SATIS_FEATURES + ["job_satisfaction"]

# ==============================================================================
# 2) 전역 모델/데이터 핸들
# ==============================================================================
cat_income = None
xgb_satis = None
cat_satis = None
klips_df = None
job_category_stats = None

# ==============================================================================
# 3) 초기화 함수 (유일한 init_app)
# ==============================================================================
def init_app(app):
    global cat_income, xgb_satis, cat_satis, klips_df, job_category_stats, llm_model, llm_tokenizer
    with app.app_context():
        paths = get_paths()

        # --- ML 모델/데이터 로드 ---
        try:
            logger.info("Pre-loading ML models and data...")

            # CatBoost는 인스턴스를 만들고 load_model 호출하는 패턴이 안전합니다.
            cat_income_model = CatBoostRegressor()
            cat_income_model.load_model(paths["CAT_INCOME_MODEL_PATH"])
            cat_income = cat_income_model

            xgb_satis = joblib.load(paths["XGB_SATIS_MODEL_PATH"])

            cat_satis_model = CatBoostRegressor()
            cat_satis_model.load_model(paths["CAT_SATIS_MODEL_PATH"])
            cat_satis = cat_satis_model

            klips_df = pd.read_csv(paths["DATA_PATH"])

            logger.info("ML models and data pre-loaded successfully.")

            logger.info("Calculating job category statistics...")
            job_category_stats = klips_df.groupby('job_category').agg({
                'monthly_income': 'mean',
                'education': 'mean',
                'job_satisfaction': 'mean'
            }).rename(columns={
                'monthly_income': 'job_category_income_avg',
                'education': 'job_category_education_avg',
                'job_satisfaction': 'job_category_satisfaction_avg'
            })
            logger.info("Job category statistics calculated successfully.")
        except FileNotFoundError as e:
            logger.critical(f"ML 모델 또는 데이터 파일을 찾을 수 없습니다: {e}")
            logger.critical("모델 파일이 올바른 경로에 있는지 확인해주세요.")
            # 기본값으로 설정하여 애플리케이션이 계속 실행될 수 있도록 함
            cat_income = None
            xgb_satis = None
            cat_satis = None
            klips_df = None
            job_category_stats = None
        except Exception as e:
            logger.critical("ML 모델 또는 데이터 로딩 중 예기치 못한 오류가 발생했습니다.")
            logger.critical(f"오류 세부사항: {e}")
            # 기본값으로 설정하여 애플리케이션이 계속 실행될 수 있도록 함
            cat_income = None
            xgb_satis = None
            cat_satis = None
            klips_df = None
            job_category_stats = None

# ==============================================================================
# 4) 코어 함수
# ==============================================================================
def get_similar_cases_distribution(scenario_data):
    if klips_df is None:
        return None
    try:
        job_cat = scenario_data['job_category']
        age = scenario_data['age']
        edu = scenario_data['education']
        gender = scenario_data['gender']

        filters = [
            (klips_df['job_category'] == job_cat) & (klips_df['age'].between(age - 5, age + 5)) & (klips_df['education'] == edu) & (klips_df['gender'] == gender),
            (klips_df['job_category'] == job_cat) & (klips_df['age'].between(age - 7, age + 7)) & (klips_df['education'] == edu),
            (klips_df['job_category'] == job_cat) & (klips_df['age'].between(age - 10, age + 10))
        ]
        similar_df = pd.DataFrame()
        for f in filters:
            similar_df = klips_df[f]
            if len(similar_df) >= 10:
                break
        if len(similar_df) < 10:
            return None

        income_data = similar_df['income_change_rate']
        Q1, Q3 = income_data.quantile(0.25), income_data.quantile(0.75)
        IQR = Q3 - Q1
        income_hist_data = income_data[~((income_data < (Q1 - 1.5 * IQR)) | (income_data > (Q3 + 1.5 * IQR)))]
        if len(income_hist_data) < 10:
            income_hist_data = income_data

        income_hist = np.histogram(income_hist_data, bins=8)
        satis_hist = np.histogram(similar_df['satisfaction_change_score'], bins=8)
        return {
            "income": {"counts": income_hist[0].tolist(), "bins": income_hist[1].tolist()},
            "satisfaction": {"counts": satis_hist[0].tolist(), "bins": satis_hist[1].tolist()}
        }
    except Exception:
        return None

def predict_scenario(row):
    income_df = pd.DataFrame([row])[INCOME_FEATURES]
    satis_df = pd.DataFrame([row])[SATIS_FEATURES]
    income_pred = cat_income.predict(income_df)[0]
    satis_pred_blend = ALPHA * xgb_satis.predict(satis_df)[0] + (1 - ALPHA) * cat_satis.predict(satis_df)[0]
    satis_pred_processed = np.clip(satis_pred_blend, MIN_SATISFACTION_CHANGE, MAX_SATISFACTION_CHANGE)
    distribution_data = get_similar_cases_distribution(row)
    return round(income_pred, 4), round(satis_pred_processed, 4), distribution_data

def run_prediction(scenarios_data):
    if klips_df is None or cat_income is None or xgb_satis is None or cat_satis is None:
        logger.warning("ML 모델이 로드되지 않았습니다. 기본값을 반환합니다.")
        # 모델이 로드되지 않은 경우 기본값 반환
        return [{
            "income_change_rate": 0.05,  # 기본 5% 증가
            "satisfaction_change_score": 0.0,  # 기본 변화 없음
            "distribution": None,
            "warning": "ML 모델을 사용할 수 없어 기본 예측값을 제공합니다."
        } for _ in scenarios_data]
    
    results = []
    for scenario in scenarios_data:
        try:
            income, satis, dist = predict_scenario(scenario)
            results.append({
                "income_change_rate": income,
                "satisfaction_change_score": satis,
                "distribution": dist
            })
        except Exception as e:
            logger.error(f"시나리오 예측 중 오류 발생: {scenario}. 오류: {e}")
            results.append({
                "income_change_rate": 0.02,  # 기본 2% 증가
                "satisfaction_change_score": 0.0,  # 기본 변화 없음
                "distribution": None,
                "error": "이 시나리오 예측 중 오류가 발생했습니다."
            })
    return results

# ==============================================================================
# 5) 단독 실행 시 (디버그용)
# ==============================================================================
def main():
    try:
        scenarios_data = json.loads(sys.stdin.read())
        results = run_prediction(scenarios_data)
        logger.info(json.dumps(results))
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
