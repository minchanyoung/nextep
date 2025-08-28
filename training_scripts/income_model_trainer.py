import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import numpy as np

# ==============================================================================
# 1. 데이터 및 피처 준비 (Data & Feature Preparation)
# ==============================================================================
def load_data(file_path="data/klips_data_23.csv"):
    """CSV 파일에서 데이터를 로드합니다."""
    return pd.read_csv(file_path)

def prepare_features_and_target(df):
    """데이터프레임에서 피처와 타겟 변수를 분리합니다."""
    features = [
        "age", "gender", "education", "monthly_income", "job_category",
        "satis_wage", "satis_stability", "satis_growth", "satis_task_content",
        "satis_work_env", "satis_work_time", "satis_communication",
        "satis_fair_eval", "satis_welfare", "job_satisfaction", "prev_job_satisfaction", 
        "prev_monthly_income", "job_category_income_avg",
        "income_relative_to_job", "job_category_education_avg", "education_relative_to_job",
        "job_category_satisfaction_avg", "age_x_job_category", "monthly_income_x_job_category",
        "education_x_job_category", "income_relative_to_job_x_job_category",
        # 새로운 만족도 통계 피처
        "satisfaction_mean", "satisfaction_std", "satisfaction_min", 
        "satisfaction_max", "satisfaction_range"
    ]
    
    # 존재하는 컬럼만 선택
    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]
    
    if missing_features:
        print(f"Warning: Missing features in data: {missing_features}")
    
    X = df[available_features]
    y = df["income_change_rate"]
    return X, y

def split_data_by_year(df, X, y):
    """최신 연도를 테스트 데이터로 사용하여 시계열 분할을 수행합니다."""
    latest_year = df["year"].max()
    X_train = X[df["year"] < latest_year]
    y_train = y[df["year"] < latest_year]
    X_test = X[df["year"] == latest_year]
    y_test = y[df["year"] == latest_year]
    return X_train, y_train, X_test, y_test

# ==============================================================================
# 2. 모델 튜닝 및 학습 (Model Tuning & Training)
# ==============================================================================
def tune_xgboost(X_train, y_train):
    """GridSearchCV를 사용하여 XGBoost 모델의 하이퍼파라미터를 튜닝합니다."""
    tscv = TimeSeriesSplit(n_splits=5)
    xgb_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    xgb_search = GridSearchCV(
        estimator=XGBRegressor(random_state=42),
        param_grid=xgb_grid,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    xgb_search.fit(X_train, y_train)
    return xgb_search.best_estimator_

def tune_catboost(X_train, y_train):
    """GridSearchCV를 사용하여 CatBoost 모델의 하이퍼파라미터를 튜닝합니다."""
    tscv = TimeSeriesSplit(n_splits=5)
    cat_grid = {
        'iterations': [100, 150],
        'depth': [3, 5],
        'learning_rate': [0.05, 0.1]
    }
    cat_search = GridSearchCV(
        estimator=CatBoostRegressor(verbose=0, random_state=42),
        param_grid=cat_grid,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    cat_search.fit(X_train, y_train, cat_features=["gender", "education", "job_category"])
    return cat_search.best_estimator_

# ==============================================================================
# 3. 평가 및 시각화 (Evaluation & Visualization)
# ==============================================================================
def evaluate(name, y_true, y_pred):
    """모델 성능을 평가하고 결과를 출력합니다."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n-------- {name} --------")
    print(f" RMSE: {rmse:.4f}")
    print(f" MAE : {mae:.4f}")
    print(f" R²  : {r2:.4f}")

def plot_feature_importance(model, feature_names, model_name, top_n=15):
    """모델의 피처 중요도를 시각화합니다."""
    if isinstance(model, XGBRegressor):
        importances = model.feature_importances_
        palette = 'Greens_d'
    elif isinstance(model, CatBoostRegressor):
        importances = model.get_feature_importance()
        palette = 'Blues_d'
    else:
        return

    fi = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=fi, palette=palette)
    plt.title(f'{model_name} Feature Importance (Top {top_n})')
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 4. 모델 저장 (Model Saving)
# ==============================================================================
def save_models(xgb_model, cat_model):
    """학습된 모델을 파일로 저장합니다."""
    output_dir = "app/ml/saved_models"
    os.makedirs(output_dir, exist_ok=True)

    xgb_path = os.path.join(output_dir, "xgb_income_change_model.pkl")
    joblib.dump(xgb_model, xgb_path)
    print(f"[저장 완료] XGBoost 모델 → {xgb_path}")

    cat_path = os.path.join(output_dir, "cat_income_change_model.cbm")
    cat_model.save_model(cat_path)
    print(f"[저장 완료] CatBoost 모델 → {cat_path}")

# ==============================================================================
# 5. 메인 실행 로직 (Main Execution)
# ==============================================================================
def main():
    """전체 모델 학습 및 평가 파이프라인을 실행합니다."""
    # 데이터 로드 및 준비
    df = load_data()
    X, y = prepare_features_and_target(df)
    X_train, y_train, X_test, y_test = split_data_by_year(df, X, y)

    # 모델 튜닝 및 학습
    print("--- XGBoost 모델 튜닝 시작 ---")
    xgb_best = tune_xgboost(X_train, y_train)
    
    print("\n--- CatBoost 모델 튜닝 시작 ---")
    cat_best = tune_catboost(X_train, y_train)

    # 예측
    y_pred_xgb = xgb_best.predict(X_test)
    y_pred_cat = cat_best.predict(X_test)
    
    # Soft-Blending
    alpha = 0.3
    y_pred_blend = alpha * y_pred_xgb + (1 - alpha) * y_pred_cat

    # 평가
    evaluate("XGBoost (Tuned)", y_test, y_pred_xgb)
    evaluate("CatBoost (Tuned)", y_test, y_pred_cat)
    evaluate("Soft-Blended Ensemble", y_test, y_pred_blend)

    # 시각화
    plot_feature_importance(xgb_best, X.columns, "XGBoost")
    plot_feature_importance(cat_best, X.columns, "CatBoost")

    # 모델 저장
    save_models(xgb_best, cat_best)

if __name__ == "__main__":
    main()
