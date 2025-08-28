import pandas as pd
import numpy as np
import xgboost
print(f"XGBoost Version: {xgboost.__version__}")
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# 1. 데이터 불러오기
df = pd.read_csv("data/klips_data_23.csv")

# 1-1. 만족도 통계 피처 생성
satisfaction_factors = [
    "satis_wage", "satis_stability", "satis_growth", "satis_task_content",
    "satis_work_env", "satis_work_time", "satis_communication",
    "satis_fair_eval", "satis_welfare"
]

satisfaction_data = df[satisfaction_factors].fillna(3)
df["satisfaction_mean"] = satisfaction_data.mean(axis=1)
df["satisfaction_std"] = satisfaction_data.std(axis=1).fillna(0)
df["satisfaction_min"] = satisfaction_data.min(axis=1)
df["satisfaction_max"] = satisfaction_data.max(axis=1)
df["satisfaction_range"] = df["satisfaction_max"] - df["satisfaction_min"]

# 2. Feature / Target 설정
features = [
    "age", "gender", "education", "monthly_income", "job_category",
    "satis_wage", "satis_stability", "satis_growth", "satis_task_content",
    "satis_work_env", "satis_work_time", "satis_communication",
    "satis_fair_eval", "satis_welfare", "prev_job_satisfaction",
    "prev_monthly_income", "job_category_income_avg",
    "income_relative_to_job", "job_category_education_avg", "education_relative_to_job",
    "job_category_satisfaction_avg", "age_x_job_category", "monthly_income_x_job_category",
    "education_x_job_category", "income_relative_to_job_x_job_category",
    "satisfaction_mean", "satisfaction_std", "satisfaction_min", 
    "satisfaction_max", "satisfaction_range"
]

available_features = [f for f in features if f in df.columns]
X = df[available_features]
y = df["satisfaction_change_score"]

MIN_SATISFACTION_CHANGE = -4
MAX_SATISFACTION_CHANGE = 3
print(f"Satisfaction Change Score의 실제 최소값: {MIN_SATISFACTION_CHANGE}")
print(f"Satisfaction Change Score의 실제 최대값: {MAX_SATISFACTION_CHANGE}")

# 3. 훈련/테스트 분리
latest_year = df["year"].max()
X_train_full = X[df["year"] < latest_year]
y_train_full = y[df["year"] < latest_year]
X_test = X[df["year"] == latest_year]
y_test = y[df["year"] == latest_year]

# GridSearchCV와 최종 훈련을 위해 훈련 데이터를 다시 훈련용/검증용으로 분리
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# 4. XGBoost 하이퍼파라미터 튜닝
tscv = TimeSeriesSplit(n_splits=5)

# ================= 모델 재훈련 시 조정 가이드 =================
# 모델 성능 개선을 위해 아래 파라미터 범위를 조정하며 실험해볼 수 있습니다.
# n_estimators: 모델의 복잡도. 너무 높으면 과적합, 낮으면 과소적합. [100, 200, 300] 등으로 조정.
# max_depth: 트리의 최대 깊이. [3, 5, 7] 등으로 조정.
# learning_rate: 학습률. [0.01, 0.05, 0.1] 등으로 작게 조정하며 n_estimators를 늘리는 것이 일반적.
# reg_alpha (L1), reg_lambda (L2): 정규화 파라미터. 과적합 방지에 중요. [0, 0.1, 1, 2] 등 다양한 값 테스트.
# ==========================================================
xgb_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 1.5]
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
xgb_best_params = xgb_search.best_params_
print(f"XGBoost Best Params: {xgb_best_params}")

# GridSearchCV가 찾은 최적의 모델을 최종 모델로 사용합니다.
# 참고: GridSearchCV는 기본적으로 최적의 파라미터로 전체 훈련 데이터에 대해 모델을 다시 훈련시킵니다 (refit=True).
# 환경 문제로 early_stopping을 직접 사용하지 못하지만, CV를 통해 최적의 n_estimators가 선택됩니다.
print("\n--- XGBoost Final Model Selection ---")
xgb_final = xgb_search.best_estimator_

y_pred_xgb = xgb_final.predict(X_test)


# 5. CatBoost 하이퍼파라미터 튜닝
# ================= 모델 재훈련 시 조정 가이드 =================
# iterations: n_estimators와 동일.
# depth: max_depth와 동일.
# l2_leaf_reg: L2 정규화. 과적합 방지에 중요. [1, 3, 5, 10] 등으로 조정.
# ==========================================================
cat_grid = {
    'iterations': [100, 150],
    'depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5]
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
cat_best_params = cat_search.best_params_
print(f"CatBoost Best Params: {cat_best_params}")

# 최적 파라미터로 최종 모델 훈련 (조기 종료 적용)
print("\n--- CatBoost Final Model Training with Early Stopping ---")
cat_final = CatBoostRegressor(**cat_best_params, verbose=0, random_state=42)
cat_final.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10,
    cat_features=["gender", "education", "job_category"],
    verbose=False
)

# 학습 곡선 시각화
eval_results = cat_final.get_evals_result()
plt.figure(figsize=(10, 6))
plt.plot(eval_results['validation']['RMSE'], label='Validation RMSE')
plt.title('CatBoost Learning Curve')
plt.xlabel('Boosting Round')
plt.ylabel('RMSE')
plt.legend()
plt.show()

y_pred_cat = cat_final.predict(X_test)

# 6. Soft-Blending
alpha = 0.3
y_pred_blend = alpha * y_pred_xgb + (1 - alpha) * y_pred_cat

# --- 예측값 후처리: 실제 데이터 범위로 클리핑 (정보 손실 최소화) ---
y_pred_xgb_clipped = np.clip(y_pred_xgb, MIN_SATISFACTION_CHANGE, MAX_SATISFACTION_CHANGE)
y_pred_cat_clipped = np.clip(y_pred_cat, MIN_SATISFACTION_CHANGE, MAX_SATISFACTION_CHANGE)
y_pred_blend_clipped = np.clip(y_pred_blend, MIN_SATISFACTION_CHANGE, MAX_SATISFACTION_CHANGE)
# ----------------------------------------------------

# 7. 평가 함수
def evaluate(name, y_true, y_pred):
    # RMSE는 MSE에 제곱근을 취한 값입니다.
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n-------- {name} --------")
    print("RMSE:", round(rmse, 4))
    print("MAE :", round(mae, 4))
    print("R2  :", round(r2, 4))

# 8. 결과 출력 (후처리된 예측값으로 평가)
evaluate("XGBoost (Tuned, Clipped)", y_test, y_pred_xgb_clipped)
evaluate("CatBoost (Tuned, Clipped)", y_test, y_pred_cat_clipped)
evaluate("Soft-Blended Ensemble (Clipped)", y_test, y_pred_blend_clipped)

# 9. Feature Importance 시각화
def plot_importance(model, feature_names, model_type, top_n=15):
    if model_type == 'xgb':
        importances = model.feature_importances_
    elif model_type == 'cat':
        importances = model.get_feature_importance()
    
    fi = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=fi, palette='viridis')
    plt.title(f'{model_type.upper()} Feature Importance (Top {top_n})')
    plt.tight_layout()
    plt.show()

plot_importance(xgb_final, X_train.columns, 'xgb')
plot_importance(cat_final, X_train.columns, 'cat')

# 10. 모델 저장
os.makedirs("app/ml/saved_models", exist_ok=True)

xgb_path = "app/ml/saved_models/xgb_satisfaction_change_model.pkl"
joblib.dump(xgb_final, xgb_path)
print(f"\n[저장 완료] XGBoost 모델 → {xgb_path}")

cat_path = "app/ml/saved_models/cat_satisfaction_change_model.cbm"
cat_final.save_model(cat_path)
print(f"[저장 완료] CatBoost 모델 → {cat_path}")
