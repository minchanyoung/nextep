import os
import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings("ignore")

def create_advanced_features_satis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["pid", "year"]).reset_index(drop=True)
    df["satisfaction_lag_1"] = df.groupby("pid")["job_satisfaction"].shift(1)
    df["income_lag_1"] = df.groupby("pid")["monthly_income"].shift(1)
    df["satisfaction_roll_mean_3"] = (
        df.groupby("pid")["job_satisfaction"].shift(1).rolling(3, min_periods=1).mean()
    )
    df["satisfaction_roll_std_3"] = (
        df.groupby("pid")["job_satisfaction"].shift(1).rolling(3, min_periods=1).std()
    )
    df["income_roll_mean_3"] = (
        df.groupby("pid")["monthly_income"].shift(1).rolling(3, min_periods=1).mean()
    )
    df["career_length"] = (df["age"] - df.groupby("pid")["age"].transform("min")).clip(lower=0)
    df["career_stage"] = pd.cut(df["age"], bins=[0, 29, 39, 49, 100], labels=[1, 2, 3, 4], right=True).astype(int)
    satis_cols = [f"satis_{c}" for c in ["wage","stability","growth","task_content","work_env","work_time","communication","fair_eval","welfare"]]
    df[satis_cols] = df[satis_cols].fillna(3)
    df["satisfaction_mean"] = df[satis_cols].mean(axis=1)
    df["satisfaction_std"] = df[satis_cols].std(axis=1)
    df["satisfaction_range"] = df[satis_cols].max(axis=1) - df[satis_cols].min(axis=1)
    df["age_x_education"] = df["age"] * df["education"]
    df["income_x_satisfaction_lag_1"] = df["income_lag_1"] * df["satisfaction_lag_1"]
    df = df.fillna(method="bfill").fillna(method="ffill")
    return df

def split_and_add_stats(df: pd.DataFrame):
    # 더 안정적인 시계열 분할: 최신 2년을 테스트로 사용
    latest_year = df["year"].max()
    test_years = [latest_year, latest_year - 1]
    train_val_df = df[~df["year"].isin(test_years)].copy()
    test_df = df[df["year"].isin(test_years)].copy()
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, stratify=train_val_df["year"])
    job_cat_stats = (
        train_df.groupby("job_category").agg(
            job_cat_income_avg=("monthly_income", "mean"),
            job_cat_satis_avg=("job_satisfaction", "mean"),
        )
    ).reset_index()
    def add_stats(d):
        return d.merge(job_cat_stats, on="job_category", how="left")
    train_df = add_stats(train_df)
    val_df = add_stats(val_df)
    test_df = add_stats(test_df)
    train_df = create_advanced_features_satis(train_df)
    val_df = create_advanced_features_satis(val_df)
    test_df = create_advanced_features_satis(test_df)
    return train_df, val_df, test_df, latest_year

def train_one(model, X_tr, y_tr, X_val, y_val, name, cat_features_idx=None):
    if name == "XGBoost":
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    elif name == "LightGBM":
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    elif name == "CatBoost":
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=cat_features_idx, early_stopping_rounds=100, verbose=False)
    pred_val = model.predict(X_val)
    r2 = r2_score(y_val, pred_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred_val))
    mae = mean_absolute_error(y_val, pred_val)
    return model, r2

def plot_feature_importance(model, features, model_name, save_dir):
    if not hasattr(model, "feature_importances_"):
        return
    imp = model.feature_importances_
    fi = (pd.DataFrame({"feature": features, "importance": imp}).sort_values("importance", ascending=False).head(20))
    plt.figure(figsize=(10, 8))
    sns.barplot(x="importance", y="feature", data=fi, orient="h")
    plt.title(f"{model_name} Feature Importance (Top 20)")
    plt.tight_layout()
    path = os.path.join(save_dir, f"feature_importance_{model_name.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    data_path = "C:/Users/User/Desktop/nextep/data/klips_data_23.csv"
    save_dir = "app/ml/saved_models"
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    train_df, val_df, test_df, latest_year = split_and_add_stats(df)
    base_features = [
        "age","gender","education","monthly_income","job_category",
        "satis_wage","satis_stability","satis_growth","satis_task_content","satis_work_env",
        "satis_work_time","satis_communication","satis_fair_eval","satis_welfare","prev_job_satisfaction",
    ]
    # 피처 누출 가능성이 있는 satisfaction 관련 lag 피처 제거
    gen_features = [
        "income_lag_1","income_roll_mean_3","career_length","career_stage",
        "satisfaction_mean","satisfaction_std","satisfaction_range",
        "age_x_education","job_cat_income_avg","job_cat_satis_avg",
    ]
    feature_cols = [f for f in base_features + gen_features if f in train_df.columns]
    target_col = "satisfaction_change_score"
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]
    cat_cols = [c for c in ["gender","education","job_category","career_stage"] if c in feature_cols]
    cat_idx = [feature_cols.index(c) for c in cat_cols]

    # CatBoost 모델만 훈련
    cat_model = ctb.CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3, loss_function="RMSE", random_state=42, allow_writing_files=False)
    cat_model, val_r2_cat = train_one(cat_model, X_train, y_train, X_val, y_val, "CatBoost", cat_features_idx=cat_idx)

    # 테스트 세트에서 예측 및 평가
    pred_cat = cat_model.predict(X_test)
    def scores(y_true, pred):
        return (r2_score(y_true, pred), np.sqrt(mean_squared_error(y_true, pred)), mean_absolute_error(y_true, pred))
    r2_cat, rmse_cat, mae_cat = scores(y_test, pred_cat)

    # 모델 저장
    cat_model.save_model(os.path.join(save_dir, "final_cat_satis_model.cbm"))

    # 설정 저장 (CatBoost 단일 모델용)
    cat_cfg = {
        "features": feature_cols,
        "cat_features": cat_cols,
        "split": {"test_year": int(latest_year), "val_ratio": 0.2, "stratify": "year"},
        "model_type": "catboost_only"
    }
    with open(os.path.join(save_dir, "final_catboost_satis_config.json"), "w") as f:
        json.dump(cat_cfg, f, indent=2)

    # 피처 중요도 출력
    cat_fi = cat_model.get_feature_importance(prettified=True)
    print(cat_fi.head(15))

    # 성능지표 출력
    print("\n" + "=" * 80)
    print("SATISFACTION MODEL PERFORMANCE METRICS (CatBoost Only)")
    print("=" * 80)
    print(f"CatBoost:")
    print(f"  R2   : {r2_cat:.4f}")
    print(f"  RMSE : {rmse_cat:.4f}")
    print(f"  MAE  : {mae_cat:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
