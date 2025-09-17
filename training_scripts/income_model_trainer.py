# -*- coding: utf-8 -*-
"""
income_model_trainer.py
- 데이터 로드/피처 엔지니어링
- 시계열 분할
- XGBoost / CatBoost / LightGBM (TimeSeriesSplit) 튜닝 & 학습
- 평가표 생성 + 시각화 PNG 저장 (모델 비교 / 실제vs예측 / 잔차 / 중요도)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb

warnings.filterwarnings("ignore")


# ==============================================================================
# 0) 공통 유틸: 평가 & 시각화
# ==============================================================================
def compute_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def evaluate_models(predictions_dict, y_test):
    rows = []
    for name, y_pred in predictions_dict.items():
        m = compute_metrics(y_test, y_pred)
        rows.append({"model": name, **m})
    df = pd.DataFrame(rows).set_index("model").sort_values("r2", ascending=False)
    print("\n=== 모델 성능 표 (Test) ===")
    print(df.round(4))
    return df


def plot_model_comparison(results_df, title="모델 성능 비교 (Test)", savepath=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ["rmse", "mae", "r2"]
    titles = ["RMSE", "MAE", "R²"]
    colors = ["lightcoral", "lightblue", "lightgreen"]

    for i, (metric, t, c) in enumerate(zip(metrics, titles, colors)):
        ax = axes[i]
        results_df[metric].plot(kind="barh", ax=ax, color=c, alpha=0.85)
        ax.set_xlabel(metric.upper())
        ax.set_title(t)
        for j, v in enumerate(results_df[metric]):
            ax.text(
                v + (0.01 if metric != "r2" else -0.05),
                j,
                f"{v:.3f}",
                va="center",
                fontweight="bold",
            )
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"[저장] {savepath}")
    plt.close(fig)


def plot_actual_vs_pred(
    y_test, predictions_dict, max_cols=3, title="실제 vs 예측 (Test)", savepath=None
):
    names = list(predictions_dict.keys())
    n = len(names)
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.atleast_1d(axes).ravel()

    y_min, y_max = float(np.min(y_test)), float(np.max(y_test))
    line = np.linspace(y_min, y_max, 100)

    for i, name in enumerate(names):
        ax = axes[i]
        pred = predictions_dict[name]
        ax.scatter(y_test, pred, alpha=0.5, s=10)
        ax.plot(line, line, linestyle="--", linewidth=1)
        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        ax.set_title(f"{name}  (R²={r2:.3f}, RMSE={rmse:.3f})")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"[저장] {savepath}")
    plt.close(fig)


def plot_residuals(
    y_test, predictions_dict, max_cols=3, title="잔차 분석", savepath=None
):
    names = list(predictions_dict.keys())
    n = len(names)
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i, name in enumerate(names):
        ax = axes[i]
        pred = predictions_dict[name]
        resid = y_test - pred
        ax.scatter(pred, resid, alpha=0.5, s=10)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_title(f"{name} Residuals")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"[저장] {savepath}")
    plt.close(fig)


def plot_feature_importance_generic(
    model, feature_names, top_n=20, title=None, savepath=None
):
    importances = None
    name = type(model).__name__

    if hasattr(model, "get_booster"):  # XGBRegressor
        importances = np.array(model.feature_importances_)
    elif hasattr(model, "feature_importances_"):  # LGBMRegressor
        importances = np.array(model.feature_importances_)
    elif hasattr(model, "get_feature_importance"):  # CatBoostRegressor
        importances = np.array(model.get_feature_importance())
    else:
        print(f"[경고] {name}는 중요도 속성을 찾을 수 없습니다.")
        return

    idx = np.argsort(importances)[::-1][:top_n]
    top_feats = [feature_names[i] for i in idx]
    top_imps = importances[idx]

    plt.figure(figsize=(10, 0.45 * top_n + 1))
    plt.barh(range(len(top_feats)), top_imps, alpha=0.85)
    plt.gca().invert_yaxis()
    plt.yticks(range(len(top_feats)), top_feats)
    plt.xlabel("Importance")
    plt.title(title or f"{name} Feature Importance (Top {top_n})")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"[저장] {savepath}")
    plt.close()


# ==============================================================================
# 1) 데이터 로드 & 피처 엔지니어링
# ==============================================================================
def load_data(file_path="data/klips_data_23.csv"):
    return pd.read_csv(file_path)


def prepare_features_and_target(df):
    df = df.sort_values(["pid", "year"]).reset_index(drop=True)

    # 시계열 파생
    df["income_lag1"] = df.groupby("pid")["monthly_income"].shift(1)
    df["income_lag2"] = df.groupby("pid")["monthly_income"].shift(2)
    df["income_trend"] = (
        df.groupby("pid")["monthly_income"].pct_change(periods=2).fillna(0)
    )

    df["prev_income_change"] = (
        df.groupby("pid")["income_change_rate"].shift(1).fillna(0)
    )
    df["income_volatility"] = (
        df.groupby("pid")["income_change_rate"]
        .rolling(3, min_periods=1)
        .std()
        .reset_index(0, drop=True)
        .fillna(0)
    )

    df["satisfaction_trend"] = (
        df.groupby("pid")["job_satisfaction"].pct_change().fillna(0)
    )
    df["satisfaction_volatility"] = (
        df.groupby("pid")["job_satisfaction"]
        .rolling(3, min_periods=1)
        .std()
        .reset_index(0, drop=True)
        .fillna(0)
    )

    df["career_length"] = df.groupby("pid").cumcount() + 1
    df["job_stability"] = (
        df.groupby("pid")["job_category"]
        .apply(lambda x: (x == x.iloc[0]).astype(int))
        .reset_index(0, drop=True)
    )

    economic_cycle = {
        2010: -1,
        2011: -0.5,
        2012: 0,
        2013: 0.5,
        2014: 1,
        2015: 0.5,
        2016: 0,
        2017: 0.5,
        2018: 1,
        2019: 0.5,
        2020: -2,
        2021: -1,
        2022: 0,
        2023: 0.5,
    }
    df["economic_cycle"] = df["year"].map(economic_cycle)

    df["income_age_ratio"] = df["monthly_income"] / df["age"]
    df["peak_earning_years"] = ((df["age"] >= 40) & (df["age"] <= 55)).astype(int)
    df["education_roi"] = df["monthly_income"] / (df["education"] + 1)
    df["satisfaction_income_gap"] = (
        df["satis_wage"] - (df["monthly_income"] / df["monthly_income"].mean() * 3)
    ).fillna(0)

    df["job_category_change"] = (df.groupby("pid")["job_category"].diff() != 0).astype(
        int
    )
    df["potential_promotion"] = (
        (df["satisfaction_change_score"] > 0) & (df["satis_growth"] >= 4)
    ).astype(int)

    df["career_stage"] = pd.cut(
        df["age"], bins=[0, 25, 35, 45, 55, 100], labels=[1, 2, 3, 4, 5]
    ).astype(int)

    df["year_job_income_avg"] = df.groupby(["year", "job_category"])[
        "monthly_income"
    ].transform("mean")
    df["income_vs_peers"] = df["monthly_income"] - df["year_job_income_avg"]

    features = [
        "age",
        "gender",
        "education",
        "monthly_income",
        "job_category",
        "job_satisfaction",
        "prev_job_satisfaction",
        "satis_wage",
        "satis_stability",
        "satis_growth",
        "satisfaction_change_score",
        "income_lag1",
        "income_lag2",
        "income_trend",
        "prev_income_change",
        "income_volatility",
        "satisfaction_trend",
        "satisfaction_volatility",
        "career_length",
        "job_stability",
        "economic_cycle",
        "income_age_ratio",
        "peak_earning_years",
        "education_roi",
        "satisfaction_income_gap",
        "job_category_change",
        "potential_promotion",
        "career_stage",
        "income_vs_peers",
    ]

    available_features = [f for f in features if f in df.columns]
    X = df[available_features].replace([np.inf, -np.inf], np.nan)
    y = df["income_change_rate"]

    # 극값 클리핑 + 결측 대체
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q99 = X[col].quantile(0.999)
        Q01 = X[col].quantile(0.001)
        X[col] = X[col].clip(lower=Q01, upper=Q99)
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    return df, X, y, available_features


def split_data_by_year(df, X, y, test_years=(2022, 2023)):
    valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
    Xc, yc, dff = X[valid_mask], y[valid_mask], df[valid_mask]

    test_years = list(test_years)
    tr_mask = ~dff["year"].isin(test_years)
    te_mask = dff["year"].isin(test_years)

    X_train, y_train = Xc[tr_mask], yc[tr_mask]
    X_test, y_test = Xc[te_mask], yc[te_mask]

    print(
        f"훈련 데이터: {len(X_train)}개 ({int(dff[tr_mask]['year'].min())}-{int(dff[tr_mask]['year'].max())})"
    )
    print(
        f"테스트 데이터: {len(X_test)}개 ({int(dff[te_mask]['year'].min())}-{int(dff[te_mask]['year'].max())})"
    )
    return X_train, y_train, X_test, y_test


# ==============================================================================
# 2) 모델 튜닝 (TimeSeriesSplit 기반)
# ==============================================================================
def tune_xgboost(X_train, y_train):
    tscv = TimeSeriesSplit(n_splits=3)
    xgb_grid = {
        "n_estimators": [300],
        "max_depth": [6, 8],
        "learning_rate": [0.05],
        "subsample": [0.8],
        "colsample_bytree": [0.9],
        "reg_alpha": [0.1],
        "reg_lambda": [1.5],
    }
    search = GridSearchCV(
        estimator=XGBRegressor(
            random_state=42, objective="reg:squarederror", tree_method="hist", n_jobs=-1
        ),
        param_grid=xgb_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def tune_catboost(X_train, y_train):
    tscv = TimeSeriesSplit(n_splits=3)
    cat_grid = {
        "iterations": [500],
        "depth": [6, 8],
        "learning_rate": [0.05],
        "l2_leaf_reg": [3],
        "border_count": [64],
    }
    search = GridSearchCV(
        estimator=CatBoostRegressor(
            verbose=0, random_state=42, loss_function="RMSE", bootstrap_type="Bayesian"
        ),
        param_grid=cat_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train, cat_features=["gender", "education", "job_category"])
    return search.best_estimator_


def tune_lightgbm(X_train, y_train):
    
    tscv = TimeSeriesSplit(n_splits=3)

    base = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    param_dist = {
        "num_leaves": [31, 63, 95],
        "max_depth": [-1, 6, 10],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_samples": [20, 40, 60],
        "reg_alpha": [0.0, 0.1, 0.3],
        "reg_lambda": [0.0, 0.1, 0.3],
    }

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=12,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        verbose=2,
        random_state=42,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    print("LightGBM best params:", search.best_params_)
    print("LightGBM best RMSE (CV):", -search.best_score_)
    return search.best_estimator_


# ==============================================================================
# 3) 메인: 학습 → 평가표/PNG 저장
# ==============================================================================
def main():
    # 출력 디렉토리
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # 1. 데이터
    print("=" * 80)
    print("고급 소득 변화율 예측 모델 훈련 시작")
    print("=" * 80)
    df = load_data("data/klips_data_23.csv")
    df, X, y, feature_names = prepare_features_and_target(df)
    X_train, y_train, X_test, y_test = split_data_by_year(
        df, X, y, test_years=(2022, 2023)
    )

    # 2. 모델 학습
    models, predictions = {}, {}

    print("\nLightGBM 튜닝/학습 중...")
    lgb_best = tune_lightgbm(X_train, y_train)
    models["LightGBM"] = lgb_best
    predictions["LightGBM"] = lgb_best.predict(X_test)

    print("\nXGBoost 튜닝/학습 중...")
    xgb_best = tune_xgboost(X_train, y_train)
    models["XGBoost"] = xgb_best
    predictions["XGBoost"] = xgb_best.predict(X_test)

    print("\nCatBoost 튜닝/학습 중...")
    cat_best = tune_catboost(X_train, y_train)
    models["CatBoost"] = cat_best
    predictions["CatBoost"] = cat_best.predict(X_test)

    print("\n앙상블 생성...")
    predictions["Ensemble"] = (
        0.6 * predictions["LightGBM"] + 0.4 * predictions["XGBoost"]
    )

    # 3. 평가 표
    results_df = evaluate_models(predictions, y_test)
    results_path = os.path.join(out_dir, "model_results.csv")
    results_df.to_csv(results_path)
    print(f"[저장] {results_path}")

    # 4. 시각화 저장 (PNG)
    plot_model_comparison(
        results_df,
        title="Model Comparison (Test: 2022–2023)",
        savepath=os.path.join(out_dir, "model_comparison.png"),
    )

    plot_actual_vs_pred(
        y_test,
        {
            "LightGBM": predictions["LightGBM"],
            "XGBoost": predictions["XGBoost"],
            "Ensemble": predictions["Ensemble"],
        },
        title="Actual vs Predict (Test: 2022–2023)",
        savepath=os.path.join(out_dir, "actual_vs_pred.png"),
    )

    plot_residuals(
        y_test,
        {
            "LightGBM": predictions["LightGBM"],
            "XGBoost": predictions["XGBoost"],
            "Ensemble": predictions["Ensemble"],
        },
        title="Residuals (Test: 2022–2023)",
        savepath=os.path.join(out_dir, "residuals.png"),
    )

    # 중요도 (각 모델별 저장)
    plot_feature_importance_generic(
        models["LightGBM"],
        feature_names,
        top_n=20,
        title="LightGBM Feature Importance (Top 20)",
        savepath=os.path.join(out_dir, "lgb_feature_importance.png"),
    )
    plot_feature_importance_generic(
        models["XGBoost"],
        feature_names,
        top_n=20,
        title="XGBoost Feature Importance (Top 20)",
        savepath=os.path.join(out_dir, "xgb_feature_importance.png"),
    )
    plot_feature_importance_generic(
        models["CatBoost"],
        feature_names,
        top_n=20,
        title="CatBoost Feature Importance (Top 20)",
        savepath=os.path.join(out_dir, "cat_feature_importance.png"),
    )

    # 5. 콘솔 요약 및 성능 지표 출력
    best = results_df["r2"].idxmax()
    print("\n" + "=" * 80)
    print("훈련 완료! 최종 요약")
    print("=" * 80)
    print(f"총 {len(X_train):,}개 훈련 샘플, {len(X_test):,}개 테스트 샘플")
    print(f"사용된 피처 수: {len(feature_names)}개")
    print(f"최고 성능 모델: {best} (R² = {results_df.loc[best, 'r2']:.4f})")
    print(f"결과 파일 저장 경로: {os.path.abspath(out_dir)}")

    # 성능지표 상세 출력
    print("\n" + "=" * 80)
    print("INCOME MODEL PERFORMANCE METRICS")
    print("=" * 80)
    for model_name in results_df.index:
        print(f"{model_name}:")
        print(f"  R2   : {results_df.loc[model_name, 'r2']:.4f}")
        print(f"  RMSE : {results_df.loc[model_name, 'rmse']:.4f}")
        print(f"  MAE  : {results_df.loc[model_name, 'mae']:.4f}")
        print()


if __name__ == "__main__":
    main()
