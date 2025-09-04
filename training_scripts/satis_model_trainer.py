# -*- coding: utf-8 -*-
"""
optimized_satis_trainer_like_income.py

- 목표(Target): satisfaction_change_score (만족도 변화량)
- 분할(네 기존 로직 유지): Test = 최신 연도(연도 max), Train/Val = 나머지 8:2 (연도 기준 stratify)
- 누수 방지: 직군 통계(job_category 별 평균) 등은 'train'에서만 산출 후 val/test에 merge
- 모델: XGBoost / LightGBM / CatBoost
- 비교: 각 모델 Test 성능(R2, RMSE) + 가중 앙상블 (val R2 기반)
- 산출물: 저장된 모델 + 앙상블 설정(JSON) + 중요도 PNG
"""

import os, json, warnings
warnings.filterwarnings('ignore')

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

# =============================================================================
# 1) 고급 피처 엔지니어링 (네 기존 로직 유지)
# =============================================================================
def create_advanced_features_satis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['pid', 'year']).reset_index(drop=True)

    # 시차/롤링 (모두 shift(1) 기준으로 누수 방지)
    df['satisfaction_lag_1'] = df.groupby('pid')['job_satisfaction'].shift(1)
    df['income_lag_1']       = df.groupby('pid')['monthly_income'].shift(1)
    df['satisfaction_roll_mean_3'] = df.groupby('pid')['job_satisfaction'].shift(1).rolling(3, min_periods=1).mean()
    df['satisfaction_roll_std_3']  = df.groupby('pid')['job_satisfaction'].shift(1).rolling(3, min_periods=1).std()
    df['income_roll_mean_3']       = df.groupby('pid')['monthly_income'].shift(1).rolling(3, min_periods=1).mean()

    # 경력/단계
    df['career_length'] = (df['age'] - df.groupby('pid')['age'].transform('min')).clip(lower=0)
    df['career_stage']  = pd.cut(df['age'], bins=[0, 29, 39, 49, 100],
                                 labels=[1, 2, 3, 4], right=True).astype(int)

    # 만족도 서브스케일 기본값 3점
    satis_cols = [f'satis_{c}' for c in
                  ['wage','stability','growth','task_content','work_env',
                   'work_time','communication','fair_eval','welfare']]
    df[satis_cols] = df[satis_cols].fillna(3)
    df['satisfaction_mean']  = df[satis_cols].mean(axis=1)
    df['satisfaction_std']   = df[satis_cols].std(axis=1)
    df['satisfaction_range'] = df[satis_cols].max(axis=1) - df[satis_cols].min(axis=1)

    # 상호작용
    df['age_x_education']            = df['age'] * df['education']
    df['income_x_satisfaction_lag_1'] = df['income_lag_1'] * df['satisfaction_lag_1']

    # 결측 보간(앞/뒤)
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

# =============================================================================
# 2) 분할 + 누수 방지용 통계 피처 (네 기존 전략 유지)
#    - Test: 최신 연도(연도 max)
#    - Train/Val: 나머지에서 8:2, stratify by year
# =============================================================================
def split_and_add_stats(df: pd.DataFrame):
    latest_year = df['year'].max()
    train_val_df = df[df['year'] < latest_year].copy()
    test_df      = df[df['year'] == latest_year].copy()

    # 8:2 (연도 stratify) — 네 스크립트 유지
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.2, random_state=42, stratify=train_val_df['year']
    )

    print(f"Train set: {len(train_df)} ({int(train_df['year'].min())}-{int(train_df['year'].max())})")
    print(f"Validation set: {len(val_df)} ({int(val_df['year'].min())}-{int(val_df['year'].max())})")
    print(f"Test set: {len(test_df)} (year={int(latest_year)})")

    # 직군 통계(훈련에서만 계산 → val/test로 merge)
    job_cat_stats = (train_df.groupby('job_category')
                     .agg(job_cat_income_avg=('monthly_income','mean'),
                          job_cat_satis_avg=('job_satisfaction','mean'))
                     ).reset_index()

    def add_stats(d):
        return d.merge(job_cat_stats, on='job_category', how='left')

    train_df = add_stats(train_df)
    val_df   = add_stats(val_df)
    test_df  = add_stats(test_df)

    # 파생은 split 이후에 각각 호출 (누수 최소화)
    train_df = create_advanced_features_satis(train_df)
    val_df   = create_advanced_features_satis(val_df)
    test_df  = create_advanced_features_satis(test_df)

    return train_df, val_df, test_df, latest_year

# =============================================================================
# 3) 학습/평가 & 시각화 유틸 (소득 스크립트 형식)
# =============================================================================
def train_one(model, X_tr, y_tr, X_val, y_val, name, cat_features_idx=None):
    print(f"--- Training {name} ---")
    if name == "XGBoost":
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    elif name == "LightGBM":
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    elif name == "CatBoost":
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                  cat_features=cat_features_idx, early_stopping_rounds=100, verbose=False)

    pred_val = model.predict(X_val)
    r2   = r2_score(y_val, pred_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred_val))
    mae  = mean_absolute_error(y_val, pred_val)
    print(f"{name} Validation -> R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return model, r2

def plot_feature_importance(model, features, model_name, save_dir):
    if not hasattr(model, 'feature_importances_'):
        return
    imp = model.feature_importances_
    fi = (pd.DataFrame({'feature': features, 'importance': imp})
            .sort_values('importance', ascending=False)
            .head(20))
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=fi, orient='h')
    plt.title(f'{model_name} Feature Importance (Top 20)')
    plt.tight_layout()
    path = os.path.join(save_dir, f'feature_importance_{model_name.lower()}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[저장] {path}")

# =============================================================================
# 4) 메인: 학습 → 평가 → 저장 (소득 스크립트 포맷)
# =============================================================================
def main():
    print("="*80)
    print("Optimized Satisfaction-Change Model Trainer (income-style format)")
    print("="*80)

    # 경로/출력
    data_path = "C:/Users/User/Desktop/nextep/data/klips_data_23.csv"
    save_dir  = "app/ml/saved_models"
    os.makedirs(save_dir, exist_ok=True)

    # 데이터 로드 & 분할/파생
    print("\n[STEP 1] Loading & splitting ...")
    df = pd.read_csv(data_path)
    train_df, val_df, test_df, latest_year = split_and_add_stats(df)

    # Feature/Target
    base_features = [
        'age','gender','education','monthly_income','job_category',
        'satis_wage','satis_stability','satis_growth',
        'satis_task_content','satis_work_env','satis_work_time',
        'satis_communication','satis_fair_eval','satis_welfare',
        'prev_job_satisfaction'
    ]
    gen_features = [
        'satisfaction_lag_1','income_lag_1',
        'satisfaction_roll_mean_3','satisfaction_roll_std_3','income_roll_mean_3',
        'career_length','career_stage',
        'satisfaction_mean','satisfaction_std','satisfaction_range',
        'age_x_education','income_x_satisfaction_lag_1',
        'job_cat_income_avg','job_cat_satis_avg'
    ]
    feature_cols = [f for f in base_features + gen_features if f in train_df.columns]
    target_col   = "satisfaction_change_score"

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val,   y_val   = val_df[feature_cols],   val_df[target_col]
    X_test,  y_test  = test_df[feature_cols],  test_df[target_col]
    print(f"Total features used: {len(feature_cols)}")

    # CatBoost용 카테고리 컬럼 인덱스
    cat_cols = [c for c in ['gender','education','job_category','career_stage'] if c in feature_cols]
    cat_idx  = [feature_cols.index(c) for c in cat_cols]

    # 모델 정의 (네 파라미터 유지)
    models = {}
    val_r2 = {}

    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1
    )
    lgb_model = lgb.LGBMRegressor(
        objective='regression_l1',
        n_estimators=1000, learning_rate=0.05, num_leaves=31, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1
    )
    cat_model = ctb.CatBoostRegressor(
        iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function='RMSE', random_state=42, allow_writing_files=False
    )

    print("\n[STEP 2] Training models ...")
    models['xgb'], val_r2['xgb'] = train_one(xgb_model, X_train, y_train, X_val, y_val, 'XGBoost')
    models['lgb'], val_r2['lgb'] = train_one(lgb_model, X_train, y_train, X_val, y_val, 'LightGBM')
    models['cat'], val_r2['cat'] = train_one(cat_model, X_train, y_train, X_val, y_val, 'CatBoost', cat_features_idx=cat_idx)

    # Test 성능
    print("\n[STEP 3] Test performance ...")
    pred_xgb = models['xgb'].predict(X_test)
    pred_lgb = models['lgb'].predict(X_test)
    pred_cat = models['cat'].predict(X_test)

    def scores(y_true, pred):
        return (r2_score(y_true, pred),
                np.sqrt(mean_squared_error(y_true, pred)),
                mean_absolute_error(y_true, pred))

    r2_xgb, rmse_xgb, mae_xgb = scores(y_test, pred_xgb)
    r2_lgb, rmse_lgb, mae_lgb = scores(y_test, pred_lgb)
    r2_cat, rmse_cat, mae_cat = scores(y_test, pred_cat)

    print(f"  XGBoost  -> R2: {r2_xgb:.4f}, RMSE: {rmse_xgb:.4f}, MAE: {mae_xgb:.4f}")
    print(f"  LightGBM -> R2: {r2_lgb:.4f}, RMSE: {rmse_lgb:.4f}, MAE: {mae_lgb:.4f}")
    print(f"  CatBoost -> R2: {r2_cat:.4f}, RMSE: {rmse_cat:.4f}, MAE: {mae_cat:.4f}")

    # 앙상블(Val R2 가중)
    w = np.array([val_r2['xgb'], val_r2['lgb'], val_r2['cat']], dtype=float)
    w[w < 0] = 0
    w = w / w.sum() if w.sum() > 0 else np.array([1/3, 1/3, 1/3])
    print(f"\nEnsemble Weights: XGB={w[0]:.3f}, LGB={w[1]:.3f}, CAT={w[2]:.3f}")

    pred_ens = pred_xgb * w[0] + pred_lgb * w[1] + pred_cat * w[2]
    r2_ens, rmse_ens, mae_ens = scores(y_test, pred_ens)
    print("-"*50)
    print(f"Ensemble -> R2: {r2_ens:.4f}, RMSE: {rmse_ens:.4f}, MAE: {mae_ens:.4f}")
    print("-"*50)

    # 최종 추천
    indiv = {'XGBoost': r2_xgb, 'LightGBM': r2_lgb, 'CatBoost': r2_cat}
    best_name = max(indiv, key=indiv.get)
    if indiv[best_name] >= r2_ens:
        final_choice, final_r2 = best_name, indiv[best_name]
        print(f"\nRECOMMENDATION: Use {final_choice} (R2 {final_r2:.4f} ≥ Ensemble {r2_ens:.4f})")
    else:
        final_choice, final_r2 = "Ensemble", r2_ens
        print(f"\nRECOMMENDATION: Use Ensemble (R2 {r2_ens:.4f} > Best Individual {indiv[best_name]:.4f})")

    # 저장 (소득 스크립트와 동일 포맷)
    print("\n[STEP 4] Saving models & configs & feature importance ...")
    joblib.dump(models['xgb'], os.path.join(save_dir, "final_xgb_satis_model.pkl"))
    models['lgb'].booster_.save_model(os.path.join(save_dir, "final_lgb_satis_model.txt"))
    models['cat'].save_model(os.path.join(save_dir, "final_cat_satis_model.cbm"))

    ens_cfg = {
        'weights': {'xgb': float(w[0]), 'lgb': float(w[1]), 'cat': float(w[2])},
        'features': feature_cols,
        'split': {'test_year': int(latest_year), 'val_ratio': 0.2, 'stratify': 'year'}
    }
    with open(os.path.join(save_dir, "final_ensemble_satis_config.json"), "w") as f:
        json.dump(ens_cfg, f, indent=2)
    print("[저장] 모델 및 앙상블 설정")

    # 중요도 PNG
    plot_feature_importance(models['xgb'], feature_cols, 'XGBoost', save_dir)
    plot_feature_importance(models['lgb'], feature_cols, 'LightGBM', save_dir)
    # CatBoost 중요도 표는 콘솔에 일부 출력
    cat_fi = models['cat'].get_feature_importance(prettified=True)
    print("\nCatBoost Feature Importance (Top 15):\n", cat_fi.head(15))

    # 요약
    print("\n" + "="*80)
    print("Training Done. Summary (Satisfaction)")
    print("="*80)
    print(f"Features used: {len(feature_cols)}")
    print(f"Final choice: {final_choice}  (R2={final_r2:.4f})")
    print(f"Saved to: {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    main()
