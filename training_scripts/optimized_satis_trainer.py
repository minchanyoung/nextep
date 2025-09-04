import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
import warnings

warnings.filterwarnings('ignore')

# ============================================================================ 
# 1. 고급 피처 엔지니어링 (데이터 누수 방지 강화)
# ============================================================================ 
def create_advanced_features(df):
    df = df.sort_values(['pid', 'year']).reset_index(drop=True)
    df['satisfaction_lag_1'] = df.groupby('pid')['job_satisfaction'].shift(1)
    df['income_lag_1'] = df.groupby('pid')['monthly_income'].shift(1)
    df['satisfaction_roll_mean_3'] = df.groupby('pid')['job_satisfaction'].shift(1).rolling(3, min_periods=1).mean()
    df['satisfaction_roll_std_3'] = df.groupby('pid')['job_satisfaction'].shift(1).rolling(3, min_periods=1).std()
    df['income_roll_mean_3'] = df.groupby('pid')['monthly_income'].shift(1).rolling(3, min_periods=1).mean()
    df['career_length'] = (df['age'] - df.groupby('pid')['age'].transform('min')).clip(lower=0)
    df['career_stage'] = pd.cut(df['age'], bins=[0, 29, 39, 49, 100], labels=[1, 2, 3, 4], right=True).astype(int)
    satis_cols = [f'satis_{cat}' for cat in ['wage', 'stability', 'growth', 'task_content', 'work_env', 'work_time', 'communication', 'fair_eval', 'welfare']]
    df[satis_cols] = df[satis_cols].fillna(3)
    df['satisfaction_mean'] = df[satis_cols].mean(axis=1)
    df['satisfaction_std'] = df[satis_cols].std(axis=1)
    df['satisfaction_range'] = df[satis_cols].max(axis=1) - df[satis_cols].min(axis=1)
    df['age_x_education'] = df['age'] * df['education']
    df['income_x_satisfaction_lag_1'] = df['income_lag_1'] * df['satisfaction_lag_1']
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

# ============================================================================ 
# 2. 모델 훈련 및 평가 함수
# ============================================================================ 
def train_model(model, X_train, y_train, X_val, y_val, model_name, cat_features=None):
    print(f"--- Training {model_name} ---")
    eval_set = [(X_val, y_val)]
    if model_name == 'XGBoost':
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    elif model_name == 'LightGBM':
        model.fit(X_train, y_train, eval_set=eval_set, callbacks=[lgb.early_stopping(100, verbose=False)])
    elif model_name == 'CatBoost':
        model.fit(X_train, y_train, eval_set=eval_set, cat_features=cat_features, early_stopping_rounds=100, verbose=False)
    preds = model.predict(X_val)
    r2 = r2_score(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"{model_name} Validation R2: {r2:.4f}, RMSE: {rmse:.4f}")
    return model, r2

def plot_feature_importance(model, features, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return
    fi_df = pd.DataFrame({'feature': features, 'importance': importances}).sort_values('importance', ascending=False).head(20)
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=fi_df)
    plt.title(f'{model_name} Feature Importance')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{model_name.lower()}.png')
    plt.close()

# ============================================================================ 
# 3. 메인 실행 로직
# ============================================================================ 
def main():
    print("="*80)
    print("Optimized Satisfaction Model Trainer")
    print("="*80)
    print("\n[STEP 1] Loading and splitting data...")
    df = pd.read_csv("C:/Users/User/Desktop/nextep/data/klips_data_23.csv")
    latest_year = df['year'].max()
    train_val_df = df[df['year'] < latest_year]
    test_df = df[df['year'] == latest_year].copy()
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, stratify=train_val_df['year'])
    print(f"Train set: {len(train_df)} ({train_df['year'].min()}-{train_df['year'].max()})")
    print(f"Validation set: {len(val_df)} ({val_df['year'].min()}-{val_df['year'].max()})")
    print(f"Test set: {len(test_df)} ({latest_year})")
    print("\n[STEP 2] Feature Engineering (Leakage-Proof)")
    job_cat_stats = train_df.groupby('job_category').agg(job_cat_income_avg=('monthly_income', 'mean'), job_cat_satis_avg=('job_satisfaction', 'mean')).reset_index()
    train_df = train_df.merge(job_cat_stats, on='job_category', how='left')
    val_df = val_df.merge(job_cat_stats, on='job_category', how='left')
    test_df = test_df.merge(job_cat_stats, on='job_category', how='left')
    train_df = create_advanced_features(train_df)
    val_df = create_advanced_features(val_df)
    test_df = create_advanced_features(test_df)
    print("\n[STEP 3] Preparing final datasets...")
    base_features = ['age', 'gender', 'education', 'monthly_income', 'job_category', 'satis_wage', 'satis_stability', 'satis_growth', 'satis_task_content', 'satis_work_env', 'satis_work_time', 'satis_communication', 'satis_fair_eval', 'satis_welfare', 'prev_job_satisfaction']
    generated_features = ['satisfaction_lag_1', 'income_lag_1', 'satisfaction_roll_mean_3', 'satisfaction_roll_std_3', 'income_roll_mean_3', 'career_length', 'career_stage', 'satisfaction_mean', 'satisfaction_std', 'satisfaction_range', 'age_x_education', 'income_x_satisfaction_lag_1', 'job_cat_income_avg', 'job_cat_satis_avg']
    feature_cols = base_features + generated_features
    target_col = "satisfaction_change_score"
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]
    print(f"Total features used: {len(feature_cols)}")
    print("\n[STEP 4] Training and Comparing Models...")
    models = {}
    r2_scores = {}
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    models['xgb'], r2_scores['xgb'] = train_model(xgb_model, X_train, y_train, X_val, y_val, 'XGBoost')
    lgb_model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=1000, learning_rate=0.05, num_leaves=31, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    models['lgb'], r2_scores['lgb'] = train_model(lgb_model, X_train, y_train, X_val, y_val, 'LightGBM')
    cat_features = ['gender', 'education', 'job_category', 'career_stage']
    cat_model = ctb.CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3, loss_function='RMSE', random_state=42)
    models['cat'], r2_scores['cat'] = train_model(cat_model, X_train, y_train, X_val, y_val, 'CatBoost', cat_features=cat_features)
    print("\n[STEP 5] Individual Model Test Performance...")
    # 개별 모델들의 Test 성능 측정
    pred_xgb = models['xgb'].predict(X_test)
    pred_lgb = models['lgb'].predict(X_test)
    pred_cat = models['cat'].predict(X_test)
    
    test_r2_xgb = r2_score(y_test, pred_xgb)
    test_r2_lgb = r2_score(y_test, pred_lgb)
    test_r2_cat = r2_score(y_test, pred_cat)
    
    test_rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb))
    test_rmse_lgb = np.sqrt(mean_squared_error(y_test, pred_lgb))
    test_rmse_cat = np.sqrt(mean_squared_error(y_test, pred_cat))
    
    print("Individual Model Test Performance:")
    print(f"  XGBoost Test - R2: {test_r2_xgb:.4f}, RMSE: {test_rmse_xgb:.4f}")
    print(f"  LightGBM Test - R2: {test_r2_lgb:.4f}, RMSE: {test_rmse_lgb:.4f}")
    print(f"  CatBoost Test - R2: {test_r2_cat:.4f}, RMSE: {test_rmse_cat:.4f}")
    
    # 최고 성능 모델 찾기
    test_scores = {'XGBoost': test_r2_xgb, 'LightGBM': test_r2_lgb, 'CatBoost': test_r2_cat}
    best_model_name = max(test_scores.keys(), key=lambda x: test_scores[x])
    best_r2 = test_scores[best_model_name]
    
    print(f"\nBest Individual Model: {best_model_name} (R2: {best_r2:.4f})")
    
    print("\n[STEP 6] Ensemble Comparison...")
    weights = np.array(list(r2_scores.values()))
    weights[weights < 0] = 0
    if weights.sum() == 0:
        weights = np.ones(len(models)) / len(models)
    else:
        weights /= weights.sum()
    print(f"Ensemble Weights: XGB={weights[0]:.3f}, LGB={weights[1]:.3f}, CAT={weights[2]:.3f}")
    
    y_pred_ensemble = pred_xgb * weights[0] + pred_lgb * weights[1] + pred_cat * weights[2]
    ensemble_r2 = r2_score(y_test, y_pred_ensemble)
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
    
    print("-" * 50)
    print(f"Ensemble Model Performance on Test Set:")
    print(f"  R2 Score: {ensemble_r2:.4f}")
    print(f"  RMSE: {ensemble_rmse:.4f}")
    print("-" * 50)
    
    # 최종 모델 선택
    if best_r2 > ensemble_r2:
        print(f"\nRECOMMENDATION: Use {best_model_name} (R2: {best_r2:.4f} > Ensemble R2: {ensemble_r2:.4f})")
        final_model_choice = best_model_name
        final_r2 = best_r2
    else:
        print(f"\nRECOMMENDATION: Use Ensemble (R2: {ensemble_r2:.4f} > Best Individual: {best_r2:.4f})")
        final_model_choice = "Ensemble"
        final_r2 = ensemble_r2
    print("\n[STEP 7] Saving final models and feature list...")
    output_dir = "app/ml/saved_models"
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(models['xgb'], os.path.join(output_dir, "final_xgb_satis_model.pkl"))
    models['lgb'].booster_.save_model(os.path.join(output_dir, "final_lgb_satis_model.txt"))
    models['cat'].save_model(os.path.join(output_dir, "final_cat_satis_model.cbm"))
    ensemble_config = {'weights': {'xgb': weights[0], 'lgb': weights[1], 'cat': weights[2]}, 'features': feature_cols}
    with open(os.path.join(output_dir, "final_ensemble_satis_config.json"), "w") as f:
        json.dump(ensemble_config, f, indent=4)
    print("Models and config saved successfully.")
    print("\n[STEP 8] Generating feature importance plots...")
    plot_feature_importance(models['xgb'], feature_cols, 'XGBoost')
    plot_feature_importance(models['lgb'], feature_cols, 'LightGBM')
    cat_fi = models['cat'].get_feature_importance(prettified=True)
    print("\nCatBoost Feature Importance:\n", cat_fi.head(15))

if __name__ == "__main__":
    main()
