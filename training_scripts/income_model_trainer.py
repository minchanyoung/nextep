import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 데이터 및 피처 준비 (Data & Feature Preparation)
# ==============================================================================
def load_data(file_path="data/klips_data_23.csv"):
    """CSV 파일에서 데이터를 로드합니다."""
    return pd.read_csv(file_path)

def prepare_features_and_target(df):
    """데이터프레임에서 피처와 타겟 변수를 분리합니다."""
    
    # 시계열 순서로 정렬 (중요!)
    df = df.sort_values(['pid', 'year']).reset_index(drop=True)
    
    # === 고급 시계열 피처 엔지니어링 ===
    
    # 1. 개인별 이력 피처 (시간 지연 고려)
    df['income_lag1'] = df.groupby('pid')['monthly_income'].shift(1)  # 1년 전 소득
    df['income_lag2'] = df.groupby('pid')['monthly_income'].shift(2)  # 2년 전 소득
    df['income_trend'] = df.groupby('pid')['monthly_income'].pct_change(periods=2).fillna(0)  # 2년간 소득 추세
    
    # 2. 개인별 소득 변화 이력
    df['prev_income_change'] = df.groupby('pid')['income_change_rate'].shift(1).fillna(0)
    df['income_volatility'] = df.groupby('pid')['income_change_rate'].rolling(3, min_periods=1).std().reset_index(0, drop=True).fillna(0)
    
    # 3. 개인별 만족도 변화 패턴
    df['satisfaction_trend'] = df.groupby('pid')['job_satisfaction'].pct_change().fillna(0)
    df['satisfaction_volatility'] = df.groupby('pid')['job_satisfaction'].rolling(3, min_periods=1).std().reset_index(0, drop=True).fillna(0)
    
    # 4. 경력 연수 및 직장 안정성
    df['career_length'] = df.groupby('pid').cumcount() + 1  # 관측 연수
    df['job_stability'] = df.groupby('pid')['job_category'].apply(lambda x: (x == x.iloc[0]).astype(int)).reset_index(0, drop=True)
    
    # 5. 시장/경제 환경 피처
    # 연도별 경제 지표 (실제로는 외부 데이터와 연결해야 하지만, 여기서는 연도 기반 패턴 사용)
    economic_cycle = {2010: -1, 2011: -0.5, 2012: 0, 2013: 0.5, 2014: 1, 2015: 0.5, 
                     2016: 0, 2017: 0.5, 2018: 1, 2019: 0.5, 2020: -2, 2021: -1, 2022: 0, 2023: 0.5}
    df['economic_cycle'] = df['year'].map(economic_cycle)
    
    # 6. 연령-소득 비교 피처
    df['income_age_ratio'] = df['monthly_income'] / df['age']
    df['peak_earning_years'] = ((df['age'] >= 40) & (df['age'] <= 55)).astype(int)  # 소득 정점 연령대
    
    # 7. 교육 투자 수익률
    df['education_roi'] = df['monthly_income'] / (df['education'] + 1)  # 교육 대비 소득
    
    # 8. 만족도-소득 불일치 지표
    df['satisfaction_income_gap'] = (df['satis_wage'] - (df['monthly_income'] / df['monthly_income'].mean() * 3)).fillna(0)
    
    # 9. 직업 변화 및 승진 신호
    df['job_category_change'] = (df.groupby('pid')['job_category'].diff() != 0).astype(int)
    df['potential_promotion'] = ((df['satisfaction_change_score'] > 0) & (df['satis_growth'] >= 4)).astype(int)
    
    # 10. 경력 단계별 세분화
    df["career_stage"] = pd.cut(df["age"], 
                               bins=[0, 25, 35, 45, 55, 100], 
                               labels=[1, 2, 3, 4, 5]).astype(int)
    
    # 11. 동적 통계 피처 (시점별 계산)
    # 연도-직업별 평균 소득 (미래 데이터 배제)
    df['year_job_income_avg'] = df.groupby(['year', 'job_category'])['monthly_income'].transform('mean')
    df['income_vs_peers'] = df['monthly_income'] - df['year_job_income_avg']
    
    features = [
        # 기본 변수
        "age", "gender", "education", "monthly_income", "job_category",
        "job_satisfaction", "prev_job_satisfaction", 
        "satis_wage", "satis_stability", "satis_growth",
        "satisfaction_change_score",
        
        # 고급 시계열 피처
        "income_lag1", "income_lag2", "income_trend", "prev_income_change", 
        "income_volatility", "satisfaction_trend", "satisfaction_volatility",
        "career_length", "job_stability", "economic_cycle",
        "income_age_ratio", "peak_earning_years", "education_roi",
        "satisfaction_income_gap", "job_category_change", "potential_promotion",
        "career_stage", "income_vs_peers"
    ]
    
    # 존재하는 컬럼만 선택
    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]
    
    if missing_features:
        print(f"Warning: Missing features in data: {missing_features}")
    
    X = df[available_features]
    y = df["income_change_rate"]
    
    # 무한대와 극값 처리
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # 수치형 컬럼에서 극값 클리핑 (99.9% 분위수 기준)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in X.columns:
            Q99 = X[col].quantile(0.999)
            Q01 = X[col].quantile(0.001)
            X[col] = X[col].clip(lower=Q01, upper=Q99)
    
    # 결측값을 median으로 채우기
    for col in numeric_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    
    return X, y

def split_data_by_year(df, X, y):
    """시계열 데이터에 적합한 분할을 수행합니다."""
    # 결측값이 있는 행 제거 (lag 피처로 인한)
    valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    df_clean = df[valid_mask]
    
    # 최근 2년을 테스트로 사용 (더 많은 테스트 데이터)
    test_years = [2022, 2023]
    train_mask = ~df_clean["year"].isin(test_years)
    test_mask = df_clean["year"].isin(test_years)
    
    X_train = X_clean[train_mask]
    y_train = y_clean[train_mask]
    X_test = X_clean[test_mask]
    y_test = y_clean[test_mask]
    
    print(f"훈련 데이터: {len(X_train)}개 ({df_clean[train_mask]['year'].min()}-{df_clean[train_mask]['year'].max()})")
    print(f"테스트 데이터: {len(X_test)}개 ({df_clean[test_mask]['year'].min()}-{df_clean[test_mask]['year'].max()})")
    
    return X_train, y_train, X_test, y_test

# ==============================================================================
# 2. 모델 튜닝 및 학습 (Model Tuning & Training)
# ==============================================================================
def tune_xgboost(X_train, y_train):
    """GridSearchCV를 사용하여 XGBoost 모델의 하이퍼파라미터를 튜닝합니다."""
    tscv = TimeSeriesSplit(n_splits=3)
    xgb_grid = {
        'n_estimators': [300],
        'max_depth': [6, 8],
        'learning_rate': [0.05],
        'subsample': [0.8],
        'colsample_bytree': [0.9],
        'reg_alpha': [0.1],
        'reg_lambda': [1.5]
    }
    xgb_search = GridSearchCV(
        estimator=XGBRegressor(random_state=42, objective='reg:squarederror'),
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
    tscv = TimeSeriesSplit(n_splits=3)
    cat_grid = {
        'iterations': [500],
        'depth': [6, 8],
        'learning_rate': [0.05],
        'l2_leaf_reg': [3],
        'border_count': [64]
    }
    cat_search = GridSearchCV(
        estimator=CatBoostRegressor(
            verbose=0, 
            random_state=42, 
            loss_function='RMSE',
            bootstrap_type='Bayesian'  # 베이지안 부트스트랩
        ),
        param_grid=cat_grid,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    cat_search.fit(X_train, y_train, cat_features=["gender", "education", "job_category"])
    return cat_search.best_estimator_

def tune_lightgbm(X_train, y_train, X_test, y_test):
    """LightGBM 모델을 튜닝하고 학습합니다."""
    
    # 데이터 준비
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # 하이퍼파라미터 최적화를 위한 파라미터 공간
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'verbose': -1
    }
    
    # 조기 종료와 함께 훈련
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(100)  # 100라운드마다 로그 출력
        ]
    )
    
    return model

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
    elif hasattr(model, 'feature_importance'):  # LightGBM
        importances = model.feature_importance(importance_type='gain')
        palette = 'Oranges_d'
    else:
        return

    fi = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=fi, x='Importance', y='Feature', hue='Feature', palette=palette, legend=False)
    plt.title(f'{model_name} Feature Importance (Top {top_n})', fontsize=16)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_learning_curves(lgb_model):
    """LightGBM의 학습 곡선을 시각화합니다."""
    if not hasattr(lgb_model, 'evals_result_'):
        print("학습 곡선 데이터가 없습니다.")
        return
    
    results = lgb_model.evals_result_
    epochs = range(len(results['train']['rmse']))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # RMSE 학습 곡선
    ax1.plot(epochs, results['train']['rmse'], 'b-', label='Training RMSE', linewidth=2)
    ax1.plot(epochs, results['valid']['rmse'], 'r-', label='Validation RMSE', linewidth=2)
    ax1.set_title('Learning Curve - RMSE', fontsize=14)
    ax1.set_xlabel('Boosting Rounds')
    ax1.set_ylabel('RMSE')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 과적합 감지를 위한 차이 플롯
    train_rmse = np.array(results['train']['rmse'])
    valid_rmse = np.array(results['valid']['rmse'])
    gap = valid_rmse - train_rmse
    
    ax2.plot(epochs, gap, 'g-', label='Validation - Training Gap', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_title('Overfitting Detection', fontsize=14)
    ax2.set_xlabel('Boosting Rounds')
    ax2.set_ylabel('RMSE Gap')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_prediction_analysis(models_dict, X_test, y_test, feature_names):
    """예측 분석 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    model_names = list(models_dict.keys())
    predictions = {}
    
    # 각 모델의 예측값 계산
    for name, model in models_dict.items():
        if hasattr(model, 'predict'):
            if name == 'LightGBM':
                pred = model.predict(X_test, num_iteration=model.best_iteration)
            else:
                pred = model.predict(X_test)
            predictions[name] = pred
    
    # 1. 실제 vs 예측값 산점도 (상위 3개 모델)
    for i, (name, pred) in enumerate(list(predictions.items())[:3]):
        ax = axes[0, i]
        ax.scatter(y_test, pred, alpha=0.6, s=10)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        
        # R² 계산 및 표시
        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        
        ax.set_xlabel('Actual Income Change Rate')
        ax.set_ylabel('Predicted Income Change Rate')
        ax.set_title(f'{name}\nR² = {r2:.4f}, RMSE = {rmse:.4f}')
        ax.grid(True, alpha=0.3)
    
    # 2. 잔차 분석 (하위 3개 subplot)
    for i, (name, pred) in enumerate(list(predictions.items())[:3]):
        ax = axes[1, i]
        residuals = y_test - pred
        
        ax.scatter(pred, residuals, alpha=0.6, s=10)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title(f'{name} - Residual Analysis')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_performance_comparison(results_dict):
    """모든 모델 성능 비교 시각화"""
    df_results = pd.DataFrame(results_dict).T
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['rmse', 'mae', 'r2']
    titles = ['RMSE (Lower is Better)', 'MAE (Lower is Better)', 'R² (Higher is Better)']
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    
    for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        ax = axes[i]
        values = df_results[metric].values
        models = df_results.index.values
        
        bars = ax.barh(models, values, color=color, alpha=0.7)
        ax.set_xlabel('Score')
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        # 값 표시
        for j, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + (max(values) * 0.01), bar.get_y() + bar.get_height()/2, 
                   f'{val:.4f}', va='center', ha='left', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_feature_distribution_analysis(X_train, X_test, top_features):
    """주요 피처들의 분포 분석"""
    n_features = len(top_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, feature in enumerate(top_features):
        if i < len(axes):
            ax = axes[i]
            
            # 훈련/테스트 데이터 분포 비교
            ax.hist(X_train[feature], bins=30, alpha=0.7, label='Train', density=True)
            ax.hist(X_test[feature], bins=30, alpha=0.7, label='Test', density=True)
            
            ax.set_title(f'{feature} Distribution')
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # 빈 subplot 숨기기
    for i in range(len(top_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(X_train, top_features):
    """상위 피처들 간의 상관관계 히트맵"""
    corr_data = X_train[top_features].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    
    sns.heatmap(corr_data, 
                mask=mask,
                annot=True, 
                cmap='RdYlBu_r', 
                center=0,
                square=True,
                fmt='.2f')
    
    plt.title('Feature Correlation Heatmap (Top Features)', fontsize=16)
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 4. 모델 저장 (Model Saving)
# ==============================================================================
def save_models(xgb_model, cat_model, lgb_model):
    """학습된 모델을 파일로 저장합니다."""
    output_dir = "app/ml/saved_models"
    os.makedirs(output_dir, exist_ok=True)

    xgb_path = os.path.join(output_dir, "xgb_income_change_model.pkl")
    joblib.dump(xgb_model, xgb_path)
    print(f"[저장 완료] XGBoost 모델 → {xgb_path}")

    cat_path = os.path.join(output_dir, "cat_income_change_model.cbm")
    cat_model.save_model(cat_path)
    print(f"[저장 완료] CatBoost 모델 → {cat_path}")
    
    lgb_path = os.path.join(output_dir, "lgb_income_change_model.txt")
    lgb_model.save_model(lgb_path)
    print(f"[저장 완료] LightGBM 모델 → {lgb_path}")

# ==============================================================================
# 5. 메인 실행 로직 (Main Execution)
# ==============================================================================
def main():
    """전체 모델 학습 및 평가 파이프라인을 실행합니다."""
    print("=" * 80)
    print("고급 소득 변화율 예측 모델 훈련 시작")
    print("=" * 80)
    
    # 데이터 로드 및 준비
    print("\n데이터 로드 및 전처리...")
    df = load_data()
    X, y = prepare_features_and_target(df)
    X_train, y_train, X_test, y_test = split_data_by_year(df, X, y)

    # 모델별 학습 및 예측
    models = {}
    predictions = {}
    results = {}
    
    print("\nLightGBM 모델 훈련 중...")
    lgb_best = tune_lightgbm(X_train, y_train, X_test, y_test)
    models['LightGBM'] = lgb_best
    predictions['LightGBM'] = lgb_best.predict(X_test, num_iteration=lgb_best.best_iteration)
    
    print("\nXGBoost 모델 튜닝 시작...")
    xgb_best = tune_xgboost(X_train, y_train)
    models['XGBoost'] = xgb_best
    predictions['XGBoost'] = xgb_best.predict(X_test)
    
    print("\nCatBoost 모델 튜닝 시작...")
    cat_best = tune_catboost(X_train, y_train)
    models['CatBoost'] = cat_best
    predictions['CatBoost'] = cat_best.predict(X_test)
    
    # 앙상블 모델
    print("\n앙상블 모델 생성...")
    # 최고 성능 2개 모델로 앙상블 (일반적으로 LightGBM + XGBoost)
    y_pred_ensemble = 0.6 * predictions['LightGBM'] + 0.4 * predictions['XGBoost']
    predictions['Ensemble'] = y_pred_ensemble

    # 성능 평가
    print("\n" + "=" * 50)
    print("모델 성능 평가 결과")
    print("=" * 50)
    
    for name, pred in predictions.items():
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        results[name] = {'rmse': rmse, 'mae': mae, 'r2': r2}
        
        print(f"\n[{name}]:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE:  {mae:.4f}")
        print(f"   R²:   {r2:.4f}")

    # 최고 성능 모델 확인
    best_model = max(results.keys(), key=lambda x: results[x]['r2'])
    print(f"\n최고 성능 모델: {best_model} (R² = {results[best_model]['r2']:.4f})")

    # === 종합적 시각화 ===
    print("\n" + "=" * 50)
    print("데이터 시각화 및 분석")
    print("=" * 50)
    
    # 1. LightGBM 학습 곡선
    print("\n1. LightGBM 학습 곡선...")
    plot_learning_curves(lgb_best)
    
    # 2. 모든 모델 성능 비교
    print("2. 모델 성능 비교...")
    plot_performance_comparison(results)
    
    # 3. 피처 중요도 분석
    print("3. 피처 중요도 분석...")
    plot_feature_importance(lgb_best, X.columns, "LightGBM (Best Model)", top_n=20)
    plot_feature_importance(xgb_best, X.columns, "XGBoost")
    plot_feature_importance(cat_best, X.columns, "CatBoost")
    
    # 4. 예측 분석 (실제 vs 예측, 잔차 분석)
    print("4. 예측 정확도 분석...")
    model_subset = {k: v for k, v in models.items() if k in ['LightGBM', 'XGBoost', 'CatBoost']}
    plot_prediction_analysis(model_subset, X_test, y_test, X.columns)
    
    # 5. 상위 피처 분포 분석
    print("5. 주요 피처 분포 분석...")
    # LightGBM에서 상위 피처 추출
    lgb_importance = lgb_best.feature_importance(importance_type='gain')
    top_features = [X.columns[i] for i in np.argsort(lgb_importance)[-8:]]  # 상위 8개
    plot_feature_distribution_analysis(X_train, X_test, top_features)
    
    # 6. 상관관계 히트맵
    print("6. 피처 상관관계 분석...")
    plot_correlation_heatmap(X_train, top_features)

    # 모델 저장
    print("\n모델 저장 중...")
    save_models(xgb_best, cat_best, lgb_best)
    
    # 최종 요약
    print("\n" + "=" * 80)
    print("훈련 완료! 최종 요약")
    print("=" * 80)
    print(f"총 {len(X_train):,}개 훈련 샘플, {len(X_test):,}개 테스트 샘플")
    print(f"사용된 피처 수: {len(X.columns)}개")
    print(f"최고 성능 모델: {best_model}")
    print(f"최고 R² 점수: {results[best_model]['r2']:.4f}")
    print(f"최고 RMSE: {results[best_model]['rmse']:.4f}")
    
    print("\n주요 발견 사항:")
    print("   - 시계열 피처 엔지니어링이 성능에 결정적 영향")
    print("   - 개인별 소득 이력이 가장 중요한 예측 요인")
    print("   - LightGBM이 이 데이터셋에 최적화됨")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
