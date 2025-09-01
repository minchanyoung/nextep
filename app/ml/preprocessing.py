import pandas as pd
import json

def prepare_prediction_features(user_input, ml_predictor):
    """
    사용자 입력을 받아 각 모델별로 필요한 정확한 피처를 생성합니다.
    """
    from flask import current_app
    
    # ML 리소스에서 피처 정의 가져오기
    ml_resources = current_app.extensions.get('ml_resources', {})
    income_features = ml_resources.get('income_features', [])
    satis_features = ml_resources.get('satis_features', [])
    
    # 기본 시나리오 3개(현직, 직업A, 직업B) 생성
    scenarios = []
    job_categories = [
        user_input["current_job_category"],
        user_input["job_A_category"],
        user_input["job_B_category"]
    ]

    for i, job_cat_code in enumerate(job_categories):
        scenario = {
            "age": int(user_input["age"]),
            "gender": int(user_input["gender"]),
            "education": int(user_input["education"]),
            "monthly_income": int(user_input["monthly_income"]),
            "job_satisfaction": int(user_input["job_satisfaction"]),
            "job_category": int(job_cat_code)
        }
        scenarios.append(scenario)

    df = pd.DataFrame(scenarios)

    # 9개 만족도 요인 추가
    satis_cols = [f'satis_{cat}' for cat in ['wage', 'stability', 'growth', 'task_content', 'work_env', 'work_time', 'communication', 'fair_eval', 'welfare']]
    for factor in satis_cols:
        df[factor] = int(user_input.get(factor, 3))
    
    # prev_job_satisfaction 추가 (두 모델 모두 필요)
    df['prev_job_satisfaction'] = df['job_satisfaction']

    # 그룹 통계 피처 생성
    if ml_predictor.job_category_stats is not None:
        stats = ml_predictor.job_category_stats
        df = df.merge(stats, on='job_category', how='left')
        
        # 이직 시나리오의 소득을 해당 직업군 평균으로 업데이트
        is_transfer = df['job_category'] != int(user_input["current_job_category"])
        df.loc[is_transfer, 'monthly_income'] = df.loc[is_transfer, 'job_category_income_avg'].astype(float)
        
    # === 소득 모델을 위한 고급 시계열 피처 생성 (income_model_trainer.py와 동일) ===
    
    # satisfaction_change_score는 예측 단계에서는 0으로 설정 (실제로는 타겟 변수)
    df['satisfaction_change_score'] = 0.0
    
    # 시계열 피처들 (예측 시에는 현재값으로 근사)
    df['income_lag1'] = df['monthly_income']  # 1년 전 소득 (현재값으로 근사)
    df['income_lag2'] = df['monthly_income']  # 2년 전 소득 (현재값으로 근사)
    df['income_trend'] = 0.0  # 소득 추세 (예측시에는 0)
    df['prev_income_change'] = 0.0  # 이전 소득 변화 (예측시에는 0)
    df['income_volatility'] = 0.0  # 소득 변동성 (예측시에는 0)
    df['satisfaction_trend'] = 0.0  # 만족도 추세 (예측시에는 0)
    df['satisfaction_volatility'] = 0.3  # 만족도 변동성 (기본값)
    
    # 경력 관련 피처
    df['career_length'] = (df['age'] - 18).clip(lower=1)  # 경력 연수 (고졸 기준)
    df['job_stability'] = 1  # 직장 안정성 (예측시에는 1)
    
    # 경제 사이클 (연도별 - 2023년 기준)
    df['economic_cycle'] = 0.5
    
    # 소득-연령 비율
    # 소득-연령 비율 (23세 이하 보정)
    age_adjusted = df['age'].clip(lower=25)  # 최소 25세로 보정
    df['income_age_ratio'] = df['monthly_income'] / age_adjusted
    
    # 소득 정점 연령대 (40-55세)
    df['peak_earning_years'] = ((df['age'] >= 40) & (df['age'] <= 55)).astype(int)
    
    # 교육 투자 수익률
    df['education_roi'] = df['monthly_income'] / (df['education'] + 1)
    
    # 만족도-소득 불일치 지표
    if ml_predictor.job_category_stats is not None:
        income_norm = df['monthly_income'] / df['job_category_income_avg'] * 3
        df['satisfaction_income_gap'] = df['satis_wage'] - income_norm
    else:
        df['satisfaction_income_gap'] = 0
    
    # 직업 변화 신호 
    df['job_category_change'] = 0  # 예측시에는 0
    df['potential_promotion'] = ((df['job_satisfaction'] > 3) & (df['satis_growth'] >= 4)).astype(int)
    
    # 경력 단계
    df['career_stage'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], labels=[1, 2, 3, 4, 5]).astype(int)
    
    # 동료 대비 소득 (예측시 근사)
    if ml_predictor.job_category_stats is not None:
        df['income_vs_peers'] = df['monthly_income'] - df['job_category_income_avg']
    else:
        df['income_vs_peers'] = 0

    # === 만족도 모델 전용 피처 생성 ===
    df['satisfaction_lag_1'] = df['job_satisfaction']
    df['income_lag_1'] = df['monthly_income']
    df['satisfaction_roll_mean_3'] = df['job_satisfaction']
    df['satisfaction_roll_std_3'] = 0  # 단일 값이므로 변동성은 0
    df['income_roll_mean_3'] = df['monthly_income']
    df['age_x_education'] = df['age'] * df['education']
    df['income_x_satisfaction_lag_1'] = df['income_lag_1'] * df['satisfaction_lag_1']
    
    if ml_predictor.job_category_stats is not None:
        df['job_cat_income_avg'] = df['job_category_income_avg']
        df['job_cat_satis_avg'] = df['job_category_satisfaction_avg']
    else:
        df['job_cat_income_avg'] = df['monthly_income'] * 1.05
        df['job_cat_satis_avg'] = df['job_satisfaction']

    # 공통 만족도 통계 피처
    satis_scores = df[satis_cols].fillna(3)
    df['satisfaction_mean'] = satis_scores.mean(axis=1)
    df['satisfaction_std'] = satis_scores.std(axis=1).fillna(0)
    df['satisfaction_range'] = satis_scores.max(axis=1) - satis_scores.min(axis=1)

    # === 피처 값 범위 제한 (실제 훈련 데이터 범위로) ===
    
    # 예측 정확도를 위해 피처 범위를 더 넓게 설정
    feature_limits = {
        'income_age_ratio': (3, 15),      # 범위 축소로 극단값 방지
        'education_roi': (5, 1000),       # 극단적 경우도 허용
        'monthly_income': (10, 2000),     # 소득 범위 확대
        'satisfaction_income_gap': (-10, 10), # 범위 확대
        'income_vs_peers': (-1000, 1000), # 동료 대비 차이 확대
        'income_x_satisfaction_lag_1': (20, 10000),  # 상호작용 범위 확대
        'monthly_income_x_job_category': (10, 18000), # 직업별 소득 범위 확대
        'age_x_education': (16, 400),     # 연령*교육 범위 확대
        'career_length': (0, 50),         # 경력 범위 확대
        'income_lag1': (10, 2000),        # lag 피처 범위 확대
        'income_lag2': (10, 2000),        # lag 피처 범위 확대
        'income_roll_mean_3': (10, 2000), # 이동평균 범위 확대
        'job_cat_income_avg': (50, 1500), # 직업별 평균 범위 확대
        'peak_earning_years': (0, 1),     # 바이너리 피처
        'potential_promotion': (0, 1),    # 바이너리 피처
        'career_stage': (1, 5),           # 1~5 범위 유지
        'satisfaction_mean': (1, 5),      # 만족도 평균 1~5
        'satisfaction_std': (0, 3),       # 표준편차 범위 증가
        'satisfaction_range': (0, 4),     # 범위 0~4
    }
    
    # 피처별 클리핑 적용
    for col, (min_val, max_val) in feature_limits.items():
        if col in df.columns:
            original_val = df[col].iloc[0] if len(df) > 0 else 0
            df[col] = df[col].clip(lower=min_val, upper=max_val)
            if original_val != df[col].iloc[0] and abs(original_val) > max_val * 1.1:
                print(f"피처 {col} 클리핑: {original_val:.2f} -> {df[col].iloc[0]:.2f}")
    
    # 추가 안전장치: 모든 수치형 컬럼 극값 제거
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if col not in ['age', 'gender', 'education', 'job_category']:  # 기본 필드는 제외
            # 더 유연한 범위 설정 (99.9% 분위수 기반)
            q01 = df[col].quantile(0.001)  # 0.1% 분위수
            q999 = df[col].quantile(0.999) # 99.9% 분위수
            
            # 극단적 이상치만 제거 (99.8% 데이터 유지)
            if q999 > q01:  # 유효한 범위가 있을 때만
                # 하지만 기본적인 합리성 찴크는 유지
                if col in ['monthly_income', 'income_lag1', 'income_lag2', 'income_roll_mean_3']:
                    lower_bound = max(q01, 1)      # 소득은 최소 1만원
                    upper_bound = min(q999, 5000)  # 소득은 최대 500만원
                elif 'satisfaction' in col.lower():
                    lower_bound = max(q01, 0)      # 만족도는 0 이상
                    upper_bound = min(q999, 5)     # 만족도는 5 이하
                elif 'age' in col.lower():
                    lower_bound = max(q01, 15)     # 연령 관련은 15 이상
                    upper_bound = min(q999, 500)   # 연령 곱셈 500 이하
                else:
                    lower_bound = q01
                    upper_bound = q999
                    
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # 최종 결측값 처리
    df = df.fillna(0)
    
    return df
