import pandas as pd
import json

def prepare_income_model_features(user_input, ml_predictor):
    """
    소득 모델을 위한 정확한 피처 생성
    """
    from flask import current_app
    
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

    # prev_job_satisfaction 추가
    df['prev_job_satisfaction'] = df['job_satisfaction']

    # 9개 만족도 요인 추가 (소득 모델용 - 일부만 필요)
    df['satis_wage'] = int(user_input.get('satis_wage', 3))
    df['satis_stability'] = int(user_input.get('satis_stability', 3))
    df['satis_growth'] = int(user_input.get('satis_growth', 3))
    
    # 먼저 기본 피처들을 생성한 후 이직 시나리오 조정을 위해 나중에 처리

    # 그룹 통계 피처 생성
    if ml_predictor.job_category_stats is not None:
        stats = ml_predictor.job_category_stats
        df = df.merge(stats, on='job_category', how='left')
        
        # 이직 시나리오의 소득을 점진적으로 조정 (급격한 변화 방지)
        current_income = int(user_input["monthly_income"])
        is_transfer = df['job_category'] != int(user_input["current_job_category"])
        
        for idx in df[is_transfer].index:
            target_income = df.loc[idx, 'job_category_income_avg']
            job_cat = df.loc[idx, 'job_category']
            
            # KLIPS 데이터 분포 기반 현실적 이직 소득 조정
            # 실제 평균 소득변화율: +12.69%, 5-95분위수: -40% ~ +80%
            if job_cat == 1:  # 관리자 - 매우 보수적 상승 (12% 목표, 88% 현재)
                adjusted_income = target_income * 0.12 + current_income * 0.88
            elif job_cat == 2:  # 전문가 - 적당한 상승 (18% 목표, 82% 현재)
                adjusted_income = target_income * 0.18 + current_income * 0.82
            elif job_cat == 3:  # 사무직 - 소폭 조정 (10% 목표, 90% 현재)
                adjusted_income = target_income * 0.10 + current_income * 0.90
            elif job_cat in [4, 5]:  # 서비스/판매직 - 현실적 하락 고려 (30% 목표, 70% 현재)
                adjusted_income = target_income * 0.30 + current_income * 0.70
            else:  # 기타 직업 - 일반적 조정 (25% 목표, 75% 현재)
                adjusted_income = target_income * 0.25 + current_income * 0.75
            
            df.loc[idx, 'monthly_income'] = int(adjusted_income)

    # === 소득 모델 전용 피처 생성 ===
    
    # satisfaction_change_score는 예측 단계에서는 0으로 설정
    df['satisfaction_change_score'] = 0.0
    
    # 시계열 피처들
    df['income_lag1'] = df['monthly_income']  
    df['income_lag2'] = df['monthly_income']  
    df['income_trend'] = 0.0  
    df['prev_income_change'] = 0.0  
    df['income_volatility'] = 0.0  
    df['satisfaction_trend'] = 0.0  
    df['satisfaction_volatility'] = 0.3  
    
    # 경력 관련 피처 (23세 이하 보정)
    df['career_length'] = (df['age'] - 18).clip(lower=1)  # 고졸 기준으로 변경
    df['job_stability'] = 1
    
    # 경제 사이클
    df['economic_cycle'] = 0.5
    
    # 소득-연령 비율 (23세 이하 보정)
    # 나이가 너무 어릴 때 비율이 과도하게 높아지는 것을 방지
    age_adjusted = df['age'].clip(lower=25)  # 최소 25세로 보정
    df['income_age_ratio'] = df['monthly_income'] / age_adjusted
    
    # 소득 정점 연령대
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
    df['job_category_change'] = 0
    df['potential_promotion'] = ((df['job_satisfaction'] > 3) & (df['satis_growth'] >= 4)).astype(int)
    
    # 경력 단계 (23세 이하 세분화)
    df['career_stage'] = pd.cut(df['age'], bins=[0, 23, 28, 35, 45, 55, 100], labels=[1, 2, 3, 4, 5, 6]).fillna(1).astype(int).clip(upper=5)
    
    # 동료 대비 소득
    if ml_predictor.job_category_stats is not None:
        df['income_vs_peers'] = df['monthly_income'] - df['job_category_income_avg']
    else:
        df['income_vs_peers'] = 0

    # 피처 값 범위 제한 (23세 이하 특별 처리)
    feature_limits = {
        'income_age_ratio': (3, 15),  # 범위 축소로 극단값 방지
        'education_roi': (5, 1000),
        'monthly_income': (10, 2000),
        'satisfaction_income_gap': (-10, 10),
        'income_vs_peers': (-1000, 1000),
        'career_length': (0, 50),
        'income_lag1': (10, 2000),
        'income_lag2': (10, 2000),
        'peak_earning_years': (0, 1),
        'potential_promotion': (0, 1),
        'career_stage': (1, 5),
    }
    
    for col, (min_val, max_val) in feature_limits.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=min_val, upper=max_val)
    
    # === 이직 시나리오별 피처 조정 (피처 생성 후) ===
    current_job_cat = int(user_input["current_job_category"])
    
    for idx, row_job_cat in enumerate(df['job_category']):
        if row_job_cat != current_job_cat:  # 이직 시나리오인 경우
            df.loc[idx, 'job_category_change'] = 1
            
            # 직업별로 다른 소득 변화 시그널 생성
            if row_job_cat == 1:  # 관리자 - 승진 가능성
                df.loc[idx, 'potential_promotion'] = 1
                df.loc[idx, 'career_stage'] = min(5, df.loc[idx, 'career_stage'] + 1)
            elif row_job_cat == 2:  # 전문가 - 교육투자 수익률 증가
                df.loc[idx, 'potential_promotion'] = 1
                df.loc[idx, 'education_roi'] = df.loc[idx, 'education_roi'] * 1.15
            elif row_job_cat == 3:  # 사무직 - 안정적
                pass  # 기본 시그널만
            elif row_job_cat in [4, 5]:  # 서비스직, 판매직
                pass  # 기본 시그널만
            else:  # 기타 직종
                pass  # 기본 시그널만

    # 최종 결측값 처리
    df = df.fillna(0)
    
    # categorical 피처들을 정수로 확실히 변환 (CatBoost 오류 방지)
    categorical_cols = ['age', 'gender', 'education', 'job_category', 'career_stage']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    return df


def prepare_satisfaction_model_features(user_input, ml_predictor):
    """
    만족도 모델을 위한 정확한 피처 생성
    """
    from flask import current_app
    
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
            "job_category": int(job_cat_code)
        }
        scenarios.append(scenario)

    df = pd.DataFrame(scenarios)

    # 9개 만족도 요인 추가 (만족도 모델은 모든 요인 필요)
    satis_cols = [f'satis_{cat}' for cat in ['wage', 'stability', 'growth', 'task_content', 'work_env', 'work_time', 'communication', 'fair_eval', 'welfare']]
    for factor in satis_cols:
        df[factor] = int(user_input.get(factor, 3))
    
    # 이직 시나리오에서는 새 직업군의 특성을 반영해서 만족도 요인 대폭 조정
    current_job_cat = int(user_input["current_job_category"])
    for idx, row_job_cat in enumerate(df['job_category']):
        if row_job_cat != current_job_cat:  # 이직 시나리오인 경우
            # 직업별로 뚜렷하게 다른 만족도 요인 변화 반영
            if row_job_cat == 1:  # 관리자 - 발전가능성과 임금만족도 대폭 상승
                df.loc[idx, 'satis_growth'] = min(5, df.loc[idx, 'satis_growth'] + 1.5)
                df.loc[idx, 'satis_wage'] = min(5, df.loc[idx, 'satis_wage'] + 1.2)
                df.loc[idx, 'satis_fair_eval'] = min(5, df.loc[idx, 'satis_fair_eval'] + 1.0)
                df.loc[idx, 'satis_welfare'] = min(5, df.loc[idx, 'satis_welfare'] + 0.8)
            elif row_job_cat == 2:  # 전문가 - 최고의 발전가능성과 업무만족도
                df.loc[idx, 'satis_growth'] = min(5, df.loc[idx, 'satis_growth'] + 2.0)
                df.loc[idx, 'satis_task_content'] = min(5, df.loc[idx, 'satis_task_content'] + 1.5)
                df.loc[idx, 'satis_wage'] = min(5, df.loc[idx, 'satis_wage'] + 1.0)
                df.loc[idx, 'satis_fair_eval'] = min(5, df.loc[idx, 'satis_fair_eval'] + 1.2)
            elif row_job_cat == 3:  # 사무직 - 안정성과 근무환경 상승
                df.loc[idx, 'satis_stability'] = min(5, df.loc[idx, 'satis_stability'] + 1.5)
                df.loc[idx, 'satis_work_env'] = min(5, df.loc[idx, 'satis_work_env'] + 1.0)
                df.loc[idx, 'satis_work_time'] = min(5, df.loc[idx, 'satis_work_time'] + 1.2)
                df.loc[idx, 'satis_welfare'] = min(5, df.loc[idx, 'satis_welfare'] + 0.8)
            elif row_job_cat == 4:  # 서비스직 - 임금 감소하지만 인간관계와 업무내용 개선
                df.loc[idx, 'satis_wage'] = max(1, df.loc[idx, 'satis_wage'] - 1.0)
                df.loc[idx, 'satis_communication'] = min(5, df.loc[idx, 'satis_communication'] + 1.5)
                df.loc[idx, 'satis_task_content'] = min(5, df.loc[idx, 'satis_task_content'] + 1.0)
                df.loc[idx, 'satis_work_time'] = min(5, df.loc[idx, 'satis_work_time'] + 0.8)
            elif row_job_cat == 5:  # 판매직 - 임금 변동성, 성장 가능성은 낮음
                df.loc[idx, 'satis_wage'] = max(1, df.loc[idx, 'satis_wage'] - 0.8)
                df.loc[idx, 'satis_growth'] = max(1, df.loc[idx, 'satis_growth'] - 1.0)
                df.loc[idx, 'satis_stability'] = max(1, df.loc[idx, 'satis_stability'] - 0.5)
                df.loc[idx, 'satis_communication'] = min(5, df.loc[idx, 'satis_communication'] + 1.2)
            else:  # 기타 직종 (기능직, 노무직 등) - 임금과 발전성 감소, 안정성은 업종에 따라
                df.loc[idx, 'satis_wage'] = max(1, df.loc[idx, 'satis_wage'] - 0.5)
                df.loc[idx, 'satis_growth'] = max(1, df.loc[idx, 'satis_growth'] - 0.8)
                if row_job_cat in [7, 8]:  # 기능직, 장치조작직 - 어느정도 안정성
                    df.loc[idx, 'satis_stability'] = min(5, df.loc[idx, 'satis_stability'] + 0.5)
                else:  # 단순노무직 등
                    df.loc[idx, 'satis_stability'] = max(1, df.loc[idx, 'satis_stability'] - 0.5)
    
    # prev_job_satisfaction 추가
    df['prev_job_satisfaction'] = int(user_input["job_satisfaction"])

    # 그룹 통계 피처 생성
    if ml_predictor.job_category_stats is not None:
        stats = ml_predictor.job_category_stats
        df = df.merge(stats, on='job_category', how='left')
        
        # 이직 시나리오의 소득을 점진적으로 조정 (급격한 변화 방지)
        current_income = int(user_input["monthly_income"])
        is_transfer = df['job_category'] != int(user_input["current_job_category"])
        
        for idx in df[is_transfer].index:
            target_income = df.loc[idx, 'job_category_income_avg']
            job_cat = df.loc[idx, 'job_category']
            
            # KLIPS 데이터 분포 기반 현실적 이직 소득 조정
            # 실제 평균 소득변화율: +12.69%, 5-95분위수: -40% ~ +80%
            if job_cat == 1:  # 관리자 - 매우 보수적 상승 (12% 목표, 88% 현재)
                adjusted_income = target_income * 0.12 + current_income * 0.88
            elif job_cat == 2:  # 전문가 - 적당한 상승 (18% 목표, 82% 현재)
                adjusted_income = target_income * 0.18 + current_income * 0.82
            elif job_cat == 3:  # 사무직 - 소폭 조정 (10% 목표, 90% 현재)
                adjusted_income = target_income * 0.10 + current_income * 0.90
            elif job_cat in [4, 5]:  # 서비스/판매직 - 현실적 하락 고려 (30% 목표, 70% 현재)
                adjusted_income = target_income * 0.30 + current_income * 0.70
            else:  # 기타 직업 - 일반적 조정 (25% 목표, 75% 현재)
                adjusted_income = target_income * 0.25 + current_income * 0.75
            
            df.loc[idx, 'monthly_income'] = int(adjusted_income)

    # === 만족도 모델 전용 피처 생성 ===
    df['satisfaction_lag_1'] = int(user_input["job_satisfaction"])
    df['income_lag_1'] = df['monthly_income']
    df['satisfaction_roll_mean_3'] = int(user_input["job_satisfaction"])
    df['satisfaction_roll_std_3'] = 0  # 단일 값이므로 변동성은 0
    df['income_roll_mean_3'] = df['monthly_income']
    df['career_length'] = (df['age'] - 18).clip(lower=1)  # 고졸 기준으로 변경
    df['career_stage'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], labels=[1, 2, 3, 4, 5]).astype(int)
    df['age_x_education'] = df['age'] * df['education']
    df['income_x_satisfaction_lag_1'] = df['income_lag_1'] * df['satisfaction_lag_1']
    
    # 직업별 평균 피처
    if ml_predictor.job_category_stats is not None:
        df['job_cat_income_avg'] = df['job_category_income_avg']
        df['job_cat_satis_avg'] = df['job_category_satisfaction_avg']
    else:
        df['job_cat_income_avg'] = df['monthly_income'] * 1.05
        df['job_cat_satis_avg'] = int(user_input["job_satisfaction"])

    # 공통 만족도 통계 피처
    satis_scores = df[satis_cols].fillna(3)
    df['satisfaction_mean'] = satis_scores.mean(axis=1)
    df['satisfaction_std'] = satis_scores.std(axis=1).fillna(0)
    df['satisfaction_range'] = satis_scores.max(axis=1) - satis_scores.min(axis=1)

    # 피처 값 범위 제한
    feature_limits = {
        'monthly_income': (10, 2000),
        'income_x_satisfaction_lag_1': (20, 10000),
        'age_x_education': (16, 400),
        'career_length': (0, 50),
        'income_roll_mean_3': (10, 2000),
        'job_cat_income_avg': (50, 1500),
        'career_stage': (1, 5),
        'satisfaction_mean': (1, 5),
        'satisfaction_std': (0, 3),
        'satisfaction_range': (0, 4),
    }
    
    for col, (min_val, max_val) in feature_limits.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=min_val, upper=max_val)
    
    # 최종 결측값 처리
    df = df.fillna(0)
    
    # categorical 피처들을 정수로 확실히 변환 (CatBoost 오류 방지)
    categorical_cols = ['age', 'gender', 'education', 'job_category', 'career_stage']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    return df


def prepare_prediction_features(user_input, ml_predictor):
    """
    호환성을 위한 래퍼 함수
    각 모델별로 올바른 피처를 생성하는 새로운 함수들을 호출
    """
    # 이 함수는 이제 사용되지 않으며, 새로운 모델별 함수를 사용해야 함
    raise NotImplementedError("Use prepare_income_model_features() or prepare_satisfaction_model_features() instead")