import pandas as pd
import json
import random
import numpy as np

# 직업별 이직 성공률 (실제 경험 데이터 기반)
JOB_TRANSFER_SUCCESS_RATES = {
    1: {'income_success': 0.65, 'satisfaction_success': 0.70},  # 관리자
    2: {'income_success': 0.75, 'satisfaction_success': 0.80},  # 전문가
    3: {'income_success': 0.55, 'satisfaction_success': 0.60},  # 사무직
    4: {'income_success': 0.45, 'satisfaction_success': 0.55},  # 서비스직
    5: {'income_success': 0.40, 'satisfaction_success': 0.50},  # 판매직
    6: {'income_success': 0.35, 'satisfaction_success': 0.45},  # 농림어업
    7: {'income_success': 0.50, 'satisfaction_success': 0.55},  # 기능원
    8: {'income_success': 0.45, 'satisfaction_success': 0.50},  # 장치조작
    9: {'income_success': 0.30, 'satisfaction_success': 0.40},  # 단순노무
}

def prepare_income_model_features(user_input, ml_predictor):
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
            
            # KLIPS 데이터 분포 기반 현실적 이직 소득 조정 (더 강한 차별화)
            # 실제 평균 소득변화율: +12.69%, 5-95분위수: -40% ~ +80%
            if job_cat == 1:  # 관리자 - 고소득 직종 (25% 목표, 75% 현재)
                adjusted_income = target_income * 0.25 + current_income * 0.75
            elif job_cat == 2:  # 전문가 - 최고 소득 직종 (35% 목표, 65% 현재)  
                adjusted_income = target_income * 0.35 + current_income * 0.65
            elif job_cat == 3:  # 사무직 - 안정 직종 (15% 목표, 85% 현재)
                adjusted_income = target_income * 0.15 + current_income * 0.85
            elif job_cat == 4:  # 서비스직 - 변동성 큰 직종 (40% 목표, 60% 현재)
                adjusted_income = target_income * 0.40 + current_income * 0.60
            elif job_cat == 5:  # 판매직 - 성과 기반 직종 (45% 목표, 55% 현재)
                adjusted_income = target_income * 0.45 + current_income * 0.55
            else:  # 기타 직업 - 일반적 조정 (20% 목표, 80% 현재)
                adjusted_income = target_income * 0.20 + current_income * 0.80
            
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
    
    # 동료 대비 소득 (직업별 가중 차이)
    if ml_predictor.job_category_stats is not None:
        df['income_vs_peers'] = df['monthly_income'] - df['job_category_income_avg']
        
        # 직업별 소득 차이 가중치 적용 (차별화 강화)
        for idx in range(len(df)):
            job_cat = df.loc[idx, 'job_category']
            if job_cat == 1:  # 관리자 - 고소득 프리미엄 강조
                df.loc[idx, 'income_vs_peers'] = df.loc[idx, 'income_vs_peers'] * 1.3
            elif job_cat == 2:  # 전문가 - 최대 프리미엄 강조  
                df.loc[idx, 'income_vs_peers'] = df.loc[idx, 'income_vs_peers'] * 1.5
            elif job_cat == 3:  # 사무직 - 기본
                df.loc[idx, 'income_vs_peers'] = df.loc[idx, 'income_vs_peers'] * 1.0
            elif job_cat in [4, 5]:  # 서비스/판매직 - 변동성 반영
                df.loc[idx, 'income_vs_peers'] = df.loc[idx, 'income_vs_peers'] * 0.8
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
            
            # 직업별로 다른 소득 변화 시그널 생성 (더 강력한 차별화)
            if row_job_cat == 1:  # 관리자 - 고소득 잠재력
                df.loc[idx, 'potential_promotion'] = 1
                df.loc[idx, 'career_stage'] = min(5, df.loc[idx, 'career_stage'] + 1)
                df.loc[idx, 'education_roi'] = df.loc[idx, 'education_roi'] * 1.25  # 관리자 프리미엄
                df.loc[idx, 'income_age_ratio'] = df.loc[idx, 'income_age_ratio'] * 1.15
                df.loc[idx, 'peak_earning_years'] = 1  # 관리자는 고소득 시기
                
            elif row_job_cat == 2:  # 전문가 - 전문성 프리미엄
                df.loc[idx, 'potential_promotion'] = 1
                df.loc[idx, 'education_roi'] = df.loc[idx, 'education_roi'] * 1.35  # 전문가 최고 프리미엄
                df.loc[idx, 'income_age_ratio'] = df.loc[idx, 'income_age_ratio'] * 1.20
                if df.loc[idx, 'education'] >= 4:  # 대졸 이상 전문가
                    df.loc[idx, 'satisfaction_income_gap'] = df.loc[idx, 'satisfaction_income_gap'] + 0.5
                    
            elif row_job_cat == 3:  # 사무직 - 안정성 중심
                df.loc[idx, 'job_stability'] = 2  # 높은 안정성
                df.loc[idx, 'income_age_ratio'] = df.loc[idx, 'income_age_ratio'] * 0.95  # 소득 증가 보수적
                df.loc[idx, 'economic_cycle'] = 0.7  # 경기 영향 적음
                
            elif row_job_cat == 4:  # 서비스직 - 경험 중심
                df.loc[idx, 'career_length'] = df.loc[idx, 'career_length'] + 2  # 경력 가산
                df.loc[idx, 'income_age_ratio'] = df.loc[idx, 'income_age_ratio'] * 0.85
                df.loc[idx, 'job_stability'] = 0  # 낮은 안정성
                
            elif row_job_cat == 5:  # 판매직 - 성과 변동성
                df.loc[idx, 'income_volatility'] = 0.8  # 높은 변동성
                df.loc[idx, 'economic_cycle'] = 1.2  # 경기 민감
                df.loc[idx, 'income_age_ratio'] = df.loc[idx, 'income_age_ratio'] * 0.90
                
            else:  # 기타 직종 - 기본 조정
                df.loc[idx, 'income_age_ratio'] = df.loc[idx, 'income_age_ratio'] * 0.92

    # 최종 결측값 처리
    df = df.fillna(0)
    
    # categorical 피처들을 정수로 확실히 변환 (CatBoost 오류 방지)
    categorical_cols = ['age', 'gender', 'education', 'job_category', 'career_stage']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
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
    
    # [리팩토링] 이직 시나리오에서 만족도 요인을 확률 기반으로 현실적으로 조정
    current_job_cat = int(user_input["current_job_category"])
    for idx, row in df.iterrows():
        target_job_cat = int(row['job_category'])
        if target_job_cat != current_job_cat:  # 이직 시나리오인 경우
            # 직업별 이직 성공 확률 가져오기
            success_prob = JOB_TRANSFER_SUCCESS_RATES.get(target_job_cat, {'satisfaction_success': 0.5})['satisfaction_success']
            
            # 각 만족도 요인에 대해 확률적 변화 적용
            for factor in satis_cols:
                if factor in df.columns:
                    current_value = df.loc[idx, factor]
                    
                    # 성공 확률에 따라 개선 또는 악화 결정
                    if random.random() < success_prob:
                        # 성공: 0.2 ~ 1.0점 사이에서 랜덤하게 개선
                        change = np.random.uniform(0.2, 1.0)
                        df.loc[idx, factor] = min(5, current_value + change)
                    else:
                        # 실패: 0.2 ~ 1.0점 사이에서 랜덤하게 악화
                        change = np.random.uniform(0.2, 1.0)
                        df.loc[idx, factor] = max(1, current_value - change)

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
            if job_cat == 1:  # 관리자
                adjusted_income = target_income * 0.12 + current_income * 0.88
            elif job_cat == 2:  # 전문가
                adjusted_income = target_income * 0.18 + current_income * 0.82
            elif job_cat == 3:  # 사무직
                adjusted_income = target_income * 0.10 + current_income * 0.90
            elif job_cat in [4, 5]:  # 서비스/판매직
                adjusted_income = target_income * 0.30 + current_income * 0.70
            else:  # 기타
                adjusted_income = target_income * 0.25 + current_income * 0.75
            
            df.loc[idx, 'monthly_income'] = int(adjusted_income)

    # === 만족도 모델 전용 피처 생성 ===
    df['satisfaction_lag_1'] = int(user_input["job_satisfaction"])
    df['income_lag_1'] = df['monthly_income']
    df['satisfaction_roll_mean_3'] = int(user_input["job_satisfaction"])
    df['satisfaction_roll_std_3'] = 0
    df['income_roll_mean_3'] = df['monthly_income']
    df['career_length'] = (df['age'] - 18).clip(lower=1)
    df['career_stage'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], labels=[1, 2, 3, 4, 5]).astype(int)
    df['age_x_education'] = df['age'] * df['education']
    df['income_x_satisfaction_lag_1'] = df['income_lag_1'] * df['satisfaction_lag_1']
    
    if ml_predictor.job_category_stats is not None:
        df['job_cat_income_avg'] = df['job_category_income_avg']
        df['job_cat_satis_avg'] = df['job_category_satisfaction_avg']
    else:
        df['job_cat_income_avg'] = df['monthly_income'] * 1.05
        df['job_cat_satis_avg'] = int(user_input["job_satisfaction"])

    satis_scores = df[satis_cols].fillna(3)
    df['satisfaction_mean'] = satis_scores.mean(axis=1)
    df['satisfaction_std'] = satis_scores.std(axis=1).fillna(0)
    df['satisfaction_range'] = satis_scores.max(axis=1) - satis_scores.min(axis=1)

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
    
    df = df.fillna(0)
    
    categorical_cols = ['age', 'gender', 'education', 'job_category', 'career_stage']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    return df


