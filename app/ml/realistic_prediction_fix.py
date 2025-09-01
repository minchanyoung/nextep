"""
현실적 예측을 위한 개선 모듈
- 확률 기반 이직 결과 생성
- 음성 시나리오 포함
- KLIPS 데이터 기반 현실적 분포 반영
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import random

# KLIPS 실제 분포 기반 상수
REAL_INCOME_DISTRIBUTION = {
    'mean': 0.126943,
    'std': 1.060963,
    'negative_ratio': 0.2795,  # 27.95% 음수
    'percentiles': {
        5: -0.4,    # 5분위수: -40%
        25: -0.048,  # 25분위수: -4.8%
        75: 0.2,    # 75분위수: +20%
        95: 0.8     # 95분위수: +80%
    }
}

REAL_SATISFACTION_DISTRIBUTION = {
    'mean': -0.008254,
    'std': 0.629151, 
    'negative_ratio': 0.1735,  # 17.35% 음수
    'zero_ratio': 0.6605,      # 66.05% 제로
    'positive_ratio': 0.1660   # 16.60% 양수
}

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

# 연령별 리스크 팩터
AGE_RISK_FACTORS = {
    (18, 25): {'income_volatility': 1.3, 'success_bonus': 0.1},  # 젊은층 높은 변동성, 약간의 보너스
    (26, 35): {'income_volatility': 1.1, 'success_bonus': 0.2},  # 중간층 안정적, 높은 성공률
    (36, 45): {'income_volatility': 0.9, 'success_bonus': 0.0},  # 중년층 안정적, 보너스 없음
    (46, 55): {'income_volatility': 0.7, 'success_bonus': -0.1}, # 중장년층 낮은 변동성, 페널티
    (56, 100): {'income_volatility': 0.5, 'success_bonus': -0.2} # 고령층 매우 안정적, 높은 페널티
}

def get_age_risk_factor(age: int) -> Dict:
    """연령별 리스크 팩터 반환"""
    for (min_age, max_age), factors in AGE_RISK_FACTORS.items():
        if min_age <= age <= max_age:
            return factors
    return AGE_RISK_FACTORS[(26, 35)]  # 기본값

def apply_realistic_income_change(
    current_job: int, 
    target_job: int, 
    current_income: int, 
    age: int,
    user_profile: Dict
) -> float:
    """
    현실적 소득 변화 계산
    - 성공/실패 확률 고려
    - KLIPS 실제 분포 반영
    - 개인 프로필 고려
    """
    
    # 이직 성공률 가져오기
    success_rates = JOB_TRANSFER_SUCCESS_RATES.get(target_job, {'income_success': 0.5})
    income_success_prob = success_rates['income_success']
    
    # 연령별 보정
    age_factor = get_age_risk_factor(age)
    adjusted_success_prob = max(0.1, min(0.9, income_success_prob + age_factor['success_bonus']))
    
    # 개인 프로필 보정 (교육수준, 현재 만족도 등)
    education = user_profile.get('education', 3)
    current_satisfaction = user_profile.get('job_satisfaction', 3)
    
    # 교육수준이 높을수록, 현재 만족도가 낮을수록 성공 확률 증가
    education_bonus = (education - 3) * 0.05  # 교육수준별 ±10% 보정
    dissatisfaction_bonus = max(0, (3 - current_satisfaction)) * 0.03  # 불만족시 동기 증가
    
    final_success_prob = max(0.05, min(0.95, adjusted_success_prob + education_bonus + dissatisfaction_bonus))
    
    # 확률적 결과 결정
    is_success = random.random() < final_success_prob
    
    if is_success:
        # 성공 케이스: KLIPS 긍정적 분포에서 샘플링
        # 25-95 분위수 범위에서 샘플링 (상위 성공 사례)
        change_rate = np.random.uniform(
            REAL_INCOME_DISTRIBUTION['percentiles'][25], 
            REAL_INCOME_DISTRIBUTION['percentiles'][95]
        )
    else:
        # 실패 케이스: 음수 또는 낮은 양수
        # 5-25 분위수 범위에서 샘플링 (하위 실패 사례)
        change_rate = np.random.uniform(
            REAL_INCOME_DISTRIBUTION['percentiles'][5],
            REAL_INCOME_DISTRIBUTION['percentiles'][25]
        )
    
    # 변동성 조정
    volatility = age_factor['income_volatility']
    change_rate *= volatility
    
    # 극단값 제한 (-50% ~ +100%)
    return max(-0.5, min(1.0, change_rate))

def apply_realistic_satisfaction_change(
    current_job: int,
    target_job: int, 
    age: int,
    user_profile: Dict
) -> float:
    """
    현실적 만족도 변화 계산
    - 66% 제로 변화 (실제 데이터 반영)
    - 17% 음수, 16% 양수
    """
    
    # 실제 분포 기반 확률적 결정
    rand = random.random()
    
    if rand < REAL_SATISFACTION_DISTRIBUTION['zero_ratio']:
        # 66% 확률로 변화 없음
        return 0.0
    elif rand < REAL_SATISFACTION_DISTRIBUTION['zero_ratio'] + REAL_SATISFACTION_DISTRIBUTION['negative_ratio']:
        # 17% 확률로 음수 변화
        return np.random.uniform(-2.0, -0.1)
    else:
        # 16% 확률로 양수 변화  
        return np.random.uniform(0.1, 2.0)

def get_realistic_fallback_prediction(
    user_profile: Dict,
    scenario_type: str  # 'current', 'job_A', 'job_B'
) -> Tuple[float, float]:
    """
    현실적 Fallback 예측 (모델 실패시)
    - KLIPS 실제 분포 기반
    - 음수 결과 포함
    """
    
    age = user_profile['age']
    current_job = user_profile['current_job_category']
    
    if scenario_type == 'current':
        # 현직 유지: 보수적 예측
        income_change = np.random.normal(0.05, 0.15)  # 평균 5%, 표준편차 15%
        satisfaction_change = np.random.choice(
            [0.0, -0.1, 0.1], 
            p=[0.8, 0.1, 0.1]  # 80% 변화없음, 10% 하락, 10% 상승
        )
    else:
        # 이직 시나리오: 실제 KLIPS 분포 기반
        target_job = user_profile['job_A_category'] if scenario_type == 'job_A' else user_profile['job_B_category']
        
        income_change = apply_realistic_income_change(
            current_job, target_job, user_profile['monthly_income'], age, user_profile
        )
        satisfaction_change = apply_realistic_satisfaction_change(
            current_job, target_job, age, user_profile
        )
    
    return round(income_change, 4), round(satisfaction_change, 4)

def add_realistic_noise_to_features(df: pd.DataFrame, user_input: Dict) -> pd.DataFrame:
    """
    피처에 현실적 노이즈 추가
    - 결정론적 개선 제거
    - 확률 기반 변화 적용
    """
    
    current_job = int(user_input["current_job_category"])
    
    for idx, row in df.iterrows():
        target_job = int(row['job_category'])
        
        if target_job != current_job:  # 이직 시나리오
            # 만족도 요인에 확률적 변화 적용 (기존의 무조건 개선 제거)
            success_prob = JOB_TRANSFER_SUCCESS_RATES.get(target_job, {'satisfaction_success': 0.5})['satisfaction_success']
            
            satis_factors = ['satis_wage', 'satis_stability', 'satis_growth', 'satis_task_content', 
                           'satis_work_env', 'satis_work_time', 'satis_communication', 'satis_fair_eval', 'satis_welfare']
            
            for factor in satis_factors:
                if factor in df.columns:
                    current_value = df.loc[idx, factor]
                    
                    # 성공 확률 기반으로 개선/악화 결정
                    if random.random() < success_prob:
                        # 성공: 0.5-1.5점 개선
                        change = np.random.uniform(0.2, 1.0)
                        df.loc[idx, factor] = min(5, current_value + change)
                    else:
                        # 실패: 0.2-1.0점 악화
                        change = np.random.uniform(0.2, 1.0)  
                        df.loc[idx, factor] = max(1, current_value - change)
            
            # potential_promotion도 확률적으로 설정
            df.loc[idx, 'potential_promotion'] = 1 if random.random() < 0.3 else 0
    
    return df

# 사용 예시 함수
def apply_realistic_corrections_to_preprocessing():
    """
    preprocessing_fixed.py에 적용할 현실적 보정 가이드
    """
    corrections = {
        "만족도 요인 조정": "확정적 개선 → 확률 기반 변화",
        "소득 조정": "상향 편향 → 실제 KLIPS 분포 반영", 
        "시그널 생성": "긍정적 신호만 → 성공/실패 혼재",
        "Fallback 로직": "낙관적 → 현실적 (음수 포함)"
    }
    return corrections