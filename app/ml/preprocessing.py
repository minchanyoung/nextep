import pandas as pd
import numpy as np

JOB_TRANSFER_SUCCESS_RATES = {
    1: {'income_success': 0.65, 'satisfaction_success': 0.70},
    2: {'income_success': 0.75, 'satisfaction_success': 0.80},
    3: {'income_success': 0.55, 'satisfaction_success': 0.60},
    4: {'income_success': 0.45, 'satisfaction_success': 0.55},
    5: {'income_success': 0.40, 'satisfaction_success': 0.50},
    6: {'income_success': 0.35, 'satisfaction_success': 0.45},
    7: {'income_success': 0.50, 'satisfaction_success': 0.55},
    8: {'income_success': 0.45, 'satisfaction_success': 0.50},
    9: {'income_success': 0.30, 'satisfaction_success': 0.40},
}

def _ensure_feature_order(df: pd.DataFrame, feature_order: list) -> pd.DataFrame:
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0
    return df[feature_order]

def _cast_dtypes_for_models(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    for c in categorical_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype('int32')
    for c in df.columns:
        if c not in categorical_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype('float32')
    return df

def prepare_income_model_features(user_input, ml_predictor, scenario_codes=None):
    if scenario_codes is None:
        scenario_codes = ['current', 'jobA', 'jobB']
    scenarios = []
    for code in scenario_codes:
        job_cat_code = user_input["current_job_category"]
        if code == 'jobA':
            job_cat_code = user_input["job_A_category"]
        elif code == 'jobB':
            job_cat_code = user_input["job_B_category"]
        elif code != 'current':
            job_cat_code = code
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
    df['prev_job_satisfaction'] = df['job_satisfaction']
    df['satis_wage'] = int(user_input.get('satis_wage', 3))
    df['satis_stability'] = int(user_input.get('satis_stability', 3))
    df['satis_growth'] = int(user_input.get('satis_growth', 3))
    if ml_predictor.job_category_stats is not None:
        stats = ml_predictor.job_category_stats
        df = df.merge(stats, on='job_category', how='left')
        current_income = int(user_input["monthly_income"])
        is_transfer = df['job_category'] != int(user_input["current_job_category"])
        for idx in df[is_transfer].index:
            target_income = df.loc[idx, 'job_category_income_avg']
            job_cat = df.loc[idx, 'job_category']
            if job_cat == 1:
                adjusted_income = target_income * 0.25 + current_income * 0.75
            elif job_cat == 2:
                adjusted_income = target_income * 0.35 + current_income * 0.65
            elif job_cat == 3:
                adjusted_income = target_income * 0.15 + current_income * 0.85
            elif job_cat == 4:
                adjusted_income = target_income * 0.40 + current_income * 0.60
            elif job_cat == 5:
                adjusted_income = target_income * 0.45 + current_income * 0.55
            else:
                adjusted_income = target_income * 0.20 + current_income * 0.80
            df.loc[idx, 'monthly_income'] = int(adjusted_income)
    df['satisfaction_change_score'] = 0.0
    df['income_lag1'] = df['monthly_income']
    df['income_lag2'] = df['monthly_income']
    df['income_trend'] = 0.0
    df['prev_income_change'] = 0.0
    df['income_volatility'] = 0.0
    df['satisfaction_trend'] = 0.0
    df['satisfaction_volatility'] = 0.3
    df['career_length'] = (df['age'] - 18).clip(lower=1)
    df['job_stability'] = 1
    df['economic_cycle'] = 0.5
    age_adjusted = df['age'].clip(lower=25)
    df['income_age_ratio'] = df['monthly_income'] / age_adjusted
    df['peak_earning_years'] = ((df['age'] >= 40) & (df['age'] <= 55)).astype(int)
    df['education_roi'] = df['monthly_income'] / (df['education'] + 1)
    if ml_predictor.job_category_stats is not None:
        income_norm = df['monthly_income'] / df['job_category_income_avg'] * 3
        df['satisfaction_income_gap'] = df['satis_wage'] - income_norm
    else:
        df['satisfaction_income_gap'] = 0
    df['job_category_change'] = 0
    df['potential_promotion'] = ((df['job_satisfaction'] > 3) & (df['satis_growth'] >= 4)).astype(int)
    df['career_stage'] = pd.cut(df['age'], bins=[0, 23, 28, 35, 45, 55, 100], labels=[1, 2, 3, 4, 5, 6]).fillna(1).astype(int).clip(upper=5)
    if ml_predictor.job_category_stats is not None:
        df['income_vs_peers'] = df['monthly_income'] - df['job_category_income_avg']
        for idx in range(len(df)):
            job_cat = df.loc[idx, 'job_category']
            if job_cat == 1:
                df.loc[idx, 'income_vs_peers'] = df.loc[idx, 'income_vs_peers'] * 1.3
            elif job_cat == 2:
                df.loc[idx, 'income_vs_peers'] = df.loc[idx, 'income_vs_peers'] * 1.5
            elif job_cat == 3:
                df.loc[idx, 'income_vs_peers'] = df.loc[idx, 'income_vs_peers'] * 1.0
            elif job_cat in [4, 5]:
                df.loc[idx, 'income_vs_peers'] = df.loc[idx, 'income_vs_peers'] * 0.8
    else:
        df['income_vs_peers'] = 0
    feature_limits = {
        'income_age_ratio': (3, 15),
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
    current_job_cat = int(user_input["current_job_category"])
    for idx, row_job_cat in enumerate(df['job_category']):
        if row_job_cat != current_job_cat:
            df.loc[idx, 'job_category_change'] = 1
            if row_job_cat == 1:
                df.loc[idx, 'potential_promotion'] = 1
                df.loc[idx, 'career_stage'] = min(5, df.loc[idx, 'career_stage'] + 1)
                df.loc[idx, 'education_roi'] = df.loc[idx, 'education_roi'] * 1.25
                df.loc[idx, 'income_age_ratio'] = df.loc[idx, 'income_age_ratio'] * 1.15
                df.loc[idx, 'peak_earning_years'] = 1
            elif row_job_cat == 2:
                df.loc[idx, 'potential_promotion'] = 1
                df.loc[idx, 'education_roi'] = df.loc[idx, 'education_roi'] * 1.35
                df.loc[idx, 'income_age_ratio'] = df.loc[idx, 'income_age_ratio'] * 1.20
                if df.loc[idx, 'education'] >= 4:
                    df.loc[idx, 'satisfaction_income_gap'] = df.loc[idx, 'satisfaction_income_gap'] + 0.5
            elif row_job_cat == 3:
                df.loc[idx, 'job_stability'] = 2
                df.loc[idx, 'income_age_ratio'] = df.loc[idx, 'income_age_ratio'] * 0.95
                df.loc[idx, 'economic_cycle'] = 0.7
            elif row_job_cat == 4:
                df.loc[idx, 'career_length'] = df.loc[idx, 'career_length'] + 2
                df.loc[idx, 'income_age_ratio'] = df.loc[idx, 'income_age_ratio'] * 0.85
                df.loc[idx, 'job_stability'] = 0
            elif row_job_cat == 5:
                df.loc[idx, 'income_volatility'] = 0.8
                df.loc[idx, 'economic_cycle'] = 1.2
                df.loc[idx, 'income_age_ratio'] = df.loc[idx, 'income_age_ratio'] * 0.90
            else:
                df.loc[idx, 'income_age_ratio'] = df.loc[idx, 'income_age_ratio'] * 0.92
    df = df.fillna(0)
    categorical_cols = ['age', 'gender', 'education', 'job_category', 'career_stage']
    df = _cast_dtypes_for_models(df, categorical_cols)
    if hasattr(ml_predictor, "features") and "income" in ml_predictor.features:
        df = _ensure_feature_order(df, ml_predictor.features["income"])
    return df

def prepare_satisfaction_model_features(user_input, ml_predictor, scenario_codes=None):
    if scenario_codes is None:
        scenario_codes = ['current', 'jobA', 'jobB']
    scenarios = []
    for code in scenario_codes:
        job_cat_code = user_input["current_job_category"]
        if code == 'jobA':
            job_cat_code = user_input["job_A_category"]
        elif code == 'jobB':
            job_cat_code = user_input["job_B_category"]
        elif code != 'current':
            job_cat_code = code
        scenario = {
            "age": int(user_input["age"]),
            "gender": int(user_input["gender"]),
            "education": int(user_input["education"]),
            "monthly_income": int(user_input["monthly_income"]),
            "job_category": int(job_cat_code)
        }
        scenarios.append(scenario)
    df = pd.DataFrame(scenarios)
    satis_cols = [f'satis_{cat}' for cat in ['wage', 'stability', 'growth', 'task_content', 'work_env', 'work_time', 'communication', 'fair_eval', 'welfare']]
    for factor in satis_cols:
        df[factor] = int(user_input.get(factor, 3))
    current_job_cat = int(user_input["current_job_category"])
    for idx, row in df.iterrows():
        target_job_cat = int(row['job_category'])
        if target_job_cat != current_job_cat:
            success_prob = JOB_TRANSFER_SUCCESS_RATES.get(target_job_cat, {'satisfaction_success': 0.5})['satisfaction_success']
            delta = (2.0 * float(success_prob) - 1.0) * 0.6
            for factor in satis_cols:
                current_value = float(df.loc[idx, factor])
                new_value = current_value + delta
                df.loc[idx, factor] = min(5.0, max(1.0, float(new_value)))
    df['prev_job_satisfaction'] = int(user_input["job_satisfaction"])
    if ml_predictor.job_category_stats is not None:
        stats = ml_predictor.job_category_stats
        df = df.merge(stats, on='job_category', how='left')
        current_income = int(user_input["monthly_income"])
        is_transfer = df['job_category'] != int(user_input["current_job_category"])
        for idx in df[is_transfer].index:
            target_income = df.loc[idx, 'job_category_income_avg']
            job_cat = df.loc[idx, 'job_category']
            if job_cat == 1:
                adjusted_income = target_income * 0.12 + current_income * 0.88
            elif job_cat == 2:
                adjusted_income = target_income * 0.18 + current_income * 0.82
            elif job_cat == 3:
                adjusted_income = target_income * 0.10 + current_income * 0.90
            elif job_cat in [4, 5]:
                adjusted_income = target_income * 0.30 + current_income * 0.70
            else:
                adjusted_income = target_income * 0.25 + current_income * 0.75
            df.loc[idx, 'monthly_income'] = int(adjusted_income)
    df['satisfaction_lag_1'] = int(user_input["job_satisfaction"])
    df['income_lag_1'] = df['monthly_income']
    df['satisfaction_roll_mean_3'] = float(user_input["job_satisfaction"])
    df['satisfaction_roll_std_3'] = 0.0
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
    df = _cast_dtypes_for_models(df, categorical_cols)
    if hasattr(ml_predictor, "features") and "satis" in ml_predictor.features:
        df = _ensure_feature_order(df, ml_predictor.features["satis"])
    return df
