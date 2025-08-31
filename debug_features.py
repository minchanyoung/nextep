#!/usr/bin/env python
"""
Feature matching debugging script
"""
import sys
import os
import pandas as pd
import json
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_feature_definitions():
    """Load feature definitions from JSON files"""
    
    # Load income features
    with open('app/ml/saved_models/income_feature_names_correct.json', 'r') as f:
        income_features = json.load(f)
    
    # Load satisfaction features from ensemble config
    with open('app/ml/saved_models/final_ensemble_satis_config.json', 'r') as f:
        satis_config = json.load(f)
        satis_features = satis_config['features']
    
    return income_features, satis_features

def simulate_feature_generation():
    """Simulate the feature generation process"""
    
    # Mock user input
    user_input = {
        'age': 30,
        'gender': 1,
        'education': 3,
        'monthly_income': 300,
        'job_satisfaction': 4,
        'current_job_category': '2',
        'job_A_category': '1', 
        'job_B_category': '3',
        'satis_wage': 4,
        'satis_stability': 3,
        'satis_growth': 4,
        'satis_task_content': 4,
        'satis_work_env': 3,
        'satis_work_time': 4,
        'satis_communication': 3,
        'satis_fair_eval': 4,
        'satis_welfare': 3
    }
    
    # Create basic scenario dataframe (like in preprocessing.py)
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

    # Add 9 satisfaction factors
    satis_cols = [f'satis_{cat}' for cat in ['wage', 'stability', 'growth', 'task_content', 'work_env', 'work_time', 'communication', 'fair_eval', 'welfare']]
    for factor in satis_cols:
        df[factor] = int(user_input.get(factor, 3))
    
    # Add prev_job_satisfaction
    df['prev_job_satisfaction'] = df['job_satisfaction']

    # Add all the feature engineering from preprocessing.py
    # Income model features
    df['satisfaction_change_score'] = 0.0
    df['income_lag1'] = df['monthly_income']
    df['income_lag2'] = df['monthly_income']
    df['income_trend'] = 0.0
    df['prev_income_change'] = 0.0
    df['income_volatility'] = 0.0
    df['satisfaction_trend'] = 0.0
    df['satisfaction_volatility'] = 0.3
    df['career_length'] = (df['age'] - 22).clip(lower=1)
    df['job_stability'] = 1
    df['economic_cycle'] = 0.5
    df['income_age_ratio'] = df['monthly_income'] / df['age']
    df['peak_earning_years'] = ((df['age'] >= 40) & (df['age'] <= 55)).astype(int)
    df['education_roi'] = df['monthly_income'] / (df['education'] + 1)
    df['satisfaction_income_gap'] = 0
    df['job_category_change'] = 0
    df['potential_promotion'] = ((df['job_satisfaction'] > 3) & (df['satis_growth'] >= 4)).astype(int)
    df['career_stage'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], labels=[1, 2, 3, 4, 5]).astype(int)
    df['income_vs_peers'] = 0

    # Satisfaction model features
    df['satisfaction_lag_1'] = df['job_satisfaction']
    df['income_lag_1'] = df['monthly_income']
    df['satisfaction_roll_mean_3'] = df['job_satisfaction']
    df['satisfaction_roll_std_3'] = 0
    df['income_roll_mean_3'] = df['monthly_income']
    df['age_x_education'] = df['age'] * df['education']
    df['income_x_satisfaction_lag_1'] = df['income_lag_1'] * df['satisfaction_lag_1']
    df['job_cat_income_avg'] = df['monthly_income'] * 1.05
    df['job_cat_satis_avg'] = df['job_satisfaction']

    # Common satisfaction statistics
    satis_scores = df[satis_cols].fillna(3)
    df['satisfaction_mean'] = satis_scores.mean(axis=1)
    df['satisfaction_std'] = satis_scores.std(axis=1).fillna(0)
    df['satisfaction_range'] = satis_scores.max(axis=1) - satis_scores.min(axis=1)

    return list(df.columns)

def analyze_features():
    """Analyze feature mismatches"""
    
    # Load expected features
    income_expected, satis_expected = load_feature_definitions()
    
    # Generate actual features
    generated_features = simulate_feature_generation()
    
    print("=== FEATURE ANALYSIS ===")
    print(f"Generated features: {len(generated_features)}")
    print(f"Income expected: {len(income_expected)}")  
    print(f"Satisfaction expected: {len(satis_expected)}")
    
    print("\n=== GENERATED FEATURES ===")
    for i, feat in enumerate(generated_features, 1):
        print(f"{i:2d}. {feat}")
    
    print("\n=== INCOME MODEL EXPECTED ===")
    for i, feat in enumerate(income_expected, 1):
        print(f"{i:2d}. {feat}")
    
    print("\n=== SATISFACTION MODEL EXPECTED ===")
    for i, feat in enumerate(satis_expected, 1):
        print(f"{i:2d}. {feat}")
    
    # Analyze mismatches
    print("\n=== INCOME MODEL MISMATCH ANALYSIS ===")
    income_missing = [f for f in income_expected if f not in generated_features]
    income_extra = [f for f in generated_features if f not in income_expected]
    
    print(f"Missing from generated: {len(income_missing)}")
    for f in income_missing:
        print(f"  - {f}")
        
    print(f"Extra in generated: {len(income_extra)}")
    for f in income_extra[:10]:
        print(f"  + {f}")
    if len(income_extra) > 10:
        print(f"  ... and {len(income_extra)-10} more")
    
    print("\n=== SATISFACTION MODEL MISMATCH ANALYSIS ===")
    satis_missing = [f for f in satis_expected if f not in generated_features]
    satis_extra = [f for f in generated_features if f not in satis_expected]
    
    print(f"Missing from generated: {len(satis_missing)}")
    for f in satis_missing:
        print(f"  - {f}")
        
    print(f"Extra in generated: {len(satis_extra)}")
    for f in satis_extra[:10]:
        print(f"  + {f}")
    if len(satis_extra) > 10:
        print(f"  ... and {len(satis_extra)-10} more")

if __name__ == "__main__":
    analyze_features()