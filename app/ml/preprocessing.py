import pandas as pd

def prepare_prediction_features(user_input, ml_predictor):
    base_features = {
        "age": int(user_input["age"]), "gender": int(user_input["gender"]),
        "education": int(user_input["education"]), "monthly_income": int(user_input["monthly_income"]),
        "prev_monthly_income": int(user_input["monthly_income"]),
        "job_satisfaction": int(user_input["job_satisfaction"]),
        "prev_job_satisfaction": int(user_input["job_satisfaction"]),
    }
    satis_keys = [
        "satis_wage", "satis_stability", "satis_task_content", "satis_work_env", 
        "satis_work_time", "satis_growth", "satis_communication", "satis_fair_eval", "satis_welfare"
    ]
    for key in satis_keys:
        base_features[key] = 1 if key == user_input["satis_focus_key"] else 0

    scenarios = []
    job_categories = [
        user_input["current_job_category"], user_input["job_A_category"], user_input["job_B_category"]
    ]
    
    for i, job_cat_code in enumerate(job_categories):
        scenario = base_features.copy()
        scenario["job_category"] = int(job_cat_code)
        
        try:
            job_stats = ml_predictor.job_category_stats.loc[int(job_cat_code)]
            job_avg_income = job_stats['job_category_income_avg']
            scenario.update({
                "job_category_income_avg": job_avg_income,
                "job_category_education_avg": job_stats['job_category_education_avg'],
                "job_category_satisfaction_avg": job_stats['job_category_satisfaction_avg']
            })
        except KeyError:
            overall_stats = ml_predictor.klips_df
            job_avg_income = overall_stats['monthly_income'].mean()
            scenario.update({
                "job_category_income_avg": job_avg_income,
                "job_category_education_avg": overall_stats['education'].mean(),
                "job_category_satisfaction_avg": overall_stats['job_satisfaction'].mean()
            })

        if i > 0 and int(job_cat_code) != int(user_input["current_job_category"]):
            scenario["monthly_income"] = job_avg_income
        
        scenario["income_relative_to_job"] = scenario["monthly_income"] - scenario["job_category_income_avg"]
        scenario["education_relative_to_job"] = scenario["education"] - scenario["job_category_education_avg"]
        scenario["age_x_job_category"] = scenario["age"] * scenario["job_category"]
        scenario["monthly_income_x_job_category"] = scenario["monthly_income"] * scenario["job_category"]
        scenario["education_x_job_category"] = scenario["education"] * scenario["job_category"]
        scenario["income_relative_to_job_x_job_category"] = scenario["income_relative_to_job"] * scenario["job_category"]
        
        scenarios.append(scenario)

    return scenarios