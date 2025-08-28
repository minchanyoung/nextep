
import json
import os
from catboost import CatBoostRegressor

# Define absolute paths
base_path = "C:\\Users\\User\\Desktop\\nextep"
model_dir = os.path.join(base_path, 'app', 'ml', 'saved_models')

satis_model_path = os.path.join(model_dir, 'cat_satisfaction_change_model.cbm')
satis_json_path = os.path.join(model_dir, 'satis_feature_names.json')

income_model_path = os.path.join(model_dir, 'cat_income_change_model.cbm')
income_json_path = os.path.join(model_dir, 'income_feature_names.json')

try:
    # Process Satisfaction Model
    print(f"Loading satisfaction model from {satis_model_path}")
    satis_model = CatBoostRegressor()
    satis_model.load_model(satis_model_path)
    satis_feature_names = satis_model.feature_names_
    with open(satis_json_path, 'w', encoding='utf-8') as f:
        json.dump(satis_feature_names, f, ensure_ascii=False, indent=4)
    print(f"Successfully saved satisfaction feature names to {satis_json_path}")
    # print(satis_feature_names)

    # Process Income Model
    print(f"Loading income model from {income_model_path}")
    income_model = CatBoostRegressor()
    income_model.load_model(income_model_path)
    income_feature_names = income_model.feature_names_
    with open(income_json_path, 'w', encoding='utf-8') as f:
        json.dump(income_feature_names, f, ensure_ascii=False, indent=4)
    print(f"Successfully saved income feature names to {income_json_path}")
    # print(income_feature_names)

except FileNotFoundError as e:
    print(f"Error: Model file not found. {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
