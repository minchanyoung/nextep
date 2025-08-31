import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedIncomePredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        
    def prepare_lstm_data(self, df, sequence_length=5):
        """LSTM용 시퀀스 데이터 준비"""
        # 개인별로 시계열 시퀀스 생성
        sequences = []
        targets = []
        
        for pid in df['pid'].unique():
            person_data = df[df['pid'] == pid].sort_values('year')
            if len(person_data) < sequence_length + 1:
                continue
                
            # 수치형 피처만 선택 (LSTM 입력용)
            numeric_features = [
                'age', 'education', 'monthly_income', 'job_satisfaction',
                'satis_wage', 'satis_stability', 'satis_growth',
                'income_lag1', 'income_trend', 'career_length',
                'income_age_ratio', 'education_roi'
            ]
            
            available_features = [f for f in numeric_features if f in person_data.columns]
            feature_data = person_data[available_features].values
            target_data = person_data['income_change_rate'].values
            
            for i in range(len(feature_data) - sequence_length):
                sequences.append(feature_data[i:i+sequence_length])
                targets.append(target_data[i+sequence_length])
                
        return np.array(sequences), np.array(targets)
    
    def build_lstm_model(self, input_shape):
        """개선된 LSTM 모델"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            BatchNormalization(),
            
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            BatchNormalization(),
            
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def build_gru_model(self, input_shape):
        """GRU 모델 (LSTM보다 빠름)"""
        model = Sequential([
            GRU(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            BatchNormalization(),
            
            GRU(64, return_sequences=False),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def train_lightgbm(self, X_train, y_train, X_test, y_test):
        """LightGBM 모델 (XGBoost보다 빠르고 종종 더 정확)"""
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'random_state': 42
        }
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(0)
            ]
        )
        return model
    
    def train_ridge_with_interactions(self, X_train, y_train):
        """상호작용 피처를 포함한 Ridge 회귀"""
        # 중요한 피처 간 상호작용 생성
        interaction_features = []
        
        # 기존 피처
        base_features = X_train.copy()
        
        # 핵심 상호작용 피처 추가
        if 'age' in X_train.columns and 'monthly_income' in X_train.columns:
            base_features['age_income_interaction'] = X_train['age'] * X_train['monthly_income']
            
        if 'education' in X_train.columns and 'job_satisfaction' in X_train.columns:
            base_features['edu_satisfaction_interaction'] = X_train['education'] * X_train['job_satisfaction']
            
        # Ridge 모델
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(base_features, y_train)
        
        return ridge, base_features.columns
    
    def create_ensemble_features(self, X):
        """앙상블용 메타 피처 생성"""
        meta_features = X.copy()
        
        # 통계적 변환
        if 'monthly_income' in X.columns:
            meta_features['income_log'] = np.log1p(X['monthly_income'])
            meta_features['income_sqrt'] = np.sqrt(X['monthly_income'])
            
        if 'age' in X.columns:
            meta_features['age_squared'] = X['age'] ** 2
            
        return meta_features
    
    def evaluate_model(self, name, y_true, y_pred):
        """모델 평가"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"\n-------- {name} --------")
        print(f" RMSE: {rmse:.4f}")
        print(f" MAE : {mae:.4f}")
        print(f" R²  : {r2:.4f}")
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2}

def main():
    """고급 모델들을 비교 테스트"""
    
    # 기존 데이터 로드 및 전처리 (기존 함수 재사용)
    from income_model_trainer import load_data, prepare_features_and_target, split_data_by_year
    
    print("=== 고급 모델 성능 비교 ===")
    
    # 데이터 준비
    df = load_data()
    X, y = prepare_features_and_target(df)
    X_train, y_train, X_test, y_test = split_data_by_year(df, X, y)
    
    predictor = AdvancedIncomePredictor()
    results = {}
    
    # 1. LightGBM 모델
    print("\n1. LightGBM 훈련 중...")
    lgb_model = predictor.train_lightgbm(X_train, y_train, X_test, y_test)
    lgb_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    results['LightGBM'] = predictor.evaluate_model("LightGBM", y_test, lgb_pred)
    
    # 2. Ridge with Interactions
    print("\n2. Ridge (상호작용 포함) 훈련 중...")
    ridge_model, ridge_features = predictor.train_ridge_with_interactions(X_train, y_train)
    
    # 테스트 데이터에도 같은 상호작용 피처 추가
    X_test_ridge = X_test.copy()
    if 'age' in X_test.columns and 'monthly_income' in X_test.columns:
        X_test_ridge['age_income_interaction'] = X_test['age'] * X_test['monthly_income']
    if 'education' in X_test.columns and 'job_satisfaction' in X_test.columns:
        X_test_ridge['edu_satisfaction_interaction'] = X_test['education'] * X_test['job_satisfaction']
    
    ridge_pred = ridge_model.predict(X_test_ridge)
    results['Ridge_Interactive'] = predictor.evaluate_model("Ridge (Interactive)", y_test, ridge_pred)
    
    # 3. LSTM 모델 (시퀀스 길이가 충분한 경우)
    print("\n3. LSTM 모델 준비 중...")
    try:
        X_seq, y_seq = predictor.prepare_lstm_data(df, sequence_length=3)
        
        if len(X_seq) > 1000:  # 충분한 시퀀스 데이터가 있는 경우
            # 훈련/테스트 분할 (시계열 순서 유지)
            train_size = int(0.8 * len(X_seq))
            X_seq_train, X_seq_test = X_seq[:train_size], X_seq[train_size:]
            y_seq_train, y_seq_test = y_seq[:train_size], y_seq[train_size:]
            
            # 데이터 정규화
            scaler = StandardScaler()
            X_seq_train_scaled = scaler.fit_transform(X_seq_train.reshape(-1, X_seq_train.shape[-1])).reshape(X_seq_train.shape)
            X_seq_test_scaled = scaler.transform(X_seq_test.reshape(-1, X_seq_test.shape[-1])).reshape(X_seq_test.shape)
            
            # LSTM 모델 훈련
            lstm_model = predictor.build_lstm_model(X_seq_train_scaled.shape[1:])
            
            callbacks = [
                EarlyStopping(patience=20, restore_best_weights=True),
                ReduceLROnPlateau(patience=10, factor=0.5)
            ]
            
            lstm_model.fit(
                X_seq_train_scaled, y_seq_train,
                validation_data=(X_seq_test_scaled, y_seq_test),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            lstm_pred = lstm_model.predict(X_seq_test_scaled, verbose=0).flatten()
            results['LSTM'] = predictor.evaluate_model("LSTM", y_seq_test, lstm_pred)
        else:
            print("시퀀스 데이터 부족으로 LSTM 건너뜀")
            
    except Exception as e:
        print(f"LSTM 모델 오류: {e}")
    
    # 4. 앙상블 모델 (기존 XGBoost + 새 모델들)
    print("\n4. 앙상블 모델...")
    ensemble_pred = (lgb_pred + ridge_pred) / 2
    results['Ensemble'] = predictor.evaluate_model("Ensemble (LGB+Ridge)", y_test, ensemble_pred)
    
    # 결과 요약
    print("\n=== 최종 성능 비교 ===")
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('r2', ascending=False)
    print(results_df.round(4))
    
    # 최고 성능 모델 저장
    best_model_name = results_df.index[0]
    print(f"\n최고 성능 모델: {best_model_name} (R² = {results_df.loc[best_model_name, 'r2']:.4f})")
    
    # LightGBM이 최고 성능인 경우 저장
    if best_model_name == 'LightGBM':
        lgb_model.save_model('app/ml/saved_models/lightgbm_income_model.txt')
        print("LightGBM 모델 저장 완료")
    
if __name__ == "__main__":
    main()