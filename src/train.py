import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- MLOps CONFIG ---
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Calorie_Prediction_System")

def train_production_model():
    with mlflow.start_run() as run:
        print("🚀 Starting Production Training Pipeline...")
        
        # 1. LOAD DATA
        # Update path if needed
        df = pd.read_csv('data/Final_data.csv')
        
        # 2. CLEANING (MATCHING YOUR NOTEBOOK)
        # Standardize columns
        df.columns = [x.lower() for x in df.columns]
        
        # Round/Int conversion
        cols_to_round = ['age', 'workout_frequency (days/week)', 'experience_level', 'daily meals frequency']
        for col in cols_to_round:
            if col in df.columns:
                df[col] = df[col].round().astype(int)
        
        # Drop "Useless" columns as per your notebook
        drop_cols = ['meal_name','benefit','workout','calories', 'cal_balance', 
                     'burns calories (per 30 min)','burns calories (per 30 min)_bc',
                     'burns_calories_bin','expected_burn','cal_from_macros', 
                     'physical exercise', 'rating']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        
        # Drop Outliers
        df = df[(df['calories_burned'] <= 2500) & (df['calories_burned'] >= 0)]
        df = df.dropna()
        
        # 3. FEATURE ENGINEERING
        # Log Transform Target (CRITICAL STEP FROM NOTEBOOK)
        df['log_calories_burned'] = np.log(df['calories_burned'])
        
        # Separate Target
        X = df.drop(columns=['calories_burned', 'log_calories_burned', 'bmi_calc', 'max_bpm'], errors='ignore')
        y = df['log_calories_burned']
        
        # One-Hot Encoding
        X_encoded = pd.get_dummies(X)
        
        # SAVE THE COLUMN LIST (CRITICAL FOR APP)
        # The App needs to know exactly which columns (e.g., 'gender_Male', 'gender_Female') the model expects.
        model_columns = list(X_encoded.columns)
        with open("model_columns.pkl", "wb") as f:
            pickle.dump(model_columns, f)
        mlflow.log_artifact("model_columns.pkl", artifact_path="model")
            
        # 4. TRAIN MODEL (Using Random Forest as the Winner)
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 5. EVALUATE
        # Predict Log values
        y_pred_log = model.predict(X_test)
        
        # Convert back to Real values (Exp)
        y_pred_real = np.exp(y_pred_log)
        y_test_real = np.exp(y_test)
        
        mae = mean_absolute_error(y_test_real, y_pred_real)
        r2 = r2_score(y_test_real, y_pred_real)
        
        print(f"✅ Model Performance -> MAE: {mae:.2f} calories, R2: {r2:.2f}")
        
        # 6. LOGGING
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model", registered_model_name="Calorie_Predictor")
        
        print("Model and Feature List saved to MLflow!")

if __name__ == "__main__":
    train_production_model()