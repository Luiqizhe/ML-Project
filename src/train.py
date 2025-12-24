# src/train.py
# SET UP BY MLOPS (YOU)
import mlflow
import pandas as pd
# ... other imports

mlflow.set_tracking_uri("sqlite:///mlflow.db")

def main():
    with mlflow.start_run():
        # ==========================================
        # ML ENGINEER ZONE (THEY WRITE THIS PART)
        # ==========================================
        # TODO: Load your data here
        # TODO: Train your model here
        # TODO: Calculate your metrics here
        
        # Example (They will replace this):
        # model = ...
        # accuracy = ...
        # ==========================================

        # ==========================================
        # MLOPS ZONE (YOU WRITE THIS PART)
        # ==========================================
        # You ensure their hard work is actually saved
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model", registered_model_name="Fitness")
        print("Model versioned and saved.")

if __name__ == "__main__":
    main()