import streamlit as st
import mlflow.sklearn
import pandas as pd

# 1. SETUP: Point to the database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

st.title("💪 Fitness Calorie Predictor")

@st.cache_resource
def load_production_model():
    model_name = "Calorie_Predictor_Prod"
    # This fetches specifically the model you tagged as "Production"
    # It dynamically updates when you change the tag in the UI!
    model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/Production")
    return model

try:
    model = load_production_model()
    st.success("✅ Connected to Production Model Pipeline")
except Exception as e:
    st.error("Could not connect to MLflow. Make sure the server is running.")

# ... rest of your UI code (inputs, buttons, etc.) ...

if st.button("Predict"):
    # ... create input_df ...
    prediction = model.predict(input_df)
    st.write(f"Prediction: {prediction[0]}")