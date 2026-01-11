import os
import joblib
import streamlit as st
import pandas as pd

# Resolve project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")

# Load trained pipeline
model = joblib.load(MODEL_PATH)

st.title("Diabetes Risk Prediction Dashboard")

st.sidebar.header("Patient Metrics")

inputs = {
    "Pregnancies": st.sidebar.slider("Pregnancies", 0, 20, 1),
    "Glucose": st.sidebar.slider("Glucose", 50, 200, 120),
    "BloodPressure": st.sidebar.slider("Blood Pressure", 40, 130, 70),
    "SkinThickness": st.sidebar.slider("Skin Thickness", 0, 100, 20),
    "Insulin": st.sidebar.slider("Insulin", 0, 900, 80),
    "BMI": st.sidebar.slider("BMI", 15.0, 50.0, 28.0),
    "DiabetesPedigreeFunction": st.sidebar.slider("Diabetes Pedigree", 0.0, 2.5, 0.5),
    "Age": st.sidebar.slider("Age", 18, 90, 45),
}

input_df = pd.DataFrame([inputs])

prob = model.predict_proba(input_df)[0][1]

st.metric("Predicted Diabetes Risk", f"{prob:.2%}")
