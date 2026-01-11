import streamlit as st
import pandas as pd
import joblib

st.title("Diabetes Risk Prediction Dashboard")

model = joblib.load("model.joblib")

st.sidebar.header("Patient Metrics")

inputs = {
    "Pregnancies": st.sidebar.slider("Pregnancies", 0, 20, 1),
    "Glucose": st.sidebar.slider("Glucose", 50, 200, 120),
    "BMI": st.sidebar.slider("BMI", 15.0, 50.0, 28.0),
    "Age": st.sidebar.slider("Age", 18, 90, 45),
}

input_df = pd.DataFrame([inputs])

prob = model.predict_proba(input_df)[0][1]

st.metric(
    "Predicted Diabetes Risk",
    f"{prob:.2%}"
)
