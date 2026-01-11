import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np

# Resolve project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

# Load trained pipeline / model
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

# Convert inputs to DataFrame
input_df = pd.DataFrame([inputs])

# ---- SAFE PREDICTION (FIX IS HERE) ----
raw_pred = model.predict(input_df)

# Always convert prediction safely to a scalar
prob = float(np.ravel(raw_pred)[0])
# -------------------------------------

st.metric("Predicted Diabetes Risk", f"{prob:.2%}")

# Optional interpretation
if prob >= 0.5:
    st.error("High diabetes risk")
else:
    st.success("Low diabetes risk")
