import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load trained model and scalers
model = load_model("health_risk_ann_model.keras")
sc_X = joblib.load("health_risk_scaler_X.pkl")
sc_y = joblib.load("health_risk_scaler_y.pkl")

# Page settings
st.set_page_config(
    page_title="AI Health Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 AI Health Risk Predictor")
st.write("Enter the person's health information below.")

# Inputs
age = st.number_input("Age", min_value=1.0, max_value=120.0, value=45.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.5)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, value=120.0)
cholesterol = st.number_input("Cholesterol", min_value=0.0, value=190.0)
glucose = st.number_input("Glucose", min_value=0.0, value=110.0)
insulin = st.number_input("Insulin", min_value=0.0, value=15.0)
heart_rate = st.number_input("Heart Rate", min_value=0.0, value=75.0)
activity_level = st.number_input("Activity Level", min_value=0.0, value=6.0)
diet_quality = st.number_input("Diet Quality", min_value=0.0, value=7.0)

smoking_status = st.selectbox(
    "Smoking Status",
    ["No", "Yes"]
)

alcohol_intake = st.number_input(
    "Alcohol Intake",
    min_value=0.0,
    value=2.0
)

# Convert smoking status to the same format used during training
smoking_value = 1 if smoking_status == "Yes" else 0

# Prediction
if st.button("Predict Health Risk"):

    input_data = np.array([[
        age,
        bmi,
        blood_pressure,
        cholesterol,
        glucose,
        insulin,
        heart_rate,
        activity_level,
        diet_quality,
        smoking_value,
        alcohol_intake
    ]])

    # Scale input
    input_scaled = sc_X.transform(input_data)

    # Predict
    prediction_scaled = model.predict(input_scaled, verbose=0)

    # Convert prediction back to original scale
    prediction = sc_y.inverse_transform(
        prediction_scaled
    )

    risk_score = prediction[0][0]

    st.success(
        f"Predicted Health Risk Score: {risk_score:.2f}"
    )

