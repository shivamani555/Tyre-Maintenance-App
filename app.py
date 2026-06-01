import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("tyre_model.pkl")

st.set_page_config(
    page_title="Tyre Maintenance App",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Tyre Maintenance & Health Prediction")
st.write(
    "Machine Learning based Tyre Monitoring System"
)

# Inputs

distance = st.number_input(
    "Distance Travelled (km)",
    min_value=0
)

load_weight = st.number_input(
    "Load Weight (kg)",
    min_value=0
)

pressure = st.number_input(
    "Tyre Pressure (PSI)",
    min_value=0
)

temperature = st.number_input(
    "Temperature (°C)",
    min_value=0
)

tyre_age = st.number_input(
    "Tyre Age (Years)",
    min_value=0
)

if st.button("Predict"):

    data = np.array([[
        distance,
        load_weight,
        pressure,
        temperature,
        tyre_age
    ]])

    prediction = model.predict(data)[0]

    # Health Score Formula

    health_score = 100

    health_score -= distance / 1000
    health_score -= load_weight / 200
    health_score -= (35 - pressure) * 2
    health_score -= temperature / 5
    health_score -= tyre_age * 5

    health_score = max(0, min(100, health_score))

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠ Maintenance Required")
    else:
        st.success("✅ Tyre Condition Good")

    st.metric(
        label="Tyre Health Score",
        value=f"{health_score:.1f}%"
    )