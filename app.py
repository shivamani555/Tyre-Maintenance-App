import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("tyre_model.pkl")

st.set_page_config(
    page_title="Tyre Maintenance Dashboard",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Tyre Maintenance and Operation App")
st.write("Machine Learning Based Predictive Maintenance System")

st.sidebar.header("Enter Tyre Details")

distance = st.sidebar.number_input(
    "Distance Travelled (km)",
    min_value=0,
    value=10000
)

load_weight = st.sidebar.number_input(
    "Load Weight (kg)",
    min_value=0,
    value=1200
)

pressure = st.sidebar.number_input(
    "Tyre Pressure (PSI)",
    min_value=0,
    value=32
)

temperature = st.sidebar.number_input(
    "Temperature (°C)",
    min_value=0,
    value=35
)

tyre_age = st.sidebar.number_input(
    "Tyre Age (Years)",
    min_value=0,
    value=2
)

if st.sidebar.button("Predict"):

    input_data = np.array([[
        distance,
        load_weight,
        pressure,
        temperature,
        tyre_age
    ]])

    prediction = model.predict(input_data)[0]

    health_score = 100

    health_score -= distance / 1000
    health_score -= load_weight / 200
    health_score -= max(0, (35-pressure)*2)
    health_score -= temperature / 5
    health_score -= tyre_age * 5

    health_score = max(0, min(100, health_score))

    st.header("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:

        if prediction == 1:
            st.error("⚠ Maintenance Required")
        else:
            st.success("✅ Tyre Condition Good")

    with col2:

        st.metric(
            "Tyre Health Score",
            f"{health_score:.1f}%"
        )

        st.progress(int(health_score))

    if health_score >= 80:
        st.success("🟢 Excellent Condition")

    elif health_score >= 60:
        st.warning("🟡 Moderate Condition")

    else:
        st.error("🔴 Immediate Maintenance Required")

    st.divider()

    st.subheader("🔍 Root Cause Analysis")

    causes = []

    if pressure < 28:
        causes.append("Low Tyre Pressure")

    if temperature > 45:
        causes.append("High Operating Temperature")

    if load_weight > 1800:
        causes.append("Excessive Vehicle Load")

    if tyre_age > 4:
        causes.append("Tyre Aging")

    if distance > 20000:
        causes.append("Excessive Distance Travelled")

    if causes:

        for cause in causes:
            st.warning(cause)

    else:
        st.success("No major issues detected")

    st.divider()

    st.subheader("🛡 Prevention Recommendations")

    if pressure < 28:
        st.write("✔ Maintain pressure between 30-35 PSI")

    if temperature > 45:
        st.write("✔ Reduce heat exposure and monitor temperature")

    if load_weight > 1800:
        st.write("✔ Reduce load weight")

    if tyre_age > 4:
        st.write("✔ Schedule tyre replacement")

    if distance > 20000:
        st.write("✔ Inspect tread depth and wear")

    if len(causes) == 0:
        st.write("✔ Continue regular maintenance")

    st.divider()

    st.subheader("🔧 Tyre Maintenance Guide")

    st.markdown("""
    - Check tyre pressure every week
    - Rotate tyres every 8,000 km
    - Check wheel alignment every 10,000 km
    - Avoid overloading vehicles
    - Inspect tread depth monthly
    - Replace tyres after 5–6 years
    - Monitor temperature during long trips
    """)

    st.divider()

    st.subheader("📊 Parameter Dashboard")

    chart_data = pd.DataFrame({
        "Parameter":[
            "Distance",
            "Load",
            "Pressure",
            "Temperature",
            "Age"
        ],
        "Value":[
            distance,
            load_weight,
            pressure,
            temperature,
            tyre_age
        ]
    })

    st.bar_chart(
        chart_data.set_index("Parameter")
    )

    st.subheader("📋 Input Summary")

    st.dataframe(chart_data)
