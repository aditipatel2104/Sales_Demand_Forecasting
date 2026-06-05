import streamlit as st
import joblib

model = joblib.load("model/sales_model.pkl")

st.title("Sales Demand Forecasting")

festival = st.selectbox(
    "Festival Season",
    [0, 1]
)

inflation = st.number_input(
    "Inflation Rate",
    value=5.0
)

temperature = st.number_input(
    "Temperature",
    value=25
)

day = st.slider(
    "Day",
    1,
    31,
    15
)

month = st.slider(
    "Month",
    1,
    12,
    6
)

if st.button("Predict Sales"):

    result = model.predict(
        [[festival,
          inflation,
          temperature,
          day,
          month]]
    )

    st.success(
        f"Predicted Sales : {result[0]:.2f}"
    )