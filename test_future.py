import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ===== Load Model & Scaler =====
@st.cache_resource
def load_artifacts():
    model = joblib.load("model_future.pkl")
    scaler = joblib.load("scaler_future.gz")
    return model, scaler

model, scaler = load_artifacts()

# ===== Load Historical Data =====
@st.cache_data
def load_data():
    df = pd.read_csv("USD-INR.csv", parse_dates=["Date"])
    df = df.sort_values("Date")
    df = df[["Date", "Close"]].rename(columns={"Close":"rate"})
    df = df.set_index("Date")
    return df

df = load_data()


# ===== Prediction Function =====
def predict_inr_amount(year: int, amount_usd: float) -> float:
    """
    Predict INR amount for a USD amount in a target future year.
    Returns INR value (float).
    """
    current_year = datetime.now().year
    if year <= current_year:
        raise ValueError("Year must be greater than the current year")

    # Convert year distance to approximate days
    # (for RandomForest this is symbolic; same prediction each future day)
    days_ahead = (year - current_year) * 365

    # Last available rate data
    last_rate = df["rate"].iloc[-1]
    last_7 = df["rate"].iloc[-7]
    last_30 = df["rate"].iloc[-30]

    # Scale inputs
    X = scaler.transform([[last_rate, last_7, last_30]])

    # Predict exchange rate (1 USD → INR)
    predicted_rate = model.predict(X)[0]

    # Convert USD → INR
    inr_value = predicted_rate * amount_usd
    return round(inr_value, 2), round(predicted_rate, 2)


# ===== Streamlit UI =====
st.set_page_config(page_title="USD to INR Future Forecast", page_icon="💱")

st.title("💱 USD → INR Future Value Prediction")
st.write("Predict the future value of USD converted to INR using a trained ML model.")

# User Inputs
year = st.number_input("Enter Future Year (e.g., 2027)", min_value=2025, max_value=2050, value=2027)
amount_usd = st.number_input("Enter Amount in USD", min_value=1.0, value=100.0)

# Predict Button
if st.button("Predict Future INR Value"):
    try:
        inr_value, predicted_rate = predict_inr_amount(year, amount_usd)

        st.success(f"📌 Predicted Exchange Rate in {year}: **₹{predicted_rate:,.2f} per 1 USD**")
        st.info(f"💰 {amount_usd:,.2f} USD ≈ **₹{inr_value:,.2f} INR** in {year}")

        # Optional: show a short summary
        st.write(
            f"Based on the model, if you have **${amount_usd:,.2f} USD**, "
            f"it may be worth approximately **₹{inr_value:,.2f} INR** in **{year}**."
        )

    except ValueError as e:
        st.error(str(e))
