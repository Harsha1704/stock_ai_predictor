import streamlit as st
import requests
from charts.candlestick_plot import get_candlestick

st.set_page_config(page_title="AI Stock Predictor", layout="centered")
st.title("\U0001F4C8 AI Stock/Crypto Market Predictor")

symbol = st.text_input("Enter Stock/Crypto Symbol (e.g., RELIANCE.NS, BTC-USD)", "RELIANCE.NS")

if st.button("Predict"):
    try:
        url = f"http://127.0.0.1:8000/predict?symbol={symbol}"
        res = requests.get(url)
        data = res.json()

        if "error" in data:
            st.error(data["error"])
        else:
            st.success(f"Prediction: **{data['prediction']}** with Confidence: {data['confidence'] * 100:.2f}%")
            st.plotly_chart(get_candlestick(symbol), use_container_width=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")