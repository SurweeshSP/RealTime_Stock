import streamlit as st
import requests
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import warnings

# 🔐 API Key
try:
    api_key = st.secrets["API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("API_KEY", "demo")

st.set_page_config(page_title="Real-Time Stock Predictor", layout="centered")

st.title("📈 Real-Time Stock Prediction App")
st.markdown("Built with **Alpha Vantage**, Linear Regression, and ARIMA.")
st.markdown("---")

# Sidebar Inputs
symbol = st.sidebar.text_input("Enter Stock Symbol", "IBM")
interval = st.sidebar.selectbox("Select Interval", ["1min", "5min", "15min", "30min", "60min"])

if st.sidebar.button("🚀 Fetch & Analyze"):

    # API Call
    url = (
        f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY"
        f"&symbol={symbol}&interval={interval}&apikey={api_key}&outputsize=compact"
    )

    r = requests.get(url)
    data = r.json()
    time_series = data.get(f"Time Series ({interval})", {})

    if not time_series:
        st.error("❌ Failed to retrieve data. Check API key or symbol.")
    else:
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().astype(float)
        df.columns = ["open", "high", "low", "close", "volume"]

        st.subheader("📊 Last 5 Rows")
        st.dataframe(df.tail())

        # Linear Regression
        X = df[['open', 'high', 'low', 'volume']].values
        y = df['close'].values
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        lr = LinearRegression()
        lr.fit(x_train, y_train)
        y_pred = lr.predict(x_test)

        r2 = r2_score(y_test, y_pred)
        st.subheader("📉 Linear Regression Metrics")
        st.write(f"**MSE**: {mean_squared_error(y_test, y_pred):.4f}")
        st.write(f"**MAE**: {mean_absolute_error(y_test, y_pred):.4f}")
        st.write(f"**R² Score**: {r2:.4f}")

        st.subheader("🔍 Actual vs Predicted (Sample)")
        st.dataframe(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).head())

        # ARIMA Prediction
        warnings.filterwarnings("ignore")
        arima_model = ARIMA(df['close'], order=(5, 1, 0))
        model_fit = arima_model.fit()
        forecast = model_fit.forecast(steps=1)

        # Price analysis
        current_price = df['close'].iloc[-1]
        predicted_price = forecast.values[0]
        price_diff = predicted_price - current_price
        price_diff_pct = (price_diff / current_price) * 100

        moving_avg = df['close'].tail(10).mean()
        std_dev = df['close'].tail(10).std()

        # Flags (Relaxed)
        low_confidence = r2 < 0.4
        strong_signal = abs(price_diff) > std_dev * 0.1
        above_avg = predicted_price > moving_avg * 1.005
        below_avg = predicted_price < moving_avg * 0.995

        st.subheader("🧠 Recommendation Engine")
        if low_confidence:
            st.warning("⚠️ R² < 0.4: Model confidence is low.")

        if not low_confidence and strong_signal:
            if above_avg and price_diff_pct > 0.2:
                st.success(f"📈 BUY — Expected gain: **{price_diff_pct:.2f}%**")
            elif below_avg and price_diff_pct < -0.2:
                st.error(f"📉 SELL — Expected drop: **{abs(price_diff_pct):.2f}%**")
            else:
                st.info("🔄 HOLD — Small movement within range.")
        else:
            st.info("📉 HOLD — Weak signal or low confidence.")

        # 🧪 Debug Info
        st.markdown("### 🧪 Debug Info")
        st.code(f"""
Current: {current_price:.2f}
Predicted: {predicted_price:.2f}
Δ%: {price_diff_pct:.2f}%
R²: {r2:.4f}
Moving Avg (10): {moving_avg:.2f}
Std Dev (10): {std_dev:.2f}
Above Avg: {above_avg}
Below Avg: {below_avg}
Strong Signal: {strong_signal}
Low Confidence: {low_confidence}
""")
