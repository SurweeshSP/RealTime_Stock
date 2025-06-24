import streamlit as st
import requests
import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Load .env
load_dotenv()
api_key = os.getenv('API_KEY', 'demo')

# Streamlit UI
st.title("📈 Real-Time Stock Prediction App (Linear Regression & ARIMA)")
st.markdown("Built with Alpha Vantage + Machine Learning models.")

# Sidebar input
symbol = st.sidebar.text_input("Enter Stock Symbol", "IBM")
interval = st.sidebar.selectbox("Select Interval", ["1min", "5min", "15min", "30min", "60min"])

if st.sidebar.button("Fetch & Analyze"):

    # Alpha Vantage API call
    url = (
        f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY"
        f"&symbol={symbol}&interval={interval}&apikey={api_key}&outputsize=compact"
    )

    r = requests.get(url)
    data = r.json()
    time_series = data.get(f"Time Series ({interval})", {})

    if not time_series:
        st.error("Failed to retrieve data. Please check your API key, symbol, or interval.")
    else:
        # DataFrame creation
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.astype(float)
        df.columns = ["open", "high", "low", "close", "volume"]

        st.subheader("📊 Last 5 Rows of Stock Data")
        st.dataframe(df.tail())

        # ========= Linear Regression =========
        X = df[['open', 'high', 'low', 'volume']].values
        y = df['close'].values
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        lr_model = LinearRegression()
        lr_model.fit(x_train, y_train)
        y_pred = lr_model.predict(x_test)

        st.subheader("🧮 Linear Regression Metrics")
        st.write(f"**Mean Squared Error (MSE):** {mean_squared_error(y_test, y_pred):.4f}")
        st.write(f"**Mean Absolute Error (MAE):** {mean_absolute_error(y_test, y_pred):.4f}")
        st.write(f"**R² Score:** {r2_score(y_test, y_pred):.4f}")

        st.subheader("🔄 Actual vs Predicted (Sample)")
        diff_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        st.dataframe(diff_df.head())

        warnings.filterwarnings("ignore")
        series = df['close']
        arima_model = ARIMA(series, order=(5, 1, 0))
        model_fit = arima_model.fit()
        forecast = model_fit.forecast(steps=1)

        st.subheader("🔮 ARIMA Model Forecast")
        st.write(f"**Next Predicted Close Price (ARIMA):** {forecast.values[0]:.2f}")
        current_price = df['close'].iloc[-1]
        predicted_price = forecast.values[0]

        price_diff = predicted_price - current_price
        price_diff_pct = (price_diff / current_price) * 100

        st.subheader("🧠 Recommendation Engine")

        if r2_score(y_test, y_pred) < 0.5:
            st.warning("⚠️ Model confidence is low (R² < 0.5). Proceed cautiously.")

        if price_diff_pct > 0.5:
            st.success(f"✅ Recommendation: BUY — Expected gain of {price_diff_pct:.2f}%")
        elif price_diff_pct < -0.5:
            st.error(f"❌ Recommendation: DO NOT BUY — Expected drop of {abs(price_diff_pct):.2f}%")
        else:
            st.info("📊 Recommendation: HOLD — No significant price movement expected.")
