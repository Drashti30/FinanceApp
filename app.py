import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import streamlit as st

# 🎯 Page setup
st.set_page_config(page_title="📈 Market Forecast", layout="wide")
st.title("📈 Predictive Analysis of Financial Market Trends")
st.caption("Created by Drashti Mehta | MS in Data Science")

# 🎛 Sidebar inputs
st.sidebar.header("🛠️ Stock Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", datetime(2000, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())
forecast_days = st.sidebar.slider("Forecast Days", 3, 30, 7)

# 📥 Load data
data = yf.download(ticker, start=start_date, end=end_date)
data.dropna(inplace=True)

if data.empty:
    st.error("⚠️ No data found. Please check the stock symbol or date range.")
    st.stop()

# 📌 Market summary
st.subheader(f"📌 Market Summary for {ticker}")
col1, col2, col3 = st.columns(3)

if 'Close' in data.columns:
    last_close = float(data['Close'].iloc[-1])
    col1.metric("💱 Last Close", f"${last_close:.2f}")
    col2.metric("📈 52-Week High", f"${float(data['Close'].max()):.2f}")
    col3.metric("📉 52-Week Low", f"${float(data['Close'].min()):.2f}")

# 🧠 Feature engineering
data['MA10'] = data['Close'].rolling(10).mean()
data['MA20'] = data['Close'].rolling(20).mean()
data['Volatility'] = data['Close'].rolling(10).std()
data['Momentum'] = data['Close'].pct_change(periods=5)
data['Return'] = data['Close'].pct_change()
data.dropna(inplace=True)

# 🚦 Risk indicators
st.subheader("🚦 Risk Indicators")
r1, r2, r3 = st.columns(3)
r1.metric("📊 10-Day Volatility", f"{data['Volatility'].iloc[-1]:.4f}")
r2.metric("🚀 5-Day Momentum", f"{data['Momentum'].iloc[-1] * 100:.2f}%")
r3.metric("🔁 1-Day Return", f"{data['Return'].iloc[-1] * 100:.2f}%")

# 🔧 Model training
X = np.arange(len(data)).reshape(-1, 1)
y = data['Close'].values.reshape(-1, 1)
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)

rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

# 🔮 Forecasting
future_X = np.arange(len(data), len(data) + forecast_days).reshape(-1, 1)
future_preds = model.predict(future_X)
future_dates = pd.date_range(data.index[-1] + timedelta(days=1), periods=forecast_days)

# 📊 Forecast plot
st.subheader(f"{ticker} Price Forecast (+{forecast_days} days)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="📈 Actual", line=dict(color='royalblue')))
fig.add_trace(go.Scatter(x=future_dates, y=future_preds.flatten(), name="🧭 Forecast", line=dict(color='orangered', dash='dash')))
fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# 🔍 Data preview
st.subheader("📄 Engineered Data Sample")
st.dataframe(data.tail(10))

# 📥 CSV download
csv = data.to_csv().encode()
st.download_button("📥 Download CSV", data=csv, file_name=f"{ticker}_data.csv", mime='text/csv')

# 📐 Model metrics
st.subheader("📈 Model Performance")
st.markdown(f"**Root Mean Squared Error (RMSE):** `{rmse:.2f}`")
st.markdown(f"**R² Score:** `{r2:.2f}`")
