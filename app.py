import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import streamlit as st

# App Layout
st.set_page_config(page_title="📈 Financial Market Forecast", layout="wide")
st.title("📈 Predictive Analysis of Financial Market Trends")
st.caption("Created by Drashti Mehta | MS in Data Science")

# Sidebar inputs
st.sidebar.header("📊 Stock Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime(2000, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.today())
forecast_days = st.sidebar.slider("Forecast Days", min_value=3, max_value=30, value=7)

# Load data
data = yf.download(ticker, start=start_date, end=end_date)
if data.empty:
    st.warning("No data found. Check ticker or dates.")
    st.stop()

# Feature Engineering
data['MA10'] = data['Close'].rolling(10).mean()
data['MA20'] = data['Close'].rolling(20).mean()
data['Volatility'] = data['Close'].rolling(10).std()
data['Momentum'] = data['Close'].pct_change(periods=5)
data['Return'] = data['Close'].pct_change()
data.dropna(inplace=True)

# Market Summary
st.subheader(f"📌 Market Summary for {ticker}")
col1, col2, col3 = st.columns(3)
col1.metric("Last Close", f"${data['Close'].iloc[-1]:.2f}")
col2.metric("52-Week High", f"${data['Close'].max():.2f}")
col3.metric("52-Week Low", f"${data['Close'].min():.2f}")

# Risk Indicators
st.subheader("🚦 Risk Indicators")
r1, r2, r3 = st.columns(3)
r1.metric("📈 10-Day Volatility", f"{data['Volatility'].iloc[-1]:.4f}")
r2.metric("🚀 5-Day Momentum", f"{data['Momentum'].iloc[-1]*100:.2f}%")
r3.metric("📉 1-Day Return", f"{data['Return'].iloc[-1]*100:.2f}%")

# Forecast using Linear Regression
X = np.arange(len(data)).reshape(-1, 1)
y = data['Close'].values.reshape(-1, 1)
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)

future_X = np.arange(len(data), len(data) + forecast_days).reshape(-1, 1)
future_preds = model.predict(future_X)
future_dates = pd.date_range(data.index[-1] + timedelta(days=1), periods=forecast_days)

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Actual", line=dict(color='royalblue')))
fig.add_trace(go.Scatter(x=future_dates, y=future_preds.flatten(), name="Forecast", line=dict(color='orangered', dash='dash')))
fig.update_layout(title=f"{ticker} Price Forecast (+{forecast_days} Days)", xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig, use_container_width=True)

# Data Table
st.subheader("📄 Latest Data Snapshot")
st.dataframe(data.tail(10))

# Model Metrics
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
st.subheader("📊 Model Performance")
st.write(f"**RMSE:** {rmse:.2f} | **R² Score:** {r2:.2f}")

# CSV Download
st.download_button("📥 Download CSV", data.to_csv().encode(), f"{ticker}_data.csv", mime='text/csv')
