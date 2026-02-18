
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

sns.set(style="whitegrid")

def fetch_stock_data(ticker, start, end, market):
    try:
        if market.upper() == 'BSE':
            ticker += ".BO"
        elif market.upper() == 'NSE':
            ticker += ".NS"
        stock_data = yf.download(ticker, start=start, end=end)
        if stock_data.empty:
            raise ValueError("No data found for the given ticker.")
        return stock_data['Close']
    except Exception as e:
        print("Error fetching stock data:", e)
        return None

def train_arima_model(series):
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit

def predict_future(model_fit, series, future_days):
    forecast = model_fit.forecast(steps=future_days)
    last_date = series.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]
    return future_dates, forecast

def plot_stock_trend(series, model_fit, future_dates, future_prices, period_days):
    fig, ax = plt.subplots(figsize=(14, 7))

    if period_days <= 7:
        recent_days = 30
    elif period_days <= 30:
        recent_days = 90
    elif period_days <= 182:
        recent_days = 180
    elif period_days <= 365:
        recent_days = 365
    else:
        recent_days = 730

    recent_series = series[-recent_days:]
    ax.plot(recent_series.index, recent_series.values, color='blue', label='Recent Actual Prices', linestyle='solid')
    in_sample_pred = model_fit.predict(start=len(series) - recent_days, end=len(series) - 1, dynamic=False)
    ax.plot(recent_series.index, in_sample_pred, color='green', label='Recent Predicted Trend', linestyle='dashed')
    ax.plot(future_dates, future_prices, color='red', label='Future Prediction', linestyle='dashed', marker='o', markersize=3)

    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price (₹)')
    ax.set_title(f'Stock Price Prediction Starting {future_dates[0].strftime("%Y-%m-%d")} (Next {period_days} Days)')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    ax.grid(True, linestyle='--', linewidth=0.5)
    fig.tight_layout()

    return fig
