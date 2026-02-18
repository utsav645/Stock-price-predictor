from flask import Flask, render_template, request, url_for
from model import fetch_stock_data, train_arima_model, predict_future, plot_stock_trend
from datetime import datetime
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure static folder exists
    os.makedirs("static", exist_ok=True)

    ticker = request.form['ticker']
    market = request.form['market']
    duration = int(request.form['duration'])

    START_DATE = "2021-01-01"
    END_DATE = datetime.now().strftime('%Y-%m-%d')

    series = fetch_stock_data(ticker, START_DATE, END_DATE, market)
    if series is None:
        return render_template('index.html', error="Failed to fetch stock data.")

    model_fit = train_arima_model(series)
    future_dates, future_prices = predict_future(model_fit, series, duration)

    fig = plot_stock_trend(series, model_fit, future_dates, future_prices, duration)
    static_dir = os.path.join(app.root_path, 'static')
    os.makedirs(static_dir, exist_ok=True)  # Make sure it exists
    plot_filepath = os.path.join(static_dir, 'plot.png')
    fig.savefig(plot_filepath)

    plt.close(fig)

    plot_url = url_for('static', filename='plot.png')  # Fixed URL for browser
    predictions = zip([d.strftime('%Y-%m-%d') for d in future_dates], future_prices)
    return render_template('index.html', predictions=predictions, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
