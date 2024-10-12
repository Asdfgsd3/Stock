import yfinance as yf
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import matplotlib.pyplot as plt
import matplotlib
import joblib
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import io
import base64
import datetime
import os
import logging
from sklearn.preprocessing import MinMaxScaler

# Set matplotlib backend to non-interactive
matplotlib.use('Agg')

# Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logging.error(f"Error loading index page: {str(e)}")
        return f"An error occurred while loading the index page: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        ticker = request.form['ticker'].upper()
        model_path = f'models/{ticker}_stock_predictor.h5'
        scaler_price_path = f'models/{ticker}_scaler_price.pkl'
        scaler_rsi_path = f'models/{ticker}_scaler_rsi.pkl'
        scaler_sma_path = f'models/{ticker}_scaler_sma.pkl'
        scaler_pe_path = f'models/{ticker}_scaler_pe.pkl'

        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')

        # Fetch Historical Data
        stock = yf.Ticker(ticker)
        data = stock.history(period='5y')
        pe_ratio = stock.info.get('trailingPE', np.nan)
        data['PE_Ratio'] = pe_ratio  # Extend period to 5 years to ensure enough data is available
        if data.empty:
            logging.warning(f"No historical data available for ticker: {ticker}")
            return f"No historical data available for the ticker: {ticker}. Please enter a valid ticker."
        data['RSI'] = calculate_rsi(data)
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        data.dropna(subset=['RSI', 'SMA_50', 'SMA_200', 'PE_Ratio'], inplace=True)
        if len(data) < 60:
            logging.warning(f"Not enough historical data for ticker: {ticker}")
            return f"Not enough historical data available for the ticker: {ticker} to make predictions."

        close_data = data['Close'].values.reshape(-1, 1)
        rsi_data = data['RSI'].values.reshape(-1, 1)
        sma_50_data = data['SMA_50'].values.reshape(-1, 1)
        sma_200_data = data['SMA_200'].values.reshape(-1, 1)
        pe_ratio_data = data['PE_Ratio'].values.reshape(-1, 1)

        # Check if model and scaler exist, if not train a new model
        if not os.path.exists(model_path) or not os.path.exists(scaler_price_path):
            logging.info(f"Training new model for ticker: {ticker}")
            train_model(ticker)

        model = load_model(model_path)
        scaler_price = joblib.load(scaler_price_path)
        scaler_rsi = joblib.load(scaler_rsi_path)
        scaler_sma = joblib.load(scaler_sma_path)
        scaler_pe = joblib.load(scaler_pe_path)

        # Prepare Input for Future Prediction
        scaled_close_data = scaler_price.transform(close_data)
        scaled_rsi_data = scaler_rsi.transform(rsi_data)
        scaled_sma_50_data = scaler_sma.transform(sma_50_data)
        scaled_sma_200_data = scaler_sma.transform(sma_200_data)
        scaled_pe_data = scaler_pe.transform(pe_ratio_data)

        last_60_days_close = scaled_close_data[-60:]
        last_60_days_rsi = scaled_rsi_data[-60:]
        last_60_days_sma_50 = scaled_sma_50_data[-60:]
        last_60_days_sma_200 = scaled_sma_200_data[-60:]
        last_60_days_pe = scaled_pe_data[-60:]

        if len(last_60_days_close) < 60:
            padding = 60 - len(last_60_days_close)
            last_60_days_close = np.pad(last_60_days_close, ((padding, 0), (0, 0)), 'constant', constant_values=0)
            last_60_days_rsi = np.pad(last_60_days_rsi, ((padding, 0), (0, 0)), 'constant', constant_values=0)
            last_60_days_sma_50 = np.pad(last_60_days_sma_50, ((padding, 0), (0, 0)), 'constant', constant_values=0)
            last_60_days_sma_200 = np.pad(last_60_days_sma_200, ((padding, 0), (0, 0)), 'constant', constant_values=0)
            last_60_days_pe = np.pad(last_60_days_pe, ((padding, 0), (0, 0)), 'constant', constant_values=0)

        last_60_days = np.hstack((last_60_days_close, last_60_days_rsi, last_60_days_sma_50, last_60_days_sma_200, last_60_days_pe))
        future_input = last_60_days.reshape(1, 60, 5)

        # Predict Future Values (Next 30 Days)
        future_predictions = []
        current_date = data.index[-1]

        for _ in range(30):
            predicted_price = model.predict(future_input)
            future_predictions.append(predicted_price[0, 0])
            predicted_value = np.array([predicted_price[0, 0], last_60_days_rsi[-1][0], last_60_days_sma_50[-1][0], last_60_days_sma_200[-1][0], last_60_days_pe[-1][0]]).reshape(1, 1, 5)
            future_input = np.append(future_input[:, 1:, :], predicted_value, axis=1)
            current_date += datetime.timedelta(days=1)

        # Reverse scaling
        future_predictions = scaler_price.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # Plot Predictions
        future_dates = pd.date_range(start=data.index[-1] + datetime.timedelta(days=1), periods=len(future_predictions))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), gridspec_kw={'height_ratios': [2, 1]})
        fig.patch.set_facecolor('#f0f2f5')

        # Plot stock prices
        ax1.plot(data.index[-60:], close_data.flatten()[-60:], label='Actual Prices', color='blue', linewidth=2)
        ax1.plot(np.append(data.index[-1:], future_dates), np.concatenate((close_data[-1:].flatten(), future_predictions.flatten())), label='Future Predictions (Next 30 Days)', color='orange', linewidth=2, linestyle='--')
        ax1.set_ylabel('Stock Price', fontsize=14, fontname='Arial')
        ax1.set_xlabel('Date', fontsize=14, fontname='Arial')
        ax1.legend(loc='upper left', fontsize=12, prop={'family': 'Arial'})
        ax1.set_title(f'{ticker} Stock Price Prediction', fontsize=18, fontname='Arial')
        ax1.grid(True, linestyle='--', linewidth=0.5)
        ax1.set_facecolor('#ffffff')

        # Plot RSI underneath the stock price graph
        ax2.plot(data.index[-60:], data['RSI'][-60:], label='RSI', color='black', alpha=1, linewidth=1.5)
        ax2.set_xlim(ax1.get_xlim())
        ax2.axhline(70, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Overbought (70)')
        ax2.axhline(30, color='green', linestyle='--', alpha=0.7, linewidth=1.5, label='Oversold (30)')
        ax2.set_xlabel('Date', fontsize=14, fontname='Arial')
        ax2.set_ylabel('RSI', fontsize=14, fontname='Arial')
        ax2.legend(loc='upper left', fontsize=12, prop={'family': 'Arial'})
        ax2.grid(True, linestyle='--', linewidth=0.5)
        ax2.set_facecolor('#ffffff')

        plt.tight_layout()
        
        # Convert plot to PNG image for HTML rendering
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()

        return f'<div style="background-color: #f0f2f5; padding: 20px; text-align: center;"><h1 style="color: #007bff; font-family: Arial;">{ticker} Stock Price Prediction</h1><img src="data:image/png;base64,{graph_url}" style="max-width: 90%; height: auto; display: block; margin: 0 auto; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); border-radius: 10px;"></div>'
    except Exception as e:
        logging.error(f"Error during prediction for ticker {ticker}: {str(e)}")
        return f"An error occurred: {str(e)}"

def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def train_model(ticker):
    # Fetch Historical Stock Data
    stock = yf.Ticker(ticker)
    data = stock.history(period="5y")  # Extend period to 5 years to ensure enough data is available
    if data.empty:
        logging.error(f"No historical data available for ticker: {ticker}")
        raise ValueError(f"No historical data available for the ticker: {ticker}")
    data['RSI'] = calculate_rsi(data)
    data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50-day Simple Moving Average
    data['SMA_200'] = data['Close'].rolling(window=200).mean()  # 200-day Simple Moving Average
    data['PE_Ratio'] = stock.info.get('trailingPE', np.nan)
    data.dropna(subset=['RSI', 'SMA_50', 'SMA_200', 'PE_Ratio'], inplace=True)
    if len(data) < 60:
        logging.error(f"Not enough historical data for ticker: {ticker} to train the model.")
        raise ValueError(f"Not enough historical data available for the ticker: {ticker} to train the model.")
    
    close_data = data['Close'].values.reshape(-1, 1)
    rsi_data = data['RSI'].values.reshape(-1, 1)
    sma_50_data = data['SMA_50'].values.reshape(-1, 1)
    sma_200_data = data['SMA_200'].values.reshape(-1, 1)
    pe_ratio_data = data['PE_Ratio'].values.reshape(-1, 1)

    # Preprocess Data
    scaler_price = MinMaxScaler(feature_range=(0, 1))
    scaler_rsi = MinMaxScaler(feature_range=(0, 1))
    scaler_sma = MinMaxScaler(feature_range=(0, 1))
    scaler_pe = MinMaxScaler(feature_range=(0, 1))
    scaled_close_data = scaler_price.fit_transform(close_data)
    scaled_rsi_data = scaler_rsi.fit_transform(rsi_data)
    scaled_sma_50_data = scaler_sma.fit_transform(sma_50_data)
    scaled_sma_200_data = scaler_sma.fit_transform(sma_200_data)
    scaled_pe_data = scaler_pe.fit_transform(pe_ratio_data)

    time_step = 60
    X, y = [], []
    for i in range(len(scaled_close_data) - time_step):
        X.append(np.hstack((scaled_close_data[i:(i + time_step), 0].reshape(-1, 1),
                            scaled_rsi_data[i:(i + time_step), 0].reshape(-1, 1),
                            scaled_sma_50_data[i:(i + time_step), 0].reshape(-1, 1),
                            scaled_sma_200_data[i:(i + time_step), 0].reshape(-1, 1),
                            scaled_pe_data[i:(i + time_step), 0].reshape(-1, 1))))
        y.append(scaled_close_data[i + time_step, 0])
    X, y = np.array(X), np.array(y)

    # Split Data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Reshape for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 5)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 5)

    # Build and Train LSTM Model
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(time_step, 5)))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=20)

    # Save Model and Scaler
    model.save(f'models/{ticker}_stock_predictor.h5')
    joblib.dump(scaler_price, f'models/{ticker}_scaler_price.pkl')
    joblib.dump(scaler_rsi, f'models/{ticker}_scaler_rsi.pkl')
    joblib.dump(scaler_sma, f'models/{ticker}_scaler_sma.pkl')
    joblib.dump(scaler_pe, f'models/{ticker}_scaler_pe.pkl')

    logging.info(f"Model and Scaler for {ticker} saved.")

if __name__ == "__main__":
    app.run(debug=True, port=5001)