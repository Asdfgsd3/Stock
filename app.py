import yfinance as yf
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import matplotlib.pyplot as plt
import matplotlib
import joblib
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, LSTM
import io
import base64
import datetime
import os
from sklearn.preprocessing import MinMaxScaler

# Set matplotlib backend to non-interactive
matplotlib.use('Agg')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    model_path = f'{ticker}_stock_predictor.h5'
    scaler_path = f'{ticker}_scaler.pkl'

    # Check if model and scaler exist, if not train a new model
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        train_model(ticker)

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    # Fetch Historical Data
    stock = yf.Ticker(ticker)
    data = stock.history(period="5y")
    close_data = data['Close'].values.reshape(-1, 1)

    # Prepare Input for Future Prediction
    scaled_data = scaler.transform(close_data)
    last_60_days = scaled_data[-60:]
    future_input = last_60_days.reshape(1, 60, 1)

    # Predict Future Values (Next 30 Days)
    future_predictions = []
    current_date = data.index[-1]

    for _ in range(30):
        predicted_price = model.predict(future_input)
        predicted_price = predicted_price.reshape(1, 1, 1)  # Reshape to match dimensions
        future_predictions.append(predicted_price[0, 0])
        future_input = np.append(future_input[:, 1:, :], predicted_price, axis=1)
        current_date += datetime.timedelta(days=1)

    # Reverse scaling
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Plot Predictions
    future_dates = pd.date_range(start=data.index[-1] + datetime.timedelta(days=1), periods=len(future_predictions))
    plt.figure(figsize=(14, 6))
    plt.plot(data.index, close_data, label='Actual Prices', color='green')
    plt.plot(future_dates, future_predictions, label='Future Predictions (Next 30 Days)', color='red')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.title(f'{ticker} Stock Price Prediction')
    
    # Convert plot to PNG image for HTML rendering
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()

    return f'<img src="data:image/png;base64,{graph_url}"/>'

def train_model(ticker):
    # Fetch Historical Stock Data
    stock = yf.Ticker(ticker)
    data = stock.history(period="5y")
    close_data = data['Close'].values.reshape(-1, 1)

    # Preprocess Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)
    time_step = 60
    X, y = [], []
    for i in range(len(scaled_data) - time_step):
        X.append(scaled_data[i:(i + time_step), 0])
        y.append(scaled_data[i + time_step, 0])
    X, y = np.array(X), np.array(y)

    # Split Data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Reshape for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build and Train LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=10)

    # Save Model and Scaler
    model.save(f'{ticker}_stock_predictor.h5')
    joblib.dump(scaler, f'{ticker}_scaler.pkl')

    print(f"Model and Scaler for {ticker} saved.")

if __name__ == "__main__":
    app.run(debug=True, port=5001)