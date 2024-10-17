import yfinance as yf
import numpy as np
import pandas as pd
from flask import Flask, request
import plotly.graph_objs as go
import plotly.offline as pyo
import joblib
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import datetime
import os
import logging
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, render_template

# Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

class RSI:
    def __init__(self, period):
        self.period = period
        self.prev_avg_gain = 0
        self.prev_avg_loss = 0
        self.rsi = []

    def compute_moving_average(self, series):
        if len(self.rsi) == 0:
            sum_gain = 0
            sum_loss = 0
            for v1, v2 in zip(series[:-1], series[1:]):
                if v2 > v1:
                    sum_gain += v2 - v1
                elif v1 > v2:
                    sum_loss += v1 - v2
            avg_gain = sum_gain / self.period
            avg_loss = sum_loss / self.period
        else:
            if series[-1] >= series[-2]:
                gain = series[-1] - series[-2]
                loss = 0
            elif series[-2] > series[-1]:
                gain = 0
                loss = series[-2] - series[-1]
            avg_gain = (gain + ((self.period - 1) * self.prev_avg_gain)) / self.period
            avg_loss = (loss + ((self.period - 1) * self.prev_avg_loss)) / self.period
        self.prev_avg_gain = avg_gain
        self.prev_avg_loss = avg_loss
        return avg_gain, avg_loss

    def compute_single_rsi(self, series):
        avg_gain, avg_loss = self.compute_moving_average(series)
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute_rsi(self, series):
        self.rsi = []
        for i in range(self.period, len(series)):
            self.rsi.append(self.compute_single_rsi(series[i - self.period:i + 1]))
        return self.rsi

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
        scaler_mfi_path = f'models/{ticker}_scaler_mfi.pkl'

        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')

        # Remove files older than 1 day
        now = datetime.datetime.now()
        for file in os.listdir('models'):
            file_path = os.path.join('models', file)
            if os.path.isfile(file_path):
                file_creation_time = datetime.datetime.fromtimestamp(os.path.getctime(file_path))
                if (now - file_creation_time).total_seconds() > 1:
                    os.remove(file_path)
                    logging.info(f"Removed old file: {file_path}")

        # Fetch Historical Data
        stock = yf.Ticker(ticker)
        data = stock.history(period='5y')
        pe_ratio = stock.info.get('trailingPE', np.nan)
        data['PE_Ratio'] = pe_ratio if not np.isnan(pe_ratio) else np.nan
        if data.empty:
            data = stock.history(period='max')
            
        rsi_calculator = RSI(period=14)
        data['RSI'] = pd.Series(rsi_calculator.compute_rsi(data['Close'].values), index=data.index[14:])
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        data['MFI'] = calculate_mfi(data)
        data.dropna(subset=['RSI', 'SMA_50', 'SMA_200', 'MFI'], inplace=True)
        if len(data) < 60:
            logging.warning(f"Not enough historical data for ticker: {ticker}")
            return f"Not enough historical data available for the ticker: {ticker} to make predictions."

        close_data = data['Close'].values.reshape(-1, 1)
        rsi_data = data['RSI'].values.reshape(-1, 1)
        sma_50_data = data['SMA_50'].values.reshape(-1, 1)
        sma_200_data = data['SMA_200'].values.reshape(-1, 1)
        pe_ratio_data = data['PE_Ratio'].values.reshape(-1, 1)
        mfi_data = data['MFI'].values.reshape(-1, 1)

        # Check if model and scaler exist, if not train a new model
        if not os.path.exists(model_path) or not os.path.exists(scaler_price_path) or not os.path.exists(scaler_rsi_path) or not os.path.exists(scaler_sma_path) or not os.path.exists(scaler_pe_path) or not os.path.exists(scaler_mfi_path):
            logging.info(f"Training new model for ticker: {ticker}")
            train_model(ticker)

        model = load_model(model_path)
        scaler_price = joblib.load(scaler_price_path)
        scaler_rsi = joblib.load(scaler_rsi_path)
        scaler_sma = joblib.load(scaler_sma_path)
        scaler_pe = joblib.load(scaler_pe_path)
        scaler_mfi = joblib.load(scaler_mfi_path)

        # Prepare Input for Future Prediction
        scaled_close_data = scaler_price.transform(close_data)
        scaled_rsi_data = scaler_rsi.transform(rsi_data)
        scaled_sma_50_data = scaler_sma.transform(sma_50_data)
        scaled_sma_200_data = scaler_sma.transform(sma_200_data)
        scaled_pe_data = scaler_pe.transform(pe_ratio_data)
        scaled_mfi_data = scaler_mfi.transform(mfi_data)

        last_60_days_close = scaled_close_data[-60:]
        last_60_days_rsi = scaled_rsi_data[-60:]
        last_60_days_sma_50 = scaled_sma_50_data[-60:]
        last_60_days_sma_200 = scaled_sma_200_data[-60:]
        last_60_days_pe = scaled_pe_data[-60:]
        last_60_days_mfi = scaled_mfi_data[-60:]

        if len(last_60_days_close) < 60:
            padding = 60 - len(last_60_days_close)
            last_60_days_close = np.pad(last_60_days_close, ((padding, 0), (0, 0)), 'constant', constant_values=0)
            last_60_days_rsi = np.pad(last_60_days_rsi, ((padding, 0), (0, 0)), 'constant', constant_values=0)
            last_60_days_sma_50 = np.pad(last_60_days_sma_50, ((padding, 0), (0, 0)), 'constant', constant_values=0)
            last_60_days_sma_200 = np.pad(last_60_days_sma_200, ((padding, 0), (0, 0)), 'constant', constant_values=0)
            last_60_days_pe = np.pad(last_60_days_pe, ((padding, 0), (0, 0)), 'constant', constant_values=0)
            last_60_days_mfi = np.pad(last_60_days_mfi, ((padding, 0), (0, 0)), 'constant', constant_values=0)

        last_60_days = np.hstack((last_60_days_close, last_60_days_rsi, last_60_days_sma_50, last_60_days_sma_200, last_60_days_pe, last_60_days_mfi))
        future_input = last_60_days.reshape(1, 60, 6)

        # Predict Future Values (Next 30 Days)
        future_predictions = []
        current_date = data.index[-1]

        for _ in range(30):
            predicted_price = model.predict(future_input)
            future_predictions.append(predicted_price[0, 0])
            predicted_value = np.array([predicted_price[0, 0], last_60_days_rsi[-1][0], last_60_days_sma_50[-1][0], last_60_days_sma_200[-1][0], last_60_days_pe[-1][0], last_60_days_mfi[-1][0]]).reshape(1, 1, 6)
            future_input = np.append(future_input[:, 1:, :], predicted_value, axis=1)
            current_date += datetime.timedelta(days=1)

        # Reverse scaling
        future_predictions = scaler_price.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # Create Interactive Plotly Graph
        future_dates = pd.date_range(start=data.index[-1] + datetime.timedelta(days=1), periods=len(future_predictions))

        fig = go.Figure()
        # Plot stock prices
        fig.add_trace(go.Scatter(x=data.index[-60:], y=close_data.flatten()[-60:], mode='lines', name='Actual Prices', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=np.append(data.index[-1:], future_dates), y=np.concatenate((close_data[-1:].flatten(), future_predictions.flatten())), mode='lines', name='Future Predictions (Next 30 Days)', line=dict(color='orange', dash='dash')))
        
        # Adjust graph size
        fig.update_layout(width=1310, height=400)
        
        # Enable responsive re-rendering for pan tool
        fig.update_layout(dragmode='pan', uirevision='constant')
        
        # Plot RSI underneath the stock price graph, ensuring corresponding x-axis
        fig_rsi = go.Figure()
        combined_dates = np.append(data.index[-60:], future_dates)
        fig_rsi.add_trace(go.Scatter(x=combined_dates, y=np.concatenate((data['RSI'][-60:], [np.nan] * len(future_dates))), mode='lines', name='RSI', line=dict(color='blue')))
        fig_rsi.add_trace(go.Scatter(x=combined_dates, y=[70] * len(combined_dates), mode='lines', name='Overbought (70)', line=dict(color='grey', dash='dash')))
        fig_rsi.add_trace(go.Scatter(x=combined_dates, y=[30] * len(combined_dates), mode='lines', name='Oversold (30)', line=dict(color='grey', dash='dash')))
        
        fig_rsi.update_layout(
            width=1200,
            height=300,    
            template='plotly_white',
            title=f'{ticker} Stock RSI (Relative Strength Index)',
            dragmode='pan',
            uirevision='constant'          
        )

        # Plot MFI underneath the RSI graph, ensuring corresponding x-axis
        fig_mfi = go.Figure()
        fig_mfi.add_trace(go.Scatter(x=combined_dates, y=np.concatenate((data['MFI'][-60:], [np.nan] * len(future_dates))), mode='lines', name='MFI', line=dict(color='green')))
        fig_mfi.add_trace(go.Scatter(x=combined_dates, y=[80] * len(combined_dates), mode='lines', name='Overbought (80)', line=dict(color='grey', dash='dash')))
        fig_mfi.add_trace(go.Scatter(x=combined_dates, y=[20] * len(combined_dates), mode='lines', name='Oversold (20)', line=dict(color='grey', dash='dash')))
        
        fig_mfi.update_layout(
            width=1200,
            height=300,    
            template='plotly_white',
            title=f'{ticker} Stock MFI (Money Flow Index)',
            dragmode='pan',
            uirevision='constant'          
        )

        fig.update_layout(
            title=f'{ticker} Stock Price Prediction',
            xaxis_title='Date',
            yaxis_title='Stock Price',
            template='plotly_white'
        )

        # Generate HTML for Plotly Graph
        graph_html = pyo.plot(fig, include_plotlyjs=False, output_type='div')
        rsi_html = pyo.plot(fig_rsi, include_plotlyjs=False, output_type='div')
        mfi_html = pyo.plot(fig_mfi, include_plotlyjs=False, output_type='div')

        # Return the graph directly as a response
        return f"<html><head><script src='https://cdn.plot.ly/plotly-latest.min.js'></script></head><body>{graph_html}<br>{rsi_html}<br>{mfi_html}</body></html>"
    except Exception as e:
        logging.error(f"Error during prediction for ticker {ticker}: {str(e)}")
        return f"An error occurred: {str(e)}"

def calculate_mfi(data, window=14):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    money_flow = typical_price * data['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

    positive_flow_sum = positive_flow.rolling(window=window).sum()
    negative_flow_sum = negative_flow.rolling(window=window).sum()

    mfi = 100 - (100 / (1 + (positive_flow_sum / negative_flow_sum)))
    return mfi

def train_model(ticker):
    # Fetch Historical Stock Data
    stock = yf.Ticker(ticker)
    data = stock.history(period='5y')
    if data.empty:
        data = stock.history(period='max')
    rsi_calculator = RSI(period=14)
    data['RSI'] = pd.Series(rsi_calculator.compute_rsi(data['Close'].values), index=data.index[14:])
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    pe_ratio = stock.info.get('trailingPE', np.nan)
    data['PE_Ratio'] = pe_ratio if not np.isnan(pe_ratio) else pd.Series([np.nan] * len(data), index=data.index)
    data['MFI'] = calculate_mfi(data)
    data.dropna(subset=['RSI', 'SMA_50', 'SMA_200', 'MFI'], inplace=True)
    if len(data) < 60:
        logging.error(f"Not enough historical data for ticker: {ticker} to train the model.")
        raise ValueError(f"Not enough historical data available for the ticker: {ticker} to train the model.")
    
    close_data = data['Close'].values.reshape(-1, 1)
    rsi_data = data['RSI'].values.reshape(-1, 1)
    sma_50_data = data['SMA_50'].values.reshape(-1, 1)
    sma_200_data = data['SMA_200'].values.reshape(-1, 1)
    pe_ratio_data = data['PE_Ratio'].values.reshape(-1, 1)
    mfi_data = data['MFI'].values.reshape(-1, 1)

    # Preprocess Data
    scaler_price = MinMaxScaler(feature_range=(0, 1))
    scaler_rsi = MinMaxScaler(feature_range=(0, 1))
    scaler_sma = MinMaxScaler(feature_range=(0, 1))
    scaler_pe = MinMaxScaler(feature_range=(0, 1))
    scaler_mfi = MinMaxScaler(feature_range=(0, 1))
    scaled_close_data = scaler_price.fit_transform(close_data)
    scaled_rsi_data = scaler_rsi.fit_transform(rsi_data)
    scaled_sma_50_data = scaler_sma.fit_transform(sma_50_data)
    scaled_sma_200_data = scaler_sma.fit_transform(sma_200_data)
    scaled_pe_data = scaler_pe.fit_transform(pe_ratio_data)
    scaled_mfi_data = scaler_mfi.fit_transform(mfi_data)

    time_step = 60
    X, y = [], []
    for i in range(len(scaled_close_data) - time_step):
        X.append(np.hstack((scaled_close_data[i:(i + time_step), 0].reshape(-1, 1),
                            scaled_rsi_data[i:(i + time_step), 0].reshape(-1, 1),
                            scaled_sma_50_data[i:(i + time_step), 0].reshape(-1, 1),
                            scaled_sma_200_data[i:(i + time_step), 0].reshape(-1, 1),
                            scaled_pe_data[i:(i + time_step), 0].reshape(-1, 1),
                            scaled_mfi_data[i:(i + time_step), 0].reshape(-1, 1))))
        y.append(scaled_close_data[i + time_step, 0])
    X, y = np.array(X), np.array(y)

    # Split Data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Reshape for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 6)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 6)

    # Build and Train LSTM Model
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(time_step, 6)))
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
    joblib.dump(scaler_mfi, f'models/{ticker}_scaler_mfi.pkl')

    logging.info(f"Model and Scaler for {ticker} saved.")

if __name__ == "__main__":
    app.run(debug=True, port=5001)
