# Install required libraries (only needed in Colab/first time)
!pip install yfinance --quiet
!pip install tensorflow --quiet
!pip install scikit-learn --quiet


import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Get stock ticker and date range from user
ticker = input("Enter stock ticker (e.g., AAPL, TSLA): ")
start_date = input("Enter start date (YYYY-MM-DD): ")
end_date = input("Enter end date (YYYY-MM-DD): ")

# Download historical stock data from Yahoo Finance
data = yf.download(ticker, start=start_date, end=end_date)

# If data has MultiIndex columns, flatten it
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[1] for col in data.columns]

# Standardize column names
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

# Feature engineering
data['Return'] = data['Close'].pct_change()
data['Volatility'] = data['Return'].rolling(window=5, min_periods=1).std()
data['MA_10'] = data['Close'].rolling(window=10, min_periods=1).mean()
data['MA_20'] = data['Close'].rolling(window=20, min_periods=1).mean()

# RSI calculation
delta = data['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=14, min_periods=1).mean()
avg_loss = loss.rolling(window=14, min_periods=1).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# Remove any infinite or missing values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# Prepare features and target variable
X = data[['Return', 'Volatility', 'MA_10', 'MA_20', 'RSI']]
y = data['Close']

# Scale feature values between 0 and 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

threshold = 100

if len(X) < threshold:
    print("🔹 Using Random Forest (small dataset)")
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
else:
    print("🔹 Using LSTM (large dataset)")
    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train_reshaped, y_train_scaled,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[es],
        verbose=0
    )
    y_pred_scaled = model.predict(X_test_reshaped)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

results = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred.flatten()})
print(results.head())

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")

import plotly.graph_objects as go

if 'history' in locals():
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(y=history.history['loss'], mode='lines+markers', name='Training Loss'))
    fig_loss.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines+markers', name='Validation Loss'))
    fig_loss.update_layout(
        title='LSTM Loss During Training',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        template='plotly_dark',
        width=900,
        height=500
    )
    fig_loss.show()

fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(y=y_test.values, mode='lines+markers', name='Actual'))
fig_pred.add_trace(go.Scatter(y=y_pred.flatten(), mode='lines+markers', name='Predicted'))
fig_pred.update_layout(
    title=f"{ticker} Stock Prediction",
    xaxis_title='Time',
    yaxis_title='Price',
    template='plotly_dark',
    width=900,
    height=500
)
fig_pred.show()
