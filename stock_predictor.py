# Install required libraries (only needed in Colab/first time)
!pip install yfinance --quiet
!pip install tensorflow --quiet
!pip install scikit-learn --quiet
!pip install plotly --quiet
!pip install seaborn --quiet

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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# Get user inputs for stock
ticker = input("Enter stock ticker (e.g., AAPL, TSLA): ")
start_date = input("Enter start date (YYYY-MM-DD): ")
end_date = input("Enter end date (YYYY-MM-DD): ")

# Download historical stock data from Yahoo Finance
data = yf.download(ticker, start=start_date, end=end_date)

# Keep only required columns
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Feature Engineering
data['Return'] = data['Close'].pct_change()   # Daily returns
data['Volatility'] = data['Return'].rolling(window=5, min_periods=1).std()  # Rolling 5-day volatility
data['MA_10'] = data['Close'].rolling(window=10, min_periods=1).mean()       # 10-day moving average
data['MA_20'] = data['Close'].rolling(window=20, min_periods=1).mean()       # 20-day moving average

# RSI calculation
delta = data['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=14, min_periods=1).mean()
avg_loss = loss.rolling(window=14, min_periods=1).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# Remove infinite or missing values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# Targets: next-day volatility & price
data['Target_Volatility'] = data['Volatility'].shift(-1)
data['Target_Close'] = data['Close'].shift(-1)
data.dropna(inplace=True)

# Prepare features and targets
features = ['Return', 'Volatility', 'MA_10', 'MA_20', 'RSI']
X = data[features].values
y_vol = data['Target_Volatility'].values
y_price = data['Target_Close'].values

# Scale features for LSTM
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (no shuffle to preserve time sequence)
X_train, X_test, y_vol_train, y_vol_test, y_price_train, y_price_test = train_test_split(
    X_scaled, y_vol, y_price, test_size=0.2, shuffle=False
)

# Threshold to choose model
threshold = 100  # small dataset → Random Forest

if len(X) < threshold:
    # Random Forest
    model_vol = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model_vol.fit(X_train, y_vol_train)
    y_vol_pred = model_vol.predict(X_test)

    model_price = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model_price.fit(X_train, y_price_train)
    y_price_pred = model_price.predict(X_test)

else:
    # LSTM
    def create_sequences(X, y, seq_length=5):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        return np.array(X_seq), np.array(y_seq)

    seq_length = 5
    X_train_seq, y_vol_train_seq = create_sequences(X_train, y_vol_train, seq_length)
    _, y_price_train_seq = create_sequences(X_train, y_price_train, seq_length)
    X_test_seq, y_vol_test_seq = create_sequences(X_test, y_vol_test, seq_length)
    _, y_price_test_seq = create_sequences(X_test, y_price_test, seq_length)

    # LSTM for volatility
    model_vol = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        Dropout(0.2),
        Dense(1)
    ])
    model_vol.compile(optimizer='adam', loss='mse')
    es_vol = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history_vol = model_vol.fit(X_train_seq, y_vol_train_seq, epochs=50, batch_size=32,
                                validation_split=0.2, callbacks=[es_vol], verbose=0)
    y_vol_pred = model_vol.predict(X_test_seq).flatten()
    y_vol_test = y_vol_test_seq

    # LSTM for price
    model_price = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        Dropout(0.2),
        Dense(1)
    ])
    model_price.compile(optimizer='adam', loss='mse')
    es_price = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history_price = model_price.fit(X_train_seq, y_price_train_seq, epochs=50, batch_size=32,
                                    validation_split=0.2, callbacks=[es_price], verbose=0)
    y_price_pred = model_price.predict(X_test_seq).flatten()
    y_price_test = y_price_test_seq

# Results DataFrames
results_vol = pd.DataFrame({'Actual_Volatility': y_vol_test, 'Predicted_Volatility': y_vol_pred})
results_price = pd.DataFrame({'Actual_Close': y_price_test, 'Predicted_Close': y_price_pred})

print("Volatility Results (first 5 rows)")
print(results_vol.head())
print("\nPrice Results (first 5 rows)")
print(results_price.head())

# Error Metrics
rmse_vol = np.sqrt(mean_squared_error(y_vol_test, y_vol_pred))
mae_vol = mean_absolute_error(y_vol_test, y_vol_pred)
rmse_price = np.sqrt(mean_squared_error(y_price_test, y_price_pred))
mae_price = mean_absolute_error(y_price_test, y_price_pred)

print(f"\nVolatility → RMSE: {rmse_vol:.5f}, MAE: {mae_vol:.5f}")
print(f"Price → RMSE: {rmse_price:.2f}, MAE: {mae_price:.2f}")

# Interactive Charts (only 4 subplots to match specs)
fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=("Actual vs Predicted Volatility", "Actual vs Predicted Price", 
                    "Volatility Over Time", "RSI"),
    specs=[[{"colspan": 2}, None],
           [{"colspan": 2}, None],
           [{"colspan": 2}, None],
           [{"colspan": 2}, None]]
)

fig.add_trace(go.Scatter(y=y_vol_test, mode='lines+markers', name='Actual Volatility'), row=1, col=1)
fig.add_trace(go.Scatter(y=y_vol_pred, mode='lines+markers', name='Predicted Volatility'), row=1, col=1)

fig.add_trace(go.Scatter(y=y_price_test, mode='lines+markers', name='Actual Price'), row=2, col=1)
fig.add_trace(go.Scatter(y=y_price_pred, mode='lines+markers', name='Predicted Price'), row=2, col=1)

fig.add_trace(go.Scatter(y=data['Volatility'], mode='lines', name='Historical Volatility'), row=3, col=1)
fig.add_trace(go.Scatter(y=data['RSI'], mode='lines', name='RSI'), row=4, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red")
fig.add_hline(y=30, line_dash="dash", line_color="green")

fig.update_layout(height=2200, width=1000, showlegend=True, template='plotly_dark')
fig.show()

# Return distribution histogram
plt.figure(figsize=(8,5))
sns.histplot(data['Return'], kde=True, bins=50)
plt.title('Return Distribution')
plt.show()

# Feature Correlation Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(data[['Return','Volatility','MA_10','MA_20','RSI']].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

# LSTM Training Loss (if LSTM used)
if 'history_vol' in locals():
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(y=history_vol.history['loss'], mode='lines+markers', name='Volatility Training Loss'))
    fig_loss.add_trace(go.Scatter(y=history_vol.history['val_loss'], mode='lines+markers', name='Volatility Validation Loss'))
    fig_loss.update_layout(title='LSTM Volatility Loss During Training', xaxis_title='Epoch', yaxis_title='Loss', template='plotly_dark', width=900, height=500)
    fig_loss.show()

if 'history_price' in locals():
    fig_loss_price = go.Figure()
    fig_loss_price.add_trace(go.Scatter(y=history_price.history['loss'], mode='lines+markers', name='Price Training Loss'))
    fig_loss_price.add_trace(go.Scatter(y=history_price.history['val_loss'], mode='lines+markers', name='Price Validation Loss'))
    fig_loss_price.update_layout(title='LSTM Price Loss During Training', xaxis_title='Epoch', yaxis_title='Loss', template='plotly_dark', width=900, height=500)
    fig_loss_price.show()
