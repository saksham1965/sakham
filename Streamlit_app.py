# app.py
import streamlit as st

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dense, Dropout # type: ignore

st.set_page_config(layout="wide")
st.title("📈 AI Stock Price Predictor & Trading Bot")

# User inputs
ticker = st.text_input("Enter stock symbol")
period = st.selectbox("Select period:", ["1y", "2y", "5y"], index=1)

if st.button("Run Model"):
    # Get stock data
    st.write("Downloading data...")
    data = yf.download(ticker, period=period, interval='1d')['Close'].dropna()
    st.line_chart(data)

    # Scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    # Create sequences
    def create_dataset(data, seq_length=60):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    seq_length = 60
    X, y = create_dataset(scaled_data, seq_length)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    with st.spinner("Training model..."):
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    # Prediction
    predicted_prices = model.predict(X)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    real_prices = scaler.inverse_transform(y.reshape(-1, 1))

    # Plot predictions
    st.subheader("📊 Predicted vs Actual Prices")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(real_prices, label='Actual Price')
    ax1.plot(predicted_prices, label='Predicted Price')
    ax1.legend()
    st.pyplot(fig1)

    # Simulate trades
    initial_balance = 100000
    balance = initial_balance
    stock_holdings = 50
    trade_log = []

    for i in range(1, len(predicted_prices)):
        real_price = real_prices[i][0]
        predicted_price = predicted_prices[i][0]

        if predicted_price > real_price and balance >= real_price:
            num_shares = balance // real_price
            balance -= num_shares * real_price
            stock_holdings += num_shares
            trade_log.append((i, 'BUY', num_shares, real_price))

        elif predicted_price < real_price and stock_holdings > 0:
            balance += stock_holdings * real_price
            trade_log.append((i, 'SELL', stock_holdings, real_price))
            stock_holdings = 0

    final_value = balance + stock_holdings * real_prices[-1][0]
    profit = final_value - initial_balance

    st.subheader("📈 Trading Results")
    st.write(f"Initial balance: ₹{initial_balance:,.2f}")
    st.write(f"Final balance: ₹{final_value:,.2f}")
    st.write(f"Total profit: ₹{profit:,.2f}")
    st.write(f"Total trades: {len(trade_log)}")

    # Plot trading signals
    buy_signals = [i for i, action, _, _ in trade_log if action == 'BUY']
    sell_signals = [i for i, action, _, _ in trade_log if action == 'SELL']

    st.subheader("📍 Buy/Sell Signals")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(real_prices, label='Actual Price', alpha=0.6)
    ax2.scatter(buy_signals, [real_prices[i][0] for i in buy_signals], label='Buy', marker='^', color='green')
    ax2.scatter(sell_signals, [real_prices[i][0] for i in sell_signals], label='Sell', marker='v', color='red')
    ax2.legend()
    st.pyplot(fig2)
