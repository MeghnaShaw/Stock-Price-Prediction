import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import sys
if sys.prefix == sys.base_prefix:
    print("Warning: Not using a virtual environment. Consider using one.")

@st.cache_data
def get_sp500_tickers():
    """Fetch the list of S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_df = tables[0]
    return sp500_df["Symbol"].tolist()

def fetch_stock_data(stock_symbol, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance."""
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data

def prepare_data(data):
    """Scale stock price data for LSTM training."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    return scaled_data, scaler

def create_sequences(data, time_step=60):
    """Create sequences for LSTM training."""
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

def build_lstm_model():
    """Build the LSTM model for stock price prediction."""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future_price(model, last_60_days, scaler):
    """Predict the future stock price using LSTM."""
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test = np.array([last_60_days_scaled])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_price = model.predict(X_test)
    return scaler.inverse_transform(predicted_price)[0][0]  # Ensure scalar output

def main():
    st.title("📈 Stock Price Prediction")

    # Get available stock tickers
    available_stocks = get_sp500_tickers()

    # Dropdown for stock selection
    stock_symbol = st.selectbox("Select Stock Ticker:", available_stocks)

    if st.button("Predict Future Price"):
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=365 * 2)).strftime('%Y-%m-%d')
        
        data = fetch_stock_data(stock_symbol, start_date, end_date)

        if data.empty:
            st.error("No data found for the selected stock. Please try another one.")
            return
        
        scaled_data, scaler = prepare_data(data)
        X, Y = create_sequences(scaled_data)

        if len(X) == 0:
            st.error("Not enough data for training.")
            return

        X_train, Y_train = X[:int(len(X) * 0.8)], Y[:int(len(Y) * 0.8)]

        model = build_lstm_model()
        model.fit(X_train.reshape(-1, 60, 1), Y_train, epochs=5, batch_size=32, verbose=1)

        last_60_days = data[['Close']].values[-60:]
        predicted_price = predict_future_price(model, last_60_days, scaler)

                # Extract scalar value for comparison
        last_close_price = data['Close'].iloc[-1]  # Last known price

        # ✅ FIXED: Convert both to float scalars before comparison
        trend = "📈 Increase" if float(predicted_price) > float(last_close_price) else "📉 Decrease"
        trend_color = "green" if float(predicted_price) > float(last_close_price) else "red"

        # Display predicted price with trend color
        st.markdown(f"<h3 style='color: {trend_color};'>Predicted Future Price: ${predicted_price:.2f} ({trend})</h3>", unsafe_allow_html=True)

        # Plot historical prices
        st.subheader(f"{stock_symbol} Stock Price History")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data.index, data['Close'], label="Actual Price", color='blue')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
