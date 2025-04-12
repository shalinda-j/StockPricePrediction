import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import os

from data_fetcher import fetch_stock_data, fetch_crypto_data
from data_processor import preprocess_data, prepare_data_for_lstm, prepare_data_for_lr
from models import train_linear_regression, train_lstm_model, predict_prices
from visualizer import plot_stock_data, plot_prediction_vs_actual
from utils import calculate_metrics, get_default_end_date, get_default_start_date

# Set page configuration
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# Application title and description
st.title("📈 Stock Price Prediction App")
st.markdown("""
This application uses machine learning models to predict stock and cryptocurrency prices.
Choose a stock/crypto, date range, and model to see predictions!
""")

# Sidebar for user inputs
st.sidebar.header("Settings")

# Choose between stock or crypto
data_type = st.sidebar.radio("Choose Data Type:", ("Stock", "Cryptocurrency"))

# Input for ticker symbol
if data_type == "Stock":
    ticker_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, GOOG):", "AAPL")
else:
    ticker_symbol = st.sidebar.text_input("Enter Crypto Symbol (e.g., BTC-USD, ETH-USD):", "BTC-USD")

# Date range selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", get_default_start_date())
with col2:
    end_date = st.date_input("End Date", get_default_end_date())

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model:",
    ("Linear Regression", "LSTM (Long Short-Term Memory)")
)

# LSTM parameters if LSTM is selected
if model_type == "LSTM (Long Short-Term Memory)":
    sequence_length = st.sidebar.slider("Sequence Length (days):", 5, 60, 30)
    epochs = st.sidebar.slider("Training Epochs:", 10, 100, 50)
    prediction_days = st.sidebar.slider("Days to Predict:", 1, 30, 7)
else:
    prediction_days = st.sidebar.slider("Days to Predict:", 1, 30, 7)
    sequence_length = 0  # Not used for Linear Regression
    epochs = 0  # Not used for Linear Regression

# Button to start prediction
predict_button = st.sidebar.button("Predict Prices")

# Main content area
if predict_button:
    # Show loading spinner
    with st.spinner(f"Fetching data for {ticker_symbol}..."):
        try:
            # Fetch data based on type
            if data_type == "Stock":
                df = fetch_stock_data(ticker_symbol, start_date, end_date)
            else:
                df = fetch_crypto_data(ticker_symbol, start_date, end_date)

            if df is not None and not df.empty:
                # Display raw data
                st.subheader(f"Historical Data for {ticker_symbol}")
                st.dataframe(df.head())
                
                # Plotting historical data
                fig = plot_stock_data(df, ticker_symbol)
                st.pyplot(fig)
                
                # Preprocess data
                processed_df = preprocess_data(df)
                
                # Make predictions based on selected model
                with st.spinner(f"Training {model_type} model..."):
                    if model_type == "Linear Regression":
                        X_train, X_test, y_train, y_test, scaler = prepare_data_for_lr(processed_df)
                        model = train_linear_regression(X_train, y_train)
                    else:  # LSTM
                        X_train, X_test, y_train, y_test, scaler, last_sequence = prepare_data_for_lstm(processed_df, sequence_length)
                        model = train_lstm_model(X_train, y_train, sequence_length, epochs)
                    
                    # Make predictions
                    actual_prices, predicted_prices, future_dates, future_predictions = predict_prices(
                        model, X_test, y_test, processed_df, scaler, model_type, 
                        prediction_days, sequence_length, last_sequence if model_type == "LSTM (Long Short-Term Memory)" else None
                    )
                    
                    # Calculate metrics
                    mae, mse, rmse, mape = calculate_metrics(actual_prices, predicted_prices)
                    
                    # Display metrics
                    st.subheader("Model Performance Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("MAE", f"{mae:.2f}")
                    col2.metric("MSE", f"{mse:.2f}")
                    col3.metric("RMSE", f"{rmse:.2f}")
                    col4.metric("MAPE", f"{mape:.2f}%")
                    
                    # Plot prediction vs actual
                    fig = plot_prediction_vs_actual(actual_prices, predicted_prices, future_dates, future_predictions, ticker_symbol)
                    st.pyplot(fig)
                    
                    # Display future predictions in a table
                    st.subheader(f"Price Predictions for Next {prediction_days} Days")
                    future_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Price': future_predictions
                    })
                    st.dataframe(future_df)
                    
                    # Show a final summary
                    latest_price = df['Close'].iloc[-1]
                    final_prediction = future_predictions[-1]
                    pct_change = ((final_prediction - latest_price) / latest_price) * 100
                    
                    st.subheader("Summary")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Current Price", f"${latest_price:.2f}")
                    col2.metric(f"Predicted Price ({future_dates[-1].strftime('%Y-%m-%d')})", 
                               f"${final_prediction:.2f}", 
                               f"{pct_change:.2f}%")
                    col3.metric("Model Used", model_type)
                
            else:
                st.error(f"No data found for {ticker_symbol}. Please check the symbol and try again.")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your inputs and try again.")

# Add additional information
st.markdown("---")
st.markdown("""
### About This App
This app uses historical stock/cryptocurrency data to predict future prices using machine learning models:
- **Linear Regression**: A simple model that predicts based on linear relationships in the data
- **LSTM**: A deep learning model designed to remember patterns in sequential data

### Metrics Explanation
- **MAE**: Mean Absolute Error - Average absolute difference between predictions and actual prices
- **MSE**: Mean Squared Error - Average squared difference between predictions and actual prices
- **RMSE**: Root Mean Squared Error - Square root of MSE, gives error in the same unit as the price
- **MAPE**: Mean Absolute Percentage Error - Average percentage difference between predictions and actual prices
""")
