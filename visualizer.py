import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def plot_stock_data(df, ticker_symbol):
    """
    Plot historical stock/crypto data.
    
    Args:
        df (pandas.DataFrame): DataFrame with historical price data
        ticker_symbol (str): Symbol of the stock/crypto
        
    Returns:
        matplotlib.figure.Figure: Figure object with the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    dates = df.index
    closing_prices = df['Close']
    
    # Plot closing prices
    ax.plot(dates, closing_prices, label='Close Price', color='blue')
    
    # If available, plot high and low prices
    if 'High' in df.columns and 'Low' in df.columns:
        ax.plot(dates, df['High'], label='High Price', color='green', alpha=0.5, linestyle='--')
        ax.plot(dates, df['Low'], label='Low Price', color='red', alpha=0.5, linestyle='--')
    
    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.set_title(f'Historical Data for {ticker_symbol}')
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    # Format date on x-axis
    fig.autofmt_xdate()
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def plot_prediction_vs_actual(actual_prices, predicted_prices, future_dates, future_predictions, ticker_symbol):
    """
    Plot actual vs predicted prices and future predictions.
    
    Args:
        actual_prices (numpy.ndarray): Array of actual prices
        predicted_prices (numpy.ndarray): Array of predicted prices
        future_dates (list): List of future dates
        future_predictions (numpy.ndarray): Array of predicted future prices
        ticker_symbol (str): Symbol of the stock/crypto
        
    Returns:
        matplotlib.figure.Figure: Figure object with the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create date range for the historical data
    num_actual = len(actual_prices)
    historical_dates = [future_dates[0] - timedelta(days=num_actual-i) for i in range(num_actual)]
    
    # Plot actual prices
    ax.plot(historical_dates, actual_prices, label='Actual Prices', color='blue')
    
    # Plot predicted prices
    ax.plot(historical_dates, predicted_prices, label='Predicted Prices', color='red', linestyle='--')
    
    # Plot future predictions
    ax.plot(future_dates, future_predictions, label='Future Predictions', color='green', marker='o')
    
    # Highlight the separation between historical and future data
    ax.axvline(x=future_dates[0], color='gray', linestyle='-', alpha=0.7)
    ax.text(future_dates[0], min(actual_prices) - 5, 'Prediction Start', 
            rotation=90, verticalalignment='top')
    
    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.set_title(f'Price Prediction for {ticker_symbol}')
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    # Format date on x-axis
    fig.autofmt_xdate()
    
    # Tight layout
    plt.tight_layout()
    
    return fig
