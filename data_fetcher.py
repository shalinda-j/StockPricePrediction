import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import streamlit as st

def fetch_stock_data(ticker_symbol, start_date, end_date):
    """
    Fetches historical stock data using yfinance.
    
    Args:
        ticker_symbol (str): The stock symbol (e.g., 'AAPL', 'MSFT')
        start_date (datetime): Start date for historical data
        end_date (datetime): End date for historical data
        
    Returns:
        pandas.DataFrame: DataFrame containing historical stock data
    """
    try:
        # Convert dates to string format
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Fetch data
        data = yf.download(ticker_symbol, start=start_date_str, end=end_date_str)
        
        if data.empty:
            st.warning(f"No data available for {ticker_symbol} in the selected date range.")
            return None
            
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None

def fetch_crypto_data(ticker_symbol, start_date, end_date):
    """
    Fetches historical cryptocurrency data using yfinance.
    
    Args:
        ticker_symbol (str): The crypto symbol (e.g., 'BTC-USD', 'ETH-USD')
        start_date (datetime): Start date for historical data
        end_date (datetime): End date for historical data
        
    Returns:
        pandas.DataFrame: DataFrame containing historical crypto data
    """
    try:
        # Convert dates to string format
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Fetch data using yfinance
        data = yf.download(ticker_symbol, start=start_date_str, end=end_date_str)
        
        if data.empty:
            st.warning(f"No data available for {ticker_symbol} in the selected date range.")
            return None
            
        return data
    except Exception as e:
        st.error(f"Error fetching cryptocurrency data: {str(e)}")
        return None
