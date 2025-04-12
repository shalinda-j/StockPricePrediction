import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io
from datetime import datetime, timedelta
import streamlit as st

def fetch_stock_data(ticker_symbol, start_date, end_date):
    """
    Fetches historical stock data using yfinance or CSE data for Sri Lankan stocks.
    
    Args:
        ticker_symbol (str): The stock symbol (e.g., 'AAPL', 'MSFT', 'COMB.N0000')
        start_date (datetime): Start date for historical data
        end_date (datetime): End date for historical data
        
    Returns:
        pandas.DataFrame: DataFrame containing historical stock data
    """
    try:
        # Check if it's a Sri Lankan stock (CSE)
        if '.N' in ticker_symbol or '.X' in ticker_symbol:
            return fetch_sri_lankan_stock_data(ticker_symbol, start_date, end_date)
        else:
            # Convert dates to string format
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Fetch data using yfinance for international stocks
            data = yf.download(ticker_symbol, start=start_date_str, end=end_date_str)
            
            if data.empty:
                st.warning(f"No data available for {ticker_symbol} in the selected date range.")
                return None
                
            return data
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None

def fetch_sri_lankan_stock_data(ticker_symbol, start_date, end_date):
    """
    Fetches historical data for Sri Lankan stocks. 
    
    CSE tickers typically follow the format: SYMBOL.N0000 or SYMBOL.X0000
    Examples: COMB.N0000 (Commercial Bank), JKH.N0000 (John Keells Holdings)
    
    Args:
        ticker_symbol (str): The CSE stock symbol (e.g., 'COMB.N0000', 'JKH.N0000')
        start_date (datetime): Start date for historical data
        end_date (datetime): End date for historical data
        
    Returns:
        pandas.DataFrame: DataFrame containing historical stock data
    """
    try:
        # For Sri Lankan stocks, use Yahoo Finance with CSE prefix
        # CSE tickers in Yahoo Finance format are typically prefixed with "CSE:"
        yf_ticker = f"CSE:{ticker_symbol}" if not ticker_symbol.startswith("CSE:") else ticker_symbol
        
        # Convert dates to string format
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Fetch data
        data = yf.download(yf_ticker, start=start_date_str, end=end_date_str)
        
        if data.empty:
            # Try without the CSE prefix as a fallback
            if ticker_symbol.startswith("CSE:"):
                pure_ticker = ticker_symbol[4:]
                data = yf.download(pure_ticker, start=start_date_str, end=end_date_str)
            else:
                # Try with just the base symbol (before the .N or .X)
                base_symbol = ticker_symbol.split('.')[0]
                data = yf.download(base_symbol, start=start_date_str, end=end_date_str)
        
        if data.empty:
            st.warning(f"No data available for {ticker_symbol} in the selected date range. "
                       f"Please ensure the symbol is correct for Sri Lankan stocks.")
            return None
            
        return data
    except Exception as e:
        st.error(f"Error fetching Sri Lankan stock data: {str(e)}")
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
