import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import mplfinance as mpf

def plot_stock_data(df, ticker_symbol, chart_type='line'):
    """
    Plot historical stock/crypto data with different chart types.
    
    Args:
        df (pandas.DataFrame): DataFrame with historical price data
        ticker_symbol (str): Symbol of the stock/crypto
        chart_type (str): Type of chart to display ('line', 'candlestick', 'bar', 'area', 
                          'heikinashi', 'renko', 'pnf', 'kagi', 'volume')
        
    Returns:
        matplotlib.figure.Figure: Figure object with the plot
    """
    # Make a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Ensure df index is datetime
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        df_copy.index = pd.to_datetime(df_copy.index)
    
    # Ensure all OHLC columns are numeric (float)
    required_columns = ['Open', 'High', 'Low', 'Close']
    for col in required_columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    
    # Check for NaN values and drop them
    df_copy = df_copy.dropna(subset=required_columns, how='any')
    
    # Check required columns
    has_ohlc = all(col in df_copy.columns for col in required_columns)
    has_volume = 'Volume' in df_copy.columns
    
    # If Volume exists, make sure it's numeric too
    if has_volume:
        df_copy['Volume'] = pd.to_numeric(df_copy['Volume'], errors='coerce')
        # Replace NaN values in Volume with 0
        df_copy['Volume'] = df_copy['Volume'].fillna(0)
    
    # Create default style for mplfinance
    market_colors = mpf.make_marketcolors(
        up='green', down='red',
        edge='black',
        wick={'up': 'green', 'down': 'red'},
        volume={'up': 'green', 'down': 'red'}
    )
    mpf_style = mpf.make_mpf_style(marketcolors=market_colors, gridstyle='--')
    
    # For non-mpf chart types, use matplotlib
    if chart_type == 'line':
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot closing prices
        ax.plot(df_copy.index, df_copy['Close'], label='Close Price', color='blue')
        
        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.set_title(f'Line Chart for {ticker_symbol}')
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        
        # Format date on x-axis
        fig.autofmt_xdate()
        plt.tight_layout()
        
        return fig
    
    elif chart_type == 'area':
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot area chart
        ax.fill_between(df_copy.index, df_copy['Close'], alpha=0.3, color='blue')
        ax.plot(df_copy.index, df_copy['Close'], color='blue')
        
        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.set_title(f'Area Chart for {ticker_symbol}')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Format date on x-axis
        fig.autofmt_xdate()
        plt.tight_layout()
        
        return fig
    
    elif chart_type == 'volume':
        # Create figure with two subplots: price and volume
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price on top subplot
        ax1.plot(df_copy.index, df_copy['Close'], color='blue')
        ax1.set_ylabel('Price ($)')
        ax1.set_title(f'Price and Volume for {ticker_symbol}')
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Plot volume on bottom subplot
        if has_volume:
            # Create colors for volume bars (green if price went up, red if down)
            # Handle potential errors with price comparison
            try:
                colors = ['green' if df_copy['Close'].iloc[i] > df_copy['Close'].iloc[i-1] else 'red' 
                         for i in range(1, len(df_copy))]
                colors.insert(0, 'gray')  # For the first bar
                ax2.bar(df_copy.index, df_copy['Volume'], color=colors, alpha=0.8)
            except:
                # Fallback if there's an issue with colors
                ax2.bar(df_copy.index, df_copy['Volume'], color='blue', alpha=0.8)
            
            ax2.set_ylabel('Volume')
        else:
            ax2.text(0.5, 0.5, 'Volume data not available', 
                    horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        
        ax2.set_xlabel('Date')
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # Format date on x-axis
        fig.autofmt_xdate()
        plt.tight_layout()
        
        return fig
    
    # For mplfinance charts
    else:
        if not has_ohlc:
            # Fallback to line chart if OHLC data is not available
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df_copy.index, df_copy['Close'], label='Close Price', color='blue')
            ax.set_title(f'Line Chart for {ticker_symbol} (OHLC data not available for {chart_type})')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            fig.autofmt_xdate()
            plt.tight_layout()
            return fig
        
        try:
            # Set up kwargs for mplfinance
            kwargs = {
                'type': 'candle',  # Default, will be overridden for specific chart types
                'style': mpf_style,
                'title': f'{chart_type.capitalize()} Chart for {ticker_symbol}',
                'figsize': (10, 6),
                'returnfig': True,
            }
            
            # Add volume if available
            if has_volume:
                kwargs['volume'] = True
                kwargs['volume_panel'] = 1
                kwargs['panel_ratios'] = (4, 1)
            
            # Set specific chart types
            if chart_type == 'candlestick':
                kwargs['type'] = 'candle'
            elif chart_type == 'bar':
                kwargs['type'] = 'ohlc'
            elif chart_type == 'heikinashi':
                kwargs['type'] = 'candle'
                kwargs['heikinashi'] = True
            elif chart_type == 'renko':
                try:
                    # Calculate appropriate brick size (about 2% of price range)
                    price_range = df_copy['High'].max() - df_copy['Low'].min()
                    brick_size = round(price_range * 0.02, 2)
                    # Ensure brick size is not too small
                    min_brick_size = price_range * 0.005
                    brick_size = max(brick_size, min_brick_size)
                    
                    kwargs['type'] = 'renko'
                    kwargs['renko_params'] = {'brick_size': brick_size}
                except Exception as e:
                    # Fallback to candlestick if renko fails
                    st.warning(f"Could not create renko chart: {str(e)}. Falling back to candlestick.")
                    kwargs['type'] = 'candle'
            elif chart_type == 'pnf':  # Point and Figure
                try:
                    # Calculate box size (about 1% of average price)
                    avg_price = df_copy['Close'].mean()
                    box_size = round(avg_price * 0.01, 2)
                    # Ensure box size is reasonable
                    min_box_size = avg_price * 0.005
                    box_size = max(box_size, min_box_size)
                    
                    kwargs['type'] = 'pnf'
                    kwargs['pnf_params'] = {'box_size': box_size, 'reversal': 3}
                except Exception as e:
                    # Fallback to candlestick if PnF fails
                    st.warning(f"Could not create Point and Figure chart: {str(e)}. Falling back to candlestick.")
                    kwargs['type'] = 'candle'
            elif chart_type == 'kagi':
                try:
                    # Calculate reversal size (about 3% of average price)
                    avg_price = df_copy['Close'].mean()
                    reversal = round(avg_price * 0.03, 2)
                    # Ensure reversal is reasonable
                    min_reversal = avg_price * 0.01
                    reversal = max(reversal, min_reversal)
                    
                    kwargs['type'] = 'kagi'
                    kwargs['kagi_params'] = {'reversal': reversal}
                except Exception as e:
                    # Fallback to candlestick if Kagi fails
                    st.warning(f"Could not create Kagi chart: {str(e)}. Falling back to candlestick.")
                    kwargs['type'] = 'candle'
            
            # Check if the dataframe has enough data points
            if len(df_copy) < 10:
                st.warning(f"Not enough data points to create {chart_type} chart. Falling back to line chart.")
                # Create a simple line chart as fallback
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df_copy.index, df_copy['Close'], label='Close Price', color='blue')
                ax.set_title(f'Line Chart for {ticker_symbol} (insufficient data for {chart_type})')
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend()
                fig.autofmt_xdate()
                plt.tight_layout()
                return fig
            
            # Create the figure
            fig, axes = mpf.plot(df_copy, **kwargs)
            
        except Exception as e:
            # If there's an error with mplfinance, fall back to a simple line chart
            st.error(f"Error plotting {chart_type} chart: {str(e)}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df_copy.index, df_copy['Close'], label='Close Price', color='blue')
            ax.set_title(f'Line Chart for {ticker_symbol} (fallback due to error)')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            fig.autofmt_xdate()
            plt.tight_layout()
            return fig
        
        return fig[0]  # Return the Figure object, not the tuple

def display_chart_options(df, ticker_symbol):
    """
    Display a selection of chart types with Streamlit and show the selected chart
    
    Args:
        df (pandas.DataFrame): DataFrame with historical price data
        ticker_symbol (str): Symbol of the stock/crypto
    """
    # Define available chart types
    chart_types = {
        'Candlestick': 'candlestick',
        'Line': 'line',
        'Bar': 'bar',
        'Heikin Ashi': 'heikinashi',
        'Renko': 'renko',
        'Point and Figure': 'pnf',
        'Kagi': 'kagi',
        'Volume': 'volume',
        'Area': 'area'
    }
    
    # Display chart type selector
    st.subheader("Chart Type")
    chart_type_name = st.selectbox(
        "Select chart type:",
        options=list(chart_types.keys()),
        index=0,
        key=f"chart_type_{ticker_symbol}"
    )
    
    # Get the selected chart type
    chart_type = chart_types[chart_type_name]
    
    # Display explanation for the selected chart type
    chart_explanations = {
        'candlestick': "Shows price movements with open, high, low, and close prices for a specific period.",
        'line': "Displays closing prices over time, simple for trend analysis.",
        'bar': "Similar to candlestick but uses vertical bars to show high/low and horizontal lines for open/close.",
        'heikinashi': "Modified candlestick chart that smooths price action to highlight trends.",
        'renko': "Focuses on price movement, plotting bricks when price moves a set amount, ignoring time.",
        'pnf': "Plots price changes in columns of X's (rising) and O's (falling), ignoring time.",
        'kagi': "Uses lines that change thickness based on price reversals, emphasizing significant moves.",
        'volume': "Shows trading volume alongside price, often as bars below the main chart.",
        'area': "Similar to a line chart but filled to emphasize price trends visually."
    }
    
    st.info(chart_explanations.get(chart_type, "Displays price data over time."))
    
    # Show the chart
    with st.spinner(f"Generating {chart_type_name} chart..."):
        fig = plot_stock_data(df, ticker_symbol, chart_type)
        st.pyplot(fig)

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
