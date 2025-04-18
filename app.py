import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import os

from data_fetcher import fetch_stock_data, fetch_crypto_data
from data_processor import preprocess_data, prepare_data_for_lr, prepare_data_for_random_forest
from models import train_linear_regression, train_random_forest, predict_prices
from visualizer import plot_stock_data, plot_prediction_vs_actual, display_chart_options
from utils import calculate_metrics, get_default_end_date, get_default_start_date
from technical_indicators import add_technical_indicators, display_technical_dashboard
from news_sentiment import display_news_sentiment, get_sentiment_score
from web_scraper import display_scraped_content
from enhanced_predictions import display_advanced_prediction_dashboard

# Set page configuration
st.set_page_config(
    page_title="Financial Market Prediction",
    page_icon="📈",
    layout="wide"
)

# Application title and description
st.title("📈 Financial Market Prediction System")
st.markdown("""
This advanced system uses machine learning models, technical indicators, and news sentiment analysis to predict market behavior
for international stocks, Sri Lankan stocks, and cryptocurrencies.

**Features:**
- Price prediction using machine learning (Linear Regression, Random Forest)
- Technical analysis with multiple indicators (RSI, MACD, Bollinger Bands, etc.)
- News sentiment analysis to gauge market sentiment
- Support for Sri Lankan stocks (CSE: COMB.N0000, JKH.N0000, etc.)
""")

# Sidebar for user inputs
st.sidebar.header("Settings")

# Tab selection
tab_selection = st.sidebar.radio(
    "Select Analysis Mode:",
    ["Price Prediction", "Technical Analysis", "News Sentiment", "Web Scraper", "Advanced AI Prediction", "Combined Analysis"]
)

# Choose between stock or crypto
data_type = st.sidebar.radio("Choose Data Type:", ("Stock", "Cryptocurrency"))

# Input for ticker symbol
if data_type == "Stock":
    # Provide examples of both international and Sri Lankan stocks
    ticker_symbol = st.sidebar.text_input(
        "Enter Stock Symbol:",
        "AAPL",
        help="For international stocks: AAPL, MSFT, GOOG\nFor Sri Lankan stocks: COMB.N0000, JKH.N0000, LOLC.N0000, CCS.N0000"
    )
    
    # Show additional information about Sri Lankan stocks
    if '.N' in ticker_symbol or '.X' in ticker_symbol:
        st.sidebar.info("""
        **Sri Lankan Stock Selected**
        
        Common Sri Lankan Stock Symbols:
        - COMB.N0000 (Commercial Bank)
        - JKH.N0000 (John Keells Holdings)
        - LOLC.N0000 (LOLC Holdings)
        - CCS.N0000 (CIC Holdings)
        - DIST.N0000 (Distilleries Company)
        - TJL.N0000 (Teejay Lanka)
        """)
else:
    ticker_symbol = st.sidebar.text_input("Enter Crypto Symbol (e.g., BTC-USD, ETH-USD):", "BTC-USD")

# Date range selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", get_default_start_date())
with col2:
    end_date = st.date_input("End Date", get_default_end_date())

# Analysis settings based on selected tab
if tab_selection in ["Price Prediction", "Combined Analysis"]:
    st.sidebar.subheader("Prediction Settings")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model:",
        ("Linear Regression", "Random Forest")
    )
    
    # Random Forest parameters if selected
    if model_type == "Random Forest":
        window_size = st.sidebar.slider("Window Size (days):", 1, 15, 5)
        n_estimators = st.sidebar.slider("Number of Trees:", 10, 200, 100)
        prediction_days = st.sidebar.slider("Days to Predict:", 1, 30, 7)
    else:
        prediction_days = st.sidebar.slider("Days to Predict:", 1, 30, 7)
        window_size = 5  # Default for Random Forest
        n_estimators = 100  # Default for Random Forest

# Add news sentiment days slider
if tab_selection in ["News Sentiment", "Combined Analysis"]:
    st.sidebar.subheader("News Settings")
    news_days = st.sidebar.slider("Days for News Analysis:", 1, 30, 7)
    
if tab_selection == "Web Scraper":
    st.sidebar.subheader("Web Scraper Settings")
    url = st.sidebar.text_input("Enter URL to Scrape:", 
                                placeholder="https://example.com",
                                help="Enter the full URL including https://")
    max_chars = st.sidebar.slider("Max Characters to Display:", 500, 10000, 2000)

# Execute button
analysis_button = st.sidebar.button("Run Analysis")

# If button is clicked, execute all the analyses
if analysis_button:
    # Show loading spinner
    with st.spinner(f"Fetching data for {ticker_symbol}..."):
        try:
            # Fetch data based on type
            if data_type == "Stock":
                df = fetch_stock_data(ticker_symbol, start_date, end_date)
            else:
                df = fetch_crypto_data(ticker_symbol, start_date, end_date)

            # Display web scraper if selected (don't need stock data for this)
            if tab_selection == "Web Scraper":
                # Web Scraper Tab
                st.header("Web Content Scraper")
                st.markdown("""
                This tool extracts clean, readable text content from any website URL. 
                It's useful for analyzing news articles, blog posts, and other web content related to financial markets.
                """)
                # Display web scraper functionality
                if 'url' in locals() and url:
                    display_scraped_content(url, max_chars)
                else:
                    st.info("Please enter a URL in the sidebar to scrape web content.")
            
            # For other tabs, process stock data
            elif df is not None and not df.empty:
                # Add technical indicators to the dataframe
                df_with_indicators = add_technical_indicators(df)
                
                # Preprocess data for ML models
                processed_df = preprocess_data(df)
                
                # Display content based on the selected tab
                if tab_selection == "Price Prediction":
                    # Price Prediction Tab
                    st.header("Price Prediction Analysis")
                    
                    # Display historical data
                    st.subheader(f"Historical Data for {ticker_symbol}")
                    st.dataframe(df.head())
                    
                    # Plotting historical data with advanced chart options for all data types
                    st.subheader(f"Chart Options for {ticker_symbol}")
                    display_chart_options(df, ticker_symbol)
                    
                    # Make predictions based on selected model
                    with st.spinner(f"Training {model_type} model..."):
                        if model_type == "Linear Regression":
                            X_train, X_test, y_train, y_test, scaler = prepare_data_for_lr(processed_df)
                            model = train_linear_regression(X_train, y_train)
                        else:  # Random Forest
                            X_train, X_test, y_train, y_test, scaler = prepare_data_for_random_forest(
                                processed_df, window_size=window_size
                            )
                            model = train_random_forest(X_train, y_train, n_estimators=n_estimators)
                        
                        # Make predictions
                        actual_prices, predicted_prices, future_dates, future_predictions = predict_prices(
                            model, X_test, y_test, processed_df, scaler, model_type, 
                            prediction_days
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
                
                elif tab_selection == "Technical Analysis":
                    # Technical Analysis Tab
                    st.header("Technical Analysis")
                    # Display the technical analysis dashboard
                    display_technical_dashboard(df_with_indicators, ticker_symbol)
                
                elif tab_selection == "News Sentiment":
                    # News Sentiment Tab
                    st.header("News Sentiment Analysis")
                    # Display news sentiment
                    display_news_sentiment(ticker_symbol, news_days)
                
                elif tab_selection == "Advanced AI Prediction":
                    # Advanced AI Prediction Tab
                    display_advanced_prediction_dashboard(ticker_symbol, df_with_indicators)
                
# This section is now handled earlier in the code
                
                elif tab_selection == "Combined Analysis":
                    # Combined Analysis Tab - Show everything
                    st.header("Comprehensive Market Analysis")
                    
                    # Create tabs for different analyses
                    price_tab, tech_tab, news_tab = st.tabs(["Price Prediction", "Technical Analysis", "News Sentiment"])
                    
                    with price_tab:
                        # Make predictions based on selected model
                        with st.spinner(f"Training {model_type} model..."):
                            if model_type == "Linear Regression":
                                X_train, X_test, y_train, y_test, scaler = prepare_data_for_lr(processed_df)
                                model = train_linear_regression(X_train, y_train)
                            else:  # Random Forest
                                X_train, X_test, y_train, y_test, scaler = prepare_data_for_random_forest(
                                    processed_df, window_size=window_size
                                )
                                model = train_random_forest(X_train, y_train, n_estimators=n_estimators)
                            
                            # Make predictions
                            actual_prices, predicted_prices, future_dates, future_predictions = predict_prices(
                                model, X_test, y_test, processed_df, scaler, model_type, 
                                prediction_days
                            )
                            
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
                            
                            # Show summary
                            latest_price = df['Close'].iloc[-1]
                            final_prediction = future_predictions[-1]
                            pct_change = ((final_prediction - latest_price) / latest_price) * 100
                            
                            col1, col2 = st.columns(2)
                            col1.metric("Current Price", f"${latest_price:.2f}")
                            col2.metric(f"Predicted Price ({future_dates[-1].strftime('%Y-%m-%d')})", 
                                       f"${final_prediction:.2f}", 
                                       f"{pct_change:.2f}%")
                    
                    with tech_tab:
                        # Display technical analysis
                        display_technical_dashboard(df_with_indicators, ticker_symbol)
                    
                    with news_tab:
                        # Display news sentiment
                        display_news_sentiment(ticker_symbol, news_days)
                    
                    # Final recommendation section that combines all analyses
                    st.header("Integrated Market Recommendation")
                    
                    # Get sentiment score
                    sentiment_score, article_count, _ = get_sentiment_score(ticker_symbol, news_days)
                    
                    # Determine overall recommendation
                    # Price prediction signal
                    price_signal = "NEUTRAL"
                    if pct_change > 5:
                        price_signal = "STRONG BUY"
                    elif pct_change > 1:
                        price_signal = "BUY"
                    elif pct_change < -5:
                        price_signal = "STRONG SELL"
                    elif pct_change < -1:
                        price_signal = "SELL"
                    
                    # Sentiment signal
                    sentiment_signal = "NEUTRAL"
                    if article_count > 0:
                        if sentiment_score > 0.2:
                            sentiment_signal = "STRONG BUY"
                        elif sentiment_score > 0.05:
                            sentiment_signal = "BUY"
                        elif sentiment_score < -0.2:
                            sentiment_signal = "STRONG SELL"
                        elif sentiment_score < -0.05:
                            sentiment_signal = "SELL"
                    
                    # Technical signal from technical_indicators.py
                    from technical_indicators import get_technical_signals
                    signals = get_technical_signals(df_with_indicators)
                    tech_signal = signals.get('OVERALL', {}).get('signal', "NEUTRAL")
                    
                    # Combine signals to get overall recommendation
                    signals_dict = {
                        "STRONG BUY": 2, 
                        "BUY": 1, 
                        "NEUTRAL": 0, 
                        "SELL": -1, 
                        "STRONG SELL": -2
                    }
                    
                    # Convert string signals to numeric values
                    price_value = signals_dict.get(price_signal, 0)
                    sentiment_value = signals_dict.get(sentiment_signal, 0)
                    tech_value = signals_dict.get(tech_signal, 0)
                    
                    # Calculate average signal (give more weight to price prediction)
                    if article_count > 0:
                        avg_signal = (price_value * 0.5) + (tech_value * 0.3) + (sentiment_value * 0.2)
                    else:
                        avg_signal = (price_value * 0.6) + (tech_value * 0.4)
                    
                    # Convert numeric value back to signal
                    final_signal = "NEUTRAL"
                    if avg_signal >= 1.5:
                        final_signal = "STRONG BUY"
                    elif avg_signal >= 0.5:
                        final_signal = "BUY"
                    elif avg_signal <= -1.5:
                        final_signal = "STRONG SELL"
                    elif avg_signal <= -0.5:
                        final_signal = "SELL"
                    
                    # Display the final recommendation
                    signal_color = "gray"
                    if final_signal in ["STRONG BUY", "BUY"]:
                        signal_color = "green"
                    elif final_signal in ["STRONG SELL", "SELL"]:
                        signal_color = "red"
                    
                    st.markdown(f"### Final Recommendation: <span style='color:{signal_color};font-size:24px'>{final_signal}</span>", unsafe_allow_html=True)
                    
                    # Display individual signals
                    st.subheader("Signal Breakdown")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        price_color = "gray"
                        if price_signal in ["STRONG BUY", "BUY"]:
                            price_color = "green"
                        elif price_signal in ["STRONG SELL", "SELL"]:
                            price_color = "red"
                        st.markdown(f"**Price Model:** <span style='color:{price_color}'>{price_signal}</span>", unsafe_allow_html=True)
                        st.markdown(f"Predicted change: {pct_change:.2f}%")
                    
                    with col2:
                        tech_color = "gray"
                        if tech_signal in ["STRONG BUY", "BUY"]:
                            tech_color = "green"
                        elif tech_signal in ["STRONG SELL", "SELL"]:
                            tech_color = "red"
                        st.markdown(f"**Technical Analysis:** <span style='color:{tech_color}'>{tech_signal}</span>", unsafe_allow_html=True)
                    
                    with col3:
                        sentiment_color = "gray"
                        if sentiment_signal in ["STRONG BUY", "BUY"]:
                            sentiment_color = "green"
                        elif sentiment_signal in ["STRONG SELL", "SELL"]:
                            sentiment_color = "red"
                        
                        if article_count > 0:
                            st.markdown(f"**News Sentiment:** <span style='color:{sentiment_color}'>{sentiment_signal}</span>", unsafe_allow_html=True)
                            st.markdown(f"Sentiment score: {sentiment_score:.2f}")
                        else:
                            st.markdown("**News Sentiment:** Not available")
            else:
                st.error(f"No data found for {ticker_symbol}. Please check the symbol and try again.")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your inputs and try again.")

# About section at the bottom
st.markdown("---")
st.markdown("""
### About This System
This financial market prediction system combines multiple approaches for a comprehensive analysis:

#### Price Prediction Models
- **Linear Regression**: A simple model that predicts based on linear relationships in the data
- **Random Forest**: An ensemble learning method that builds multiple decision trees and merges their predictions
- **Enhanced Prediction Model**: Advanced model with multi-factor analysis for higher accuracy predictions

#### Advanced Market Analysis
- **Support & Resistance Detection**: Automatically identifies key price levels
- **Market Sentiment Analysis**: Comprehensive analysis of market sentiment based on multiple factors
- **Risk Assessment**: Identifies potential risks and market drivers

#### Technical Indicators
- **RSI (Relative Strength Index)**: Measures the speed and change of price movements
- **MACD (Moving Average Convergence Divergence)**: Trend-following momentum indicator
- **Bollinger Bands**: Shows price volatility and potential overbought/oversold conditions
- **Moving Averages**: Identify trend direction and potential support/resistance levels
- **Stochastic Oscillator**: Compares a specific closing price to a range of prices over time

#### News Sentiment Analysis
- Analyzes recent news articles related to the asset
- Calculates sentiment scores to determine market perception
- Integrates news sentiment into the overall market recommendation

#### Metrics Explanation
- **MAE**: Mean Absolute Error - Average absolute difference between predictions and actual prices
- **MSE**: Mean Squared Error - Average squared difference between predictions and actual prices
- **RMSE**: Root Mean Squared Error - Square root of MSE, gives error in the same unit as the price
- **MAPE**: Mean Absolute Percentage Error - Average percentage difference between predictions and actual prices

#### Web Content Scraper
- Extracts clean, readable text from financial news sites and research articles
- Helps analyze market information from various sources
- Converts complex HTML to plain text for easier reading

#### Gemini AI Integration
- Future integration with Google's Gemini AI for enhanced market insights
- Set up a GEMINI_API_KEY in Replit Secrets to activate this feature
""")
