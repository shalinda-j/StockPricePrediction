import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import json
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

class EnhancedPredictionModel:
    """
    Enhanced prediction model using Random Forest with additional features
    """
    def __init__(self, sequence_length=60):
        """
        Initialize the model
        
        Args:
            sequence_length (int): Number of days to look back
        """
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_count = None
        
    def prepare_data(self, df, target_col='Close', test_size=0.2):
        """
        Prepare data for the model
        
        Args:
            df (pd.DataFrame): DataFrame with features
            target_col (str): Target column to predict
            test_size (float): Proportion of data for testing
            
        Returns:
            tuple: X_train, X_test, y_train, y_test, last_sequence
        """
        # Select features
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Add technical indicators if available
        for col in df.columns:
            # Check if column is a string type before using string methods
            if isinstance(col, str) and (col.startswith('RSI') or col.startswith('MACD') or col.startswith('BB_')):
                features.append(col)
        
        # Ensure all features exist in the DataFrame
        features = [f for f in features if f in df.columns]
        data = df[features].copy()
        
        # Store feature count
        self.feature_count = len(features)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:i + self.sequence_length].flatten())
            y.append(scaled_data[i + self.sequence_length, data.columns.get_loc(target_col)])
        
        X, y = np.array(X), np.array(y)
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Store the last sequence for future predictions
        last_sequence = scaled_data[-self.sequence_length:]
        
        return X_train, X_test, y_train, y_test, last_sequence
        
    def train(self, X_train, y_train, n_estimators=100):
        """
        Train the model
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
            n_estimators (int): Number of trees in the forest
            
        Returns:
            RandomForestRegressor: Trained model
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        return self.model
    
    def predict(self, X_test):
        """
        Make predictions
        
        Args:
            X_test (np.array): Test features
            
        Returns:
            np.array: Predicted values (inverse scaled)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
            
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Create dummy array for inverse transform
        dummy = np.zeros((len(predictions), self.feature_count))
        dummy[:, 3] = predictions  # Assuming 'Close' is at index 3
        
        # Inverse transform
        predictions_inverse = self.scaler.inverse_transform(dummy)[:, 3]
        
        return predictions_inverse
    
    def forecast_future(self, last_sequence, days=7):
        """
        Forecast future prices
        
        Args:
            last_sequence (np.array): Last sequence from the dataset
            days (int): Number of days to forecast
            
        Returns:
            tuple: (future_dates, future_predictions)
        """
        if self.model is None:
            raise ValueError("Model must be trained before forecasting")
            
        # Start with the last sequence
        curr_sequence = last_sequence.copy()
        future_predictions = []
        
        # Generate predictions for future days
        for _ in range(days):
            # Flatten the sequence for prediction
            sequence_flat = curr_sequence.flatten().reshape(1, -1)
            
            # Predict the next value
            next_pred = self.model.predict(sequence_flat)[0]
            
            # Create dummy row for inverse transform
            dummy_row = np.zeros((1, self.feature_count))
            dummy_row[0, 3] = next_pred  # Assuming 'Close' is at index 3
            
            # Inverse transform
            next_price = self.scaler.inverse_transform(dummy_row)[0, 3]
            future_predictions.append(next_price)
            
            # Update sequence: remove first row, add new prediction to the end
            new_row = np.zeros((1, self.feature_count))
            new_row[0, 3] = next_pred  # Set prediction in Close position
            
            # Use the mean of the other features from the last 5 entries
            for i in range(self.feature_count):
                if i != 3:  # Skip Close which we already set
                    new_row[0, i] = np.mean(curr_sequence[-5:, i])
            
            curr_sequence = np.vstack((curr_sequence[1:], new_row))
        
        # Generate future dates
        last_date = datetime.now()
        future_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(days)]
        
        return future_dates, np.array(future_predictions)
    
    def evaluate(self, y_test, y_pred):
        """
        Evaluate model performance
        
        Args:
            y_test (np.array): Actual values
            y_pred (np.array): Predicted values
            
        Returns:
            dict: Performance metrics
        """
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape
        }
    
    def plot_training_history(self, y_test, y_pred):
        """
        Plot actual vs predicted values
        
        Args:
            y_test (np.array): Actual values
            y_pred (np.array): Predicted values
            
        Returns:
            matplotlib.figure.Figure: Plot figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot actual vs predicted
        ax.plot(y_test, label='Actual', color='blue')
        ax.plot(y_pred, label='Predicted', color='red', linestyle='--')
        
        ax.set_title('Model Performance: Actual vs Predicted')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_future_predictions(self, dates, predictions, last_price, title="Future Price Predictions"):
        """
        Plot future predictions
        
        Args:
            dates (list): Future dates
            predictions (np.array): Predicted prices
            last_price (float): Last actual price
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Plot figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot last price and predictions
        ax.plot([-1], [last_price], 'bo', label='Last Actual Price')
        ax.plot(range(len(predictions)), predictions, 'r-o', label='Predicted Prices')
        
        # Set x-axis labels
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels(dates, rotation=45)
        
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig

def get_gemini_api_key():
    """
    Get the Gemini API key from secrets
    """
    try:
        return st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        return None

def display_prediction_model(ticker_symbol, df_with_indicators):
    """
    Display the enhanced prediction model.
    
    Args:
        ticker_symbol (str): Stock symbol
        df_with_indicators (pd.DataFrame): DataFrame with technical indicators
    """
    st.subheader("Enhanced Prediction Model Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        sequence_length = st.slider("Sequence Length (days to look back):", 10, 100, 60)
        days_to_predict = st.slider("Days to Predict:", 1, 30, 7)
    with col2:
        n_estimators = st.slider("Number of Trees:", 50, 500, 200)
        test_size = st.slider("Test Size (%):", 10, 40, 20) / 100
    
    # Automatically start model training
    with st.spinner("Preparing data for enhanced prediction model..."):
            try:
                # Prepare data
                model = EnhancedPredictionModel(sequence_length=sequence_length)
                
                if len(df_with_indicators) < sequence_length + 10:
                    st.error(f"Not enough data points. Need at least {sequence_length + 10} valid data points. Please increase the date range.")
                    return
                
                # Train the model
                with st.spinner(f"Training enhanced model with {len(df_with_indicators)} data points..."):
                    X_train, X_test, y_train, y_test, last_sequence = model.prepare_data(
                        df_with_indicators,
                        test_size=test_size
                    )
                    
                    model.train(X_train, y_train, n_estimators=n_estimators)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Evaluate the model
                    metrics = model.evaluate(y_test, y_pred)
                    
                    # Forecast future
                    future_dates, future_predictions = model.forecast_future(last_sequence, days=days_to_predict)
                
                # Display the metrics
                st.subheader("Model Performance")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MAE", f"${metrics['mae']:.2f}")
                col2.metric("MSE", f"${metrics['mse']:.2f}")
                col3.metric("RMSE", f"${metrics['rmse']:.2f}")
                col4.metric("MAPE", f"{metrics['mape']:.2f}%")
                
                # Plot actual vs predicted
                st.subheader("Model Validation")
                fig = model.plot_training_history(y_test, y_pred)
                st.pyplot(fig)
                
                # Plot future predictions
                st.subheader(f"Future Price Predictions for {ticker_symbol}")
                last_price = df_with_indicators['Close'].iloc[-1]
                fig = model.plot_future_predictions(
                    future_dates,
                    future_predictions,
                    last_price,
                    title=f"Future Price Predictions for {ticker_symbol}"
                )
                st.pyplot(fig)
                
                # Display predictions in a table
                st.subheader(f"Predicted Prices for Next {days_to_predict} Days")
                
                # Create DataFrame for display
                # Ensure all values are scalar, not Series
                dates_list = []
                for d in future_dates:
                    if isinstance(d, str):
                        dates_list.append(d)
                    elif hasattr(d, 'item'):
                        dates_list.append(d.item())
                    elif hasattr(d, 'iloc') and hasattr(d, '__len__') and len(d) > 0:
                        dates_list.append(str(d.iloc[0]))
                    else:
                        dates_list.append(str(d))
                # Ensure all prediction values are scalar numbers
                price_list = []
                for p in future_predictions:
                    if isinstance(p, (int, float)):
                        price_list.append(float(p))
                    elif hasattr(p, 'item'):
                        price_list.append(float(p.item()))
                    elif hasattr(p, 'iloc') and hasattr(p, '__len__') and len(p) > 0:
                        price_list.append(float(p.iloc[0]))
                    else:
                        try:
                            price_list.append(float(p))
                        except (ValueError, TypeError):
                            # If conversion fails, use a default value
                            price_list.append(0.0)
                change_list = [float((price - last_price) / last_price * 100) for price in price_list]
                
                pred_df = pd.DataFrame({
                    'Date': dates_list,
                    'Predicted Price': price_list,
                    'Change (%)': change_list
                })
                
                st.dataframe(pred_df)
                
                # Calculate overall prediction signal
                final_prediction = future_predictions[-1]
                
                # Convert final_prediction to scalar if it's a Series
                if hasattr(final_prediction, 'item'):
                    final_prediction = final_prediction.item()
                elif hasattr(final_prediction, 'iloc') and hasattr(final_prediction, '__len__') and len(final_prediction) > 0:
                    final_prediction = final_prediction.iloc[0]
                    
                pct_change = ((final_prediction - last_price) / last_price) * 100
                
                # Convert to float if it's a Series
                if hasattr(pct_change, 'item'):
                    pct_change = pct_change.item()
                elif hasattr(pct_change, 'iloc') and hasattr(pct_change, '__len__') and len(pct_change) > 0:
                    pct_change = pct_change.iloc[0]
                
                prediction_signal = "NEUTRAL"
                if pct_change > 5:
                    prediction_signal = "STRONG BUY"
                elif pct_change > 1:
                    prediction_signal = "BUY"
                elif pct_change < -5:
                    prediction_signal = "STRONG SELL"
                elif pct_change < -1:
                    prediction_signal = "SELL"
                
                # Display summary
                signal_color = "gray"
                if prediction_signal in ["STRONG BUY", "BUY"]:
                    signal_color = "green"
                elif prediction_signal in ["STRONG SELL", "SELL"]:
                    signal_color = "red"
                
                st.subheader("Prediction Summary")
                col1, col2, col3 = st.columns(3)
                # Ensure future_dates[-1] is a string, not a Series
                last_date = future_dates[-1]
                if isinstance(last_date, str):
                    pass  # Already a string
                elif hasattr(last_date, 'item'):
                    last_date = last_date.item()
                elif hasattr(last_date, 'iloc') and hasattr(last_date, '__len__') and len(last_date) > 0:
                    last_date = str(last_date.iloc[0])
                else:
                    last_date = str(last_date)
                    
                col1.metric("Current Price", f"${last_price:.2f}")
                col2.metric(f"Predicted Price ({last_date})", 
                           f"${final_prediction:.2f}", 
                           f"{pct_change:.2f}%")
                col3.markdown(f"**Signal:** <span style='color:{signal_color}'>{prediction_signal}</span>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error in enhanced prediction model: {str(e)}")
                import traceback
                st.text(traceback.format_exc())

class GeminiMarketAnalyzer:
    """
    Simple implementation of market analysis using contextual data
    """
    def __init__(self):
        """Initialize the analyzer"""
        self.initialized = False
        
    def analyze_market_trends(self, ticker_symbol, df_with_indicators, news_sentiment=None):
        """
        Analyze market trends based on available data
        
        Args:
            ticker_symbol (str): Stock symbol
            df_with_indicators (pd.DataFrame): DataFrame with market data and indicators
            news_sentiment (dict, optional): News sentiment data
            
        Returns:
            dict: Analysis results
        """
        # Calculate basic statistics
        current_price = df_with_indicators['Close'].iloc[-1]
        prev_price = df_with_indicators['Close'].iloc[-2]
        daily_change = (current_price - prev_price) / prev_price * 100
        
        # Get highest and lowest prices in the dataset
        highest_price = df_with_indicators['High'].max()
        lowest_price = df_with_indicators['Low'].min()
        
        # Calculate potential support and resistance levels
        support_levels = [
            round(current_price * 0.95, 2),
            round(current_price * 0.90, 2),
            round(lowest_price, 2)
        ]
        
        resistance_levels = [
            round(current_price * 1.05, 2),
            round(current_price * 1.10, 2),
            round(highest_price, 2)
        ]
        
        # Determine market sentiment based on available indicators
        market_sentiment = "Neutral"
        sentiment_explanation = ""
        
        # Check RSI if available
        rsi_signal = "Neutral"
        if 'RSI_14' in df_with_indicators.columns:
            rsi = df_with_indicators['RSI_14'].iloc[-1]
            if rsi > 70:
                rsi_signal = "Overbought - Potential Sell Signal"
                market_sentiment = "Bearish"
                sentiment_explanation += f"RSI at {rsi:.2f} indicates overbought conditions. "
            elif rsi < 30:
                rsi_signal = "Oversold - Potential Buy Signal"
                market_sentiment = "Bullish"
                sentiment_explanation += f"RSI at {rsi:.2f} indicates oversold conditions. "
            else:
                sentiment_explanation += f"RSI at {rsi:.2f} is in neutral territory. "
        
        # Check MACD if available
        macd_signal = "Neutral"
        if 'MACD_12_26_9' in df_with_indicators.columns and 'MACD_SIGNAL_12_26_9' in df_with_indicators.columns:
            macd = df_with_indicators['MACD_12_26_9'].iloc[-1]
            macd_signal_line = df_with_indicators['MACD_SIGNAL_12_26_9'].iloc[-1]
            
            if macd > macd_signal_line:
                macd_signal = "Bullish Crossover"
                if market_sentiment != "Bullish":
                    market_sentiment = "Moderately Bullish"
                sentiment_explanation += "MACD is above signal line, indicating bullish momentum. "
            elif macd < macd_signal_line:
                macd_signal = "Bearish Crossover"
                if market_sentiment != "Bearish":
                    market_sentiment = "Moderately Bearish"
                sentiment_explanation += "MACD is below signal line, indicating bearish momentum. "
            else:
                sentiment_explanation += "MACD shows neutral momentum. "
        
        # Consider news sentiment if available
        if news_sentiment and news_sentiment.get('avg_sentiment') is not None:
            avg_sentiment = news_sentiment.get('avg_sentiment', 0)
            if avg_sentiment > 0.2:
                if market_sentiment in ["Neutral", "Moderately Bullish"]:
                    market_sentiment = "Bullish"
                sentiment_explanation += f"News sentiment is positive ({avg_sentiment:.2f}). "
            elif avg_sentiment < -0.2:
                if market_sentiment in ["Neutral", "Moderately Bearish"]:
                    market_sentiment = "Bearish"
                sentiment_explanation += f"News sentiment is negative ({avg_sentiment:.2f}). "
            else:
                sentiment_explanation += f"News sentiment is neutral ({avg_sentiment:.2f}). "
        
        # Determine recommendation based on overall analysis
        recommendation = "HOLD"
        confidence_score = 60  # Default moderate confidence
        
        if market_sentiment == "Bullish":
            recommendation = "BUY"
            confidence_score = 75
        elif market_sentiment == "Moderately Bullish":
            recommendation = "BUY"
            confidence_score = 65
        elif market_sentiment == "Bearish":
            recommendation = "SELL"
            confidence_score = 75
        elif market_sentiment == "Moderately Bearish":
            recommendation = "SELL"
            confidence_score = 65
        
        # Determine factors driving prices
        key_drivers = [
            f"Daily price movement ({daily_change:.2f}%)",
            "Technical indicators pattern"
        ]
        
        if 'RSI_14' in df_with_indicators.columns:
            key_drivers.append(f"RSI momentum ({rsi_signal})")
        
        if 'MACD_12_26_9' in df_with_indicators.columns:
            key_drivers.append(f"MACD trend ({macd_signal})")
            
        if news_sentiment and news_sentiment.get('article_count', 0) > 0:
            key_drivers.append(f"News sentiment ({news_sentiment.get('article_count')} articles)")
        
        # Identify potential risks
        risks = [
            "Market volatility",
            "Economic uncertainty",
            "Industry competition"
        ]
        
        if market_sentiment == "Bullish" or market_sentiment == "Moderately Bullish":
            risks.append("Potential overbought conditions if rally continues")
        else:
            risks.append("Continued downtrend if selling pressure persists")
        
        # Prepare analysis results
        analysis = {
            "status": "success",
            "technical_analysis": f"The price of {ticker_symbol} is currently at ${current_price:.2f}, showing a {daily_change:.2f}% change from the previous close. " + sentiment_explanation,
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
            "market_sentiment": f"{market_sentiment}. {sentiment_explanation}",
            "short_term_prediction": {
                "direction": "Up" if market_sentiment in ["Bullish", "Moderately Bullish"] else "Down" if market_sentiment in ["Bearish", "Moderately Bearish"] else "Sideways",
                "target_price": round(current_price * (1.05 if market_sentiment in ["Bullish", "Moderately Bullish"] else 0.95 if market_sentiment in ["Bearish", "Moderately Bearish"] else 1.0), 2),
                "timeframe": "7 days"
            },
            "medium_term_outlook": f"Based on current indicators, the medium-term outlook for {ticker_symbol} appears {market_sentiment.lower()}. The price may test the {'resistance' if market_sentiment in ['Bullish', 'Moderately Bullish'] else 'support'} levels in the coming weeks.",
            "key_drivers": key_drivers,
            "risks": risks,
            "recommendation": recommendation,
            "confidence_score": confidence_score
        }
        
        return analysis

def display_market_analysis(ticker_symbol, df_with_indicators, news_data=None):
    """
    Display market analysis using available data.
    
    Args:
        ticker_symbol (str): Stock symbol
        df_with_indicators (pd.DataFrame): DataFrame with technical indicators
        news_data (dict, optional): News sentiment data
    """
    st.subheader("Market Analysis")
    
    # Check if Gemini API key is available - for future API integration
    gemini_api_key = get_gemini_api_key()
    if not gemini_api_key:
        st.info("""
        Future enhancement: When you add a GEMINI_API_KEY in the Replit Secrets, 
        this analysis will use Google's Gemini AI for more sophisticated insights.
        
        How to set up your Gemini API key:
        1. Sign up for a free account at [Google AI Studio](https://makersuite.google.com/)
        2. Get your API key from your account dashboard
        3. Add your API key to the Replit Secrets by clicking on the lock icon in the sidebar
        4. Set the secret name to `GEMINI_API_KEY` and the value to your API key
        """)
    
    # Automatically display a market analysis
    with st.spinner("Analyzing market data..."):
            # Use the simplified analyzer
            analyzer = GeminiMarketAnalyzer()
            analysis = analyzer.analyze_market_trends(ticker_symbol, df_with_indicators, news_data)
            
            if analysis.get('status') == 'success':
                # Display the analysis
                st.subheader(f"Market Analysis for {ticker_symbol}")
                
                # Display the recommendation
                recommendation = analysis.get('recommendation', 'HOLD')
                confidence = analysis.get('confidence_score', 0)
                
                # Set color based on recommendation
                rec_color = "gray"
                if recommendation in ["BUY", "STRONG BUY"]:
                    rec_color = "green"
                elif recommendation in ["SELL", "STRONG SELL"]:
                    rec_color = "red"
                
                # Show recommendation and confidence
                col1, col2 = st.columns(2)
                col1.markdown(f"### Recommendation: <span style='color:{rec_color};font-size:24px'>{recommendation}</span>", unsafe_allow_html=True)
                col2.metric("Confidence Score", f"{confidence}%")
                
                # Technical Analysis
                st.subheader("Technical Analysis")
                st.write(analysis.get('technical_analysis', ''))
                
                # Support and Resistance
                st.subheader("Key Levels")
                support_levels = analysis.get('support_levels', [])
                resistance_levels = analysis.get('resistance_levels', [])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Support Levels")
                    for level in support_levels:
                        st.markdown(f"- ${level}")
                with col2:
                    st.markdown("#### Resistance Levels")
                    for level in resistance_levels:
                        st.markdown(f"- ${level}")
                
                # Market Sentiment
                st.subheader("Market Sentiment")
                st.write(analysis.get('market_sentiment', ''))
                
                # Price Predictions
                st.subheader("Price Predictions")
                st.markdown("#### Short-Term Prediction")
                short_term = analysis.get('short_term_prediction', {})
                if isinstance(short_term, dict):
                    for key, value in short_term.items():
                        st.markdown(f"**{key}:** {value}")
                else:
                    st.write(short_term)
                
                st.markdown("#### Medium-Term Outlook")
                st.write(analysis.get('medium_term_outlook', ''))
                
                # Key Drivers and Risks
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Key Drivers")
                    key_drivers = analysis.get('key_drivers', [])
                    for driver in key_drivers:
                        st.markdown(f"- {driver}")
                
                with col2:
                    st.subheader("Potential Risks")
                    risks = analysis.get('risks', [])
                    for risk in risks:
                        st.markdown(f"- {risk}")
            else:
                st.error(f"Error generating analysis: {analysis.get('message', 'Unknown error')}")
                
def display_advanced_prediction_dashboard(ticker_symbol, df_with_indicators):
    """
    Display advanced prediction dashboard tab
    
    Args:
        ticker_symbol (str): Stock symbol
        df_with_indicators (pd.DataFrame): DataFrame with technical indicators
    """
    st.header("Advanced Prediction & Market Analysis")
    
    st.markdown("""
    This advanced analysis combines sophisticated prediction algorithms with comprehensive 
    market trend analysis to provide more accurate predictions and market insights.
    
    **Features:**
    - Enhanced prediction model for advanced pattern recognition
    - Key support and resistance level identification
    - Market driver analysis
    - Advanced sentiment analysis
    
    Select a tab below to access different analysis features:
    """)
    
    # Create tabs for different analyses
    pred_tab, market_tab = st.tabs(["Enhanced Prediction", "Market Analysis"])
    
    with pred_tab:
        display_prediction_model(ticker_symbol, df_with_indicators)
    
    with market_tab:
        # Get news data for market analysis
        news_data = None
        try:
            from news_sentiment import get_news_for_ticker, get_sentiment_score
            articles = get_news_for_ticker(ticker_symbol, days_back=30)
            if articles:
                avg_sentiment, article_count, _ = get_sentiment_score(ticker_symbol, days_back=30)
                news_data = {
                    "articles": articles,
                    "avg_sentiment": avg_sentiment,
                    "article_count": article_count
                }
        except Exception:
            pass
            
        # Display market analysis
        display_market_analysis(ticker_symbol, df_with_indicators, news_data)