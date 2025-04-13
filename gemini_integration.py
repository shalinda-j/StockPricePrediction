import google.generativeai as genai
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import re
import json

class GeminiMarketAnalyzer:
    """
    Use Gemini AI to analyze market data and generate insights.
    This class integrates with Google's Gemini API to enhance market predictions.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the Gemini Market Analyzer.
        
        Args:
            api_key (str, optional): Gemini API key
        """
        self.api_key = api_key
        self.model = None
        self.initialized = False
    
    def initialize(self, api_key=None):
        """
        Initialize the Gemini API with the provided API key.
        
        Args:
            api_key (str, optional): Gemini API key
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        # Use provided API key or try to get it from secrets
        if api_key:
            self.api_key = api_key
        elif not self.api_key:
            try:
                self.api_key = st.secrets.get("GEMINI_API_KEY", None)
            except Exception:
                st.error("Gemini API key not found. Please provide an API key.")
                return False
        
        # Configure the Gemini API
        try:
            genai.configure(api_key=self.api_key)
            
            # Initialize Gemini Pro model
            self.model = genai.GenerativeModel('gemini-pro')
            
            # Test with a simple query to ensure it's working
            response = self.model.generate_content("Hello, are you ready to analyze financial markets?")
            
            self.initialized = True
            return True
        except Exception as e:
            st.error(f"Failed to initialize Gemini API: {str(e)}")
            return False
    
    def analyze_market_trends(self, ticker_symbol, market_data, news_sentiment=None):
        """
        Analyze market trends using Gemini AI.
        
        Args:
            ticker_symbol (str): Symbol of the stock/crypto
            market_data (pd.DataFrame): Dataframe with recent market data
            news_sentiment (dict, optional): News sentiment data if available
            
        Returns:
            dict: Analysis results containing insights and recommendations
        """
        if not self.initialized:
            success = self.initialize()
            if not success:
                return {
                    "status": "error",
                    "message": "Failed to initialize Gemini API. Please provide a valid API key."
                }
        
        # Prepare market data summary
        data_summary = self._prepare_market_summary(ticker_symbol, market_data)
        
        # Prepare news sentiment summary if available
        sentiment_summary = ""
        if news_sentiment and news_sentiment.get('article_count', 0) > 0:
            sentiment_summary = self._prepare_sentiment_summary(news_sentiment)
        
        # Create the prompt for Gemini
        prompt = f"""
        I need a comprehensive market analysis for {ticker_symbol} based on the following data:
        
        {data_summary}
        
        {sentiment_summary}
        
        Please provide:
        1. A technical analysis of recent price movements and indicators
        2. Key support and resistance levels
        3. Market sentiment analysis
        4. A short-term price prediction (1-7 days)
        5. A medium-term outlook (1-4 weeks)
        6. Key factors driving the price
        7. Potential risks to watch
        8. A clear trading recommendation (BUY, SELL, HOLD)
        
        Format your response as a structured JSON object with the following keys:
        - technical_analysis (string): Detailed technical analysis
        - support_levels (array of numbers): Key support levels
        - resistance_levels (array of numbers): Key resistance levels
        - market_sentiment (string): Analysis of market sentiment
        - short_term_prediction (object): Short-term prediction with price targets
        - medium_term_outlook (string): Medium-term outlook
        - key_drivers (array of strings): Key factors driving the price
        - risks (array of strings): Potential risks to watch
        - recommendation (string): Trading recommendation (BUY, SELL, HOLD)
        - confidence_score (number): Confidence score for the analysis (0-100)
        
        Note that your response MUST be valid JSON that can be parsed.
        """
        
        # Generate the analysis
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Extract JSON from the response
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find any JSON object
                json_match = re.search(r'({.*})', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response_text
            
            # Clean and parse the JSON
            try:
                analysis = json.loads(json_str)
                analysis['status'] = 'success'
                return analysis
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw response
                return {
                    "status": "partial",
                    "message": "Failed to parse structured response",
                    "raw_analysis": response_text
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to generate analysis: {str(e)}"
            }
    
    def enhance_price_predictions(self, ticker_symbol, model_predictions, technical_signals=None, news_sentiment=None):
        """
        Use Gemini AI to enhance price predictions from the neural network model.
        
        Args:
            ticker_symbol (str): Symbol of the stock/crypto
            model_predictions (dict): Predictions from the neural network model
            technical_signals (dict, optional): Technical analysis signals
            news_sentiment (dict, optional): News sentiment data
            
        Returns:
            dict: Enhanced predictions
        """
        if not self.initialized:
            success = self.initialize()
            if not success:
                return model_predictions
        
        # Extract predictions info
        predicted_dates = model_predictions.get('dates', [])
        predicted_prices = model_predictions.get('prices', [])
        last_price = model_predictions.get('last_price', 0)
        
        # Format the predictions for the prompt
        predictions_text = ""
        for i, (date, price) in enumerate(zip(predicted_dates, predicted_prices)):
            predictions_text += f"Day {i+1} ({date}): {price:.2f}\n"
        
        # Format technical signals if available
        signals_text = ""
        if technical_signals:
            signals_text = "Technical Signals:\n"
            for indicator, info in technical_signals.items():
                if indicator != 'OVERALL':
                    signal = info.get('signal', 'NEUTRAL')
                    signals_text += f"- {indicator}: {signal}\n"
            signals_text += f"Overall Signal: {technical_signals.get('OVERALL', {}).get('signal', 'NEUTRAL')}\n"
        
        # Format sentiment if available
        sentiment_text = ""
        if news_sentiment and news_sentiment.get('article_count', 0) > 0:
            sentiment = news_sentiment.get('avg_sentiment', 0)
            sentiment_text = f"News Sentiment Score: {sentiment:.2f} (based on {news_sentiment.get('article_count', 0)} articles)\n"
        
        # Create the prompt
        prompt = f"""
        I need your help to enhance price predictions for {ticker_symbol}.
        
        The model has provided the following price predictions:
        Current price: {last_price:.2f}
        {predictions_text}
        
        {signals_text}
        {sentiment_text}
        
        Please analyze these predictions and provide:
        1. Adjusted price targets based on recent events and market sentiment
        2. Confidence intervals for each prediction (best case, expected case, worst case)
        3. Key factors that could accelerate or reverse the predicted trend
        4. A probability score (0-100%) that the predicted trend will materialize
        
        Format your response as a structured JSON object with the following keys:
        - adjusted_predictions (array of objects): Each with date, original_price, adjusted_price, lower_bound, upper_bound
        - key_factors (array of strings): Factors that could influence the predictions
        - probability_score (number): Probability the trend will materialize (0-100)
        - reasoning (string): Your reasoning for the adjustments
        - recommendation (string): Based on the enhanced analysis (BUY, SELL, HOLD)
        
        Ensure your response is valid JSON that can be parsed.
        """
        
        # Generate the enhanced predictions
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Extract JSON from the response
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find any JSON object
                json_match = re.search(r'({.*})', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response_text
            
            # Clean and parse the JSON
            try:
                enhancements = json.loads(json_str)
                
                # Combine original predictions with enhancements
                enhanced_predictions = model_predictions.copy()
                enhanced_predictions['gemini_enhanced'] = True
                enhanced_predictions['adjusted_predictions'] = enhancements.get('adjusted_predictions', [])
                enhanced_predictions['key_factors'] = enhancements.get('key_factors', [])
                enhanced_predictions['probability_score'] = enhancements.get('probability_score', 0)
                enhanced_predictions['reasoning'] = enhancements.get('reasoning', '')
                enhanced_predictions['recommendation'] = enhancements.get('recommendation', 'HOLD')
                
                return enhanced_predictions
            except json.JSONDecodeError:
                # If JSON parsing fails, return original predictions with raw response
                model_predictions['gemini_enhanced'] = False
                model_predictions['raw_enhancement'] = response_text
                return model_predictions
                
        except Exception as e:
            model_predictions['gemini_enhanced'] = False
            model_predictions['enhancement_error'] = str(e)
            return model_predictions
    
    def analyze_news_impact(self, ticker_symbol, news_articles):
        """
        Analyze the impact of news articles on the stock/crypto price.
        
        Args:
            ticker_symbol (str): Symbol of the stock/crypto
            news_articles (list): List of news articles
            
        Returns:
            dict: Analysis of news impact
        """
        if not self.initialized:
            success = self.initialize()
            if not success:
                return {
                    "status": "error",
                    "message": "Failed to initialize Gemini API. Please provide a valid API key."
                }
        
        if not news_articles:
            return {
                "status": "error",
                "message": "No news articles provided for analysis"
            }
        
        # Format articles for the prompt
        articles_text = ""
        for i, article in enumerate(news_articles[:5]):  # Limit to 5 articles to avoid token limits
            title = article.get('title', 'No title')
            desc = article.get('description', 'No description')
            date = article.get('published_at', 'Unknown date')
            articles_text += f"Article {i+1} ({date}):\nTitle: {title}\nDescription: {desc}\n\n"
        
        # Create the prompt
        prompt = f"""
        Please analyze the following news articles about {ticker_symbol} and assess their potential impact on the price:
        
        {articles_text}
        
        Please provide:
        1. A summary of key themes from these articles
        2. Sentiment analysis for each article (positive, negative, neutral)
        3. Potential short-term price impact
        4. Potential long-term price impact
        5. Key risks or opportunities mentioned
        
        Format your response as a structured JSON object with the following keys:
        - key_themes (array of strings): Key themes from the articles
        - article_sentiments (array of objects): Each with title, sentiment, impact_score (-5 to +5)
        - short_term_impact (string): Short-term price impact assessment
        - long_term_impact (string): Long-term price impact assessment
        - risks (array of strings): Key risks mentioned
        - opportunities (array of strings): Key opportunities mentioned
        - overall_sentiment (number): Overall sentiment score (-1 to +1)
        
        Ensure your response is valid JSON that can be parsed.
        """
        
        # Generate the analysis
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Extract JSON from the response
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find any JSON object
                json_match = re.search(r'({.*})', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response_text
            
            # Clean and parse the JSON
            try:
                analysis = json.loads(json_str)
                analysis['status'] = 'success'
                return analysis
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw response
                return {
                    "status": "partial",
                    "message": "Failed to parse structured response",
                    "raw_analysis": response_text
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to generate news impact analysis: {str(e)}"
            }
    
    def _prepare_market_summary(self, ticker_symbol, market_data):
        """
        Prepare a summary of market data for the prompt.
        
        Args:
            ticker_symbol (str): Symbol of the stock/crypto
            market_data (pd.DataFrame): Market data
            
        Returns:
            str: Summary text
        """
        if market_data.empty:
            return "No market data available."
        
        # Get recent price data
        recent_data = market_data.tail(10)  # Last 10 days
        
        # Calculate basic statistics
        latest_price = market_data['Close'].iloc[-1]
        price_change_1d = market_data['Close'].iloc[-1] - market_data['Close'].iloc[-2]
        price_change_pct_1d = (price_change_1d / market_data['Close'].iloc[-2]) * 100
        
        price_change_5d = market_data['Close'].iloc[-1] - market_data['Close'].iloc[-6] if len(market_data) >= 6 else None
        price_change_pct_5d = (price_change_5d / market_data['Close'].iloc[-6]) * 100 if price_change_5d is not None else None
        
        price_change_30d = market_data['Close'].iloc[-1] - market_data['Close'].iloc[-31] if len(market_data) >= 31 else None
        price_change_pct_30d = (price_change_30d / market_data['Close'].iloc[-31]) * 100 if price_change_30d is not None else None
        
        # Calculate volume statistics
        avg_volume = market_data['Volume'].mean() if 'Volume' in market_data.columns else None
        recent_volume = market_data['Volume'].iloc[-1] if 'Volume' in market_data.columns else None
        volume_change = ((recent_volume / avg_volume) - 1) * 100 if avg_volume and recent_volume else None
        
        # Check for technical indicators
        has_rsi = 'RSI_14' in market_data.columns
        has_macd = 'MACD_12_26_9' in market_data.columns
        has_bollinger = 'BB_UPPER_20' in market_data.columns
        
        # Prepare technical indicators summary
        indicators_summary = ""
        if has_rsi:
            latest_rsi = market_data['RSI_14'].iloc[-1]
            indicators_summary += f"RSI (14): {latest_rsi:.2f}\n"
        
        if has_macd:
            latest_macd = market_data['MACD_12_26_9'].iloc[-1]
            latest_macd_signal = market_data['MACD_SIGNAL_12_26_9'].iloc[-1] if 'MACD_SIGNAL_12_26_9' in market_data.columns else None
            latest_macd_hist = market_data['MACD_HIST_12_26_9'].iloc[-1] if 'MACD_HIST_12_26_9' in market_data.columns else None
            
            indicators_summary += f"MACD: {latest_macd:.2f}\n"
            if latest_macd_signal is not None:
                indicators_summary += f"MACD Signal: {latest_macd_signal:.2f}\n"
            if latest_macd_hist is not None:
                indicators_summary += f"MACD Histogram: {latest_macd_hist:.2f}\n"
        
        if has_bollinger:
            latest_bb_upper = market_data['BB_UPPER_20'].iloc[-1]
            latest_bb_middle = market_data['BB_MIDDLE_20'].iloc[-1] if 'BB_MIDDLE_20' in market_data.columns else None
            latest_bb_lower = market_data['BB_LOWER_20'].iloc[-1] if 'BB_LOWER_20' in market_data.columns else None
            
            indicators_summary += f"Bollinger Bands:\n"
            indicators_summary += f"  - Upper: {latest_bb_upper:.2f}\n"
            if latest_bb_middle is not None:
                indicators_summary += f"  - Middle: {latest_bb_middle:.2f}\n"
            if latest_bb_lower is not None:
                indicators_summary += f"  - Lower: {latest_bb_lower:.2f}\n"
        
        # Create the summary text
        summary = f"""Market Data Summary for {ticker_symbol}:

Current Price: {latest_price:.2f}

Price Changes:
- 1 Day: {price_change_1d:.2f} ({price_change_pct_1d:.2f}%)
"""
        
        if price_change_5d is not None:
            summary += f"- 5 Days: {price_change_5d:.2f} ({price_change_pct_5d:.2f}%)\n"
        
        if price_change_30d is not None:
            summary += f"- 30 Days: {price_change_30d:.2f} ({price_change_pct_30d:.2f}%)\n"
        
        if avg_volume is not None and recent_volume is not None:
            summary += f"""
Volume:
- Recent Volume: {recent_volume:.0f}
- Average Volume: {avg_volume:.0f}
- Volume Change: {volume_change:.2f}%
"""
        
        if indicators_summary:
            summary += f"""
Technical Indicators:
{indicators_summary}
"""
        
        # Add recent price data
        summary += """
Recent Price Data (last 10 days, most recent first):
"""
        for i, (index, row) in enumerate(recent_data.iloc[::-1].iterrows()):
            date = index.strftime('%Y-%m-%d') if hasattr(index, 'strftime') else str(index)
            summary += f"{date}: Open: {row['Open']:.2f}, High: {row['High']:.2f}, Low: {row['Low']:.2f}, Close: {row['Close']:.2f}"
            if 'Volume' in row:
                summary += f", Volume: {row['Volume']:.0f}"
            summary += "\n"
        
        return summary
    
    def _prepare_sentiment_summary(self, sentiment_data):
        """
        Prepare a summary of sentiment data for the prompt.
        
        Args:
            sentiment_data (dict): Sentiment data
            
        Returns:
            str: Summary text
        """
        avg_sentiment = sentiment_data.get('avg_sentiment', 0)
        article_count = sentiment_data.get('article_count', 0)
        sentiment_by_day = sentiment_data.get('sentiment_by_day', {})
        
        # Determine sentiment description
        sentiment_desc = "Neutral"
        if avg_sentiment > 0.2:
            sentiment_desc = "Strongly Positive"
        elif avg_sentiment > 0.05:
            sentiment_desc = "Positive"
        elif avg_sentiment < -0.2:
            sentiment_desc = "Strongly Negative"
        elif avg_sentiment < -0.05:
            sentiment_desc = "Negative"
        
        # Create the summary text
        summary = f"""News Sentiment Analysis:

Overall Sentiment: {avg_sentiment:.2f} ({sentiment_desc})
Articles Analyzed: {article_count}
"""
        
        # Add sentiment by day if available
        if sentiment_by_day:
            summary += """
Sentiment by Day:
"""
            for date, score in sentiment_by_day.items():
                summary += f"{date}: {score:.2f}\n"
        
        return summary