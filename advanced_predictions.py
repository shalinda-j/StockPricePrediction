import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import json

from neural_network_model import AdvancedNeuralNetworkModel
from gemini_integration import GeminiMarketAnalyzer
from technical_indicators import get_technical_signals

def get_gemini_api_key():
    """
    Get the Gemini API key from secrets
    """
    try:
        return st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        return None

def prepare_data_for_nn_model(df_with_indicators):
    """
    Prepare data for the neural network model.
    
    Args:
        df_with_indicators (pd.DataFrame): DataFrame with price data and technical indicators
        
    Returns:
        pd.DataFrame: Prepared DataFrame
    """
    # Make a copy to avoid modifying the original
    df = df_with_indicators.copy()
    
    # Ensure all required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column {col} not found in the data")
    
    # Drop any other columns that might cause issues
    columns_to_keep = required_columns + [col for col in df.columns if col.startswith(('RSI', 'MACD', 'BB_'))]
    df = df[columns_to_keep]
    
    # Forward fill any NaN values that might be in the technical indicators
    df = df.fillna(method='ffill')
    
    # Drop any remaining NaN values at the beginning of the DataFrame
    df = df.dropna()
    
    return df

def train_nn_model(df_prepared, sequence_length=60, lstm_units=100, dense_units=64, batch_size=32, epochs=50):
    """
    Train the neural network model.
    
    Args:
        df_prepared (pd.DataFrame): Prepared DataFrame
        sequence_length (int): Number of time steps to look back
        lstm_units (int): Number of LSTM units
        dense_units (int): Number of dense layer units
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        
    Returns:
        tuple: (model, history, metrics, future_predictions)
    """
    # Initialize the model
    model = AdvancedNeuralNetworkModel(sequence_length=sequence_length, 
                                       lstm_units=lstm_units, 
                                       dense_units=dense_units)
    
    # Prepare data for model
    X_train, X_test, y_train, y_test = model.prepare_data(df_prepared)
    
    # Train the model
    history = model.train(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    metrics = model.evaluate(X_test, y_test)
    
    # Generate future predictions
    future_dates, future_prices = model.forecast_future(days=7)
    
    # Create a dictionary with all results
    results = {
        'model': model,
        'history': history,
        'metrics': metrics,
        'actual': model.scaler.inverse_transform(np.hstack([np.zeros((len(y_test), model.feature_count-1)), y_test.reshape(-1, 1)]))[:, -1],
        'predicted': y_pred,
        'future_dates': future_dates,
        'future_prices': future_prices,
        'last_price': df_prepared['Close'].iloc[-1]
    }
    
    return results

def enhance_predictions_with_gemini(ticker_symbol, nn_results, df_with_indicators):
    """
    Enhance predictions with Gemini AI.
    
    Args:
        ticker_symbol (str): Stock symbol
        nn_results (dict): Neural network results
        df_with_indicators (pd.DataFrame): DataFrame with technical indicators
        
    Returns:
        dict: Enhanced predictions
    """
    # Initialize Gemini analyzer
    gemini = GeminiMarketAnalyzer(api_key=get_gemini_api_key())
    
    # Check if initialization was successful
    if not gemini.initialize():
        st.warning("Gemini API integration not available. Proceeding with neural network predictions only.")
        return nn_results
    
    # Extract neural network predictions
    predictions = {
        'dates': nn_results['future_dates'],
        'prices': nn_results['future_prices'],
        'last_price': nn_results['last_price']
    }
    
    # Get technical signals
    signals = get_technical_signals(df_with_indicators)
    
    # Use Gemini to enhance predictions
    enhanced_predictions = gemini.enhance_price_predictions(
        ticker_symbol, 
        predictions, 
        technical_signals=signals
    )
    
    return enhanced_predictions

def display_advanced_nn_dashboard(ticker_symbol, df_with_indicators):
    """
    Display advanced neural network prediction dashboard.
    
    Args:
        ticker_symbol (str): Stock symbol
        df_with_indicators (pd.DataFrame): DataFrame with technical indicators
    """
    st.subheader("Advanced Neural Network Model Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        sequence_length = st.slider("Sequence Length (days to look back):", 10, 100, 60)
        lstm_units = st.slider("LSTM Units:", 50, 200, 100)
    with col2:
        dense_units = st.slider("Dense Units:", 32, 128, 64)
        epochs = st.slider("Training Epochs:", 10, 100, 50)
    
    use_gemini = st.checkbox("Enhance Predictions with Gemini AI", value=True)
    
    if use_gemini:
        gemini_api_key = get_gemini_api_key()
        if not gemini_api_key:
            st.warning("Gemini API key not found. Please add a GEMINI_API_KEY in the Replit Secrets to use this feature.")
            st.markdown("""
            ### How to set up your Gemini API key:
            
            1. Sign up for a free account at [Google AI Studio](https://makersuite.google.com/)
            2. Get your API key from your account dashboard
            3. Add your API key to the Replit Secrets by clicking on the lock icon in the sidebar
            4. Set the secret name to `GEMINI_API_KEY` and the value to your API key
            """)
    
    # Button to train the model
    if st.button("Train Advanced Neural Network Model"):
        with st.spinner("Preparing data for neural network model..."):
            try:
                # Prepare data for the neural network model
                df_prepared = prepare_data_for_nn_model(df_with_indicators)
                
                if len(df_prepared) < sequence_length + 10:
                    st.error(f"Not enough data points. Need at least {sequence_length + 10} valid data points. Please increase the date range.")
                    return
                
                # Train the model and get predictions
                with st.spinner(f"Training neural network model with {len(df_prepared)} data points..."):
                    nn_results = train_nn_model(
                        df_prepared,
                        sequence_length=sequence_length,
                        lstm_units=lstm_units,
                        dense_units=dense_units,
                        epochs=epochs
                    )
                
                # Display the model metrics
                st.subheader("Model Performance")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MAE", f"${nn_results['metrics']['mae']:.2f}")
                col2.metric("MSE", f"${nn_results['metrics']['mse']:.2f}")
                col3.metric("RMSE", f"${nn_results['metrics']['rmse']:.2f}")
                col4.metric("MAPE", f"{nn_results['metrics']['mape']:.2f}%")
                
                # Plot the training history
                st.subheader("Training History")
                fig = nn_results['model'].plot_training_history(nn_results['history'])
                st.pyplot(fig)
                
                # Plot the predictions
                st.subheader("Predictions vs Actual")
                fig = nn_results['model'].plot_predictions(
                    nn_results['actual'], 
                    nn_results['predicted'],
                    title=f"Neural Network Predictions vs Actual Prices for {ticker_symbol}"
                )
                st.pyplot(fig)
                
                # If Gemini is enabled, enhance predictions
                enhanced_predictions = None
                if use_gemini and get_gemini_api_key():
                    with st.spinner("Enhancing predictions with Gemini AI..."):
                        enhanced_predictions = enhance_predictions_with_gemini(
                            ticker_symbol,
                            nn_results,
                            df_with_indicators
                        )
                
                # Display future predictions
                st.subheader(f"Future Price Predictions for {ticker_symbol}")
                
                # Plot future predictions
                fig = nn_results['model'].plot_future_predictions(
                    nn_results['future_dates'],
                    nn_results['future_prices'],
                    nn_results['last_price'],
                    title=f"Future Price Predictions for {ticker_symbol}"
                )
                st.pyplot(fig)
                
                # Display predictions in a table
                st.subheader("Predicted Prices for Next 7 Days")
                
                if enhanced_predictions and enhanced_predictions.get('gemini_enhanced', False):
                    # Display enhanced predictions
                    adj_preds = enhanced_predictions.get('adjusted_predictions', [])
                    
                    if adj_preds:
                        # Create a DataFrame for the table
                        pred_df = pd.DataFrame(adj_preds)
                        st.dataframe(pred_df)
                        
                        # Display Gemini's analysis
                        st.subheader("Gemini AI Enhanced Analysis")
                        
                        # Display probability and recommendation
                        col1, col2 = st.columns(2)
                        probability = enhanced_predictions.get('probability_score', 0)
                        recommendation = enhanced_predictions.get('recommendation', 'HOLD')
                        
                        # Set color based on recommendation
                        rec_color = "gray"
                        if recommendation == "BUY":
                            rec_color = "green"
                        elif recommendation == "SELL":
                            rec_color = "red"
                        
                        col1.metric("Probability Score", f"{probability}%")
                        col2.markdown(f"**Recommendation:** <span style='color:{rec_color}'>{recommendation}</span>", unsafe_allow_html=True)
                        
                        # Display key factors
                        st.subheader("Key Factors")
                        key_factors = enhanced_predictions.get('key_factors', [])
                        for factor in key_factors:
                            st.markdown(f"- {factor}")
                        
                        # Display reasoning
                        st.subheader("Analysis")
                        st.write(enhanced_predictions.get('reasoning', ''))
                    else:
                        # Create a simple DataFrame with original predictions
                        pred_df = pd.DataFrame({
                            'Date': nn_results['future_dates'],
                            'Predicted Price': nn_results['future_prices'],
                            'Change (%)': [(price - nn_results['last_price']) / nn_results['last_price'] * 100 
                                          for price in nn_results['future_prices']]
                        })
                        st.dataframe(pred_df)
                        
                        # If Gemini was enabled but parsing failed, show raw response
                        if 'raw_enhancement' in enhanced_predictions:
                            st.subheader("Gemini AI Analysis")
                            st.text(enhanced_predictions.get('raw_enhancement', ''))
                else:
                    # Create a simple DataFrame with original predictions
                    pred_df = pd.DataFrame({
                        'Date': nn_results['future_dates'],
                        'Predicted Price': nn_results['future_prices'],
                        'Change (%)': [(price - nn_results['last_price']) / nn_results['last_price'] * 100 
                                      for price in nn_results['future_prices']]
                    })
                    st.dataframe(pred_df)
                
                # Save the model if the user wants to
                if st.button("Save Neural Network Model"):
                    nn_results['model'].save_model(f"{ticker_symbol}_nn_model.h5")
                    st.success(f"Model saved as {ticker_symbol}_nn_model.h5")
                
            except Exception as e:
                st.error(f"Error training neural network model: {str(e)}")
                import traceback
                st.text(traceback.format_exc())

def display_gemini_market_analysis(ticker_symbol, df_with_indicators, news_data=None):
    """
    Display comprehensive market analysis using Gemini AI.
    
    Args:
        ticker_symbol (str): Stock symbol
        df_with_indicators (pd.DataFrame): DataFrame with technical indicators
        news_data (dict, optional): News sentiment data
    """
    st.subheader("Gemini AI Market Analysis")
    
    # Check if Gemini API key is available
    gemini_api_key = get_gemini_api_key()
    if not gemini_api_key:
        st.warning("Gemini API key not found. Please add a GEMINI_API_KEY in the Replit Secrets to use this feature.")
        st.markdown("""
        ### How to set up your Gemini API key:
        
        1. Sign up for a free account at [Google AI Studio](https://makersuite.google.com/)
        2. Get your API key from your account dashboard
        3. Add your API key to the Replit Secrets by clicking on the lock icon in the sidebar
        4. Set the secret name to `GEMINI_API_KEY` and the value to your API key
        """)
        return
    
    # Initialize Gemini analyzer
    gemini = GeminiMarketAnalyzer(api_key=gemini_api_key)
    
    # Check if initialization was successful
    if not gemini.initialize():
        st.error("Failed to initialize Gemini API. Please check your API key.")
        return
    
    # Button to generate analysis
    if st.button("Generate Comprehensive Market Analysis"):
        with st.spinner("Generating market analysis with Gemini AI..."):
            # Generate market analysis
            analysis = gemini.analyze_market_trends(ticker_symbol, df_with_indicators, news_data)
            
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
            
            elif analysis.get('status') == 'partial':
                # Display raw analysis if structured parsing failed
                st.warning("Could not parse structured analysis. Displaying raw analysis:")
                st.text(analysis.get('raw_analysis', ''))
            
            else:
                # Display error
                st.error(f"Error generating analysis: {analysis.get('message', 'Unknown error')}")
    
    # If news data is available, offer news impact analysis
    if news_data and news_data.get('articles'):
        if st.button("Analyze News Impact"):
            with st.spinner("Analyzing news impact with Gemini AI..."):
                # Generate news impact analysis
                news_impact = gemini.analyze_news_impact(ticker_symbol, news_data.get('articles', []))
                
                if news_impact.get('status') == 'success':
                    # Display the news impact analysis
                    st.subheader("News Impact Analysis")
                    
                    # Overall sentiment
                    overall_sentiment = news_impact.get('overall_sentiment', 0)
                    sentiment_color = "gray"
                    if overall_sentiment > 0.2:
                        sentiment_color = "green"
                    elif overall_sentiment < -0.2:
                        sentiment_color = "red"
                    
                    st.markdown(f"#### Overall Sentiment: <span style='color:{sentiment_color}'>{overall_sentiment:.2f}</span>", unsafe_allow_html=True)
                    
                    # Key themes
                    st.subheader("Key Themes")
                    key_themes = news_impact.get('key_themes', [])
                    for theme in key_themes:
                        st.markdown(f"- {theme}")
                    
                    # Article sentiments
                    st.subheader("Article Sentiments")
                    article_sentiments = news_impact.get('article_sentiments', [])
                    
                    # Create a table for article sentiments
                    if article_sentiments:
                        sentiment_df = pd.DataFrame(article_sentiments)
                        st.dataframe(sentiment_df)
                    
                    # Impact assessment
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Short-Term Impact")
                        st.write(news_impact.get('short_term_impact', ''))
                    
                    with col2:
                        st.markdown("#### Long-Term Impact")
                        st.write(news_impact.get('long_term_impact', ''))
                    
                    # Risks and opportunities
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Risks")
                        risks = news_impact.get('risks', [])
                        for risk in risks:
                            st.markdown(f"- {risk}")
                    
                    with col2:
                        st.subheader("Opportunities")
                        opportunities = news_impact.get('opportunities', [])
                        for opportunity in opportunities:
                            st.markdown(f"- {opportunity}")
                
                elif news_impact.get('status') == 'partial':
                    # Display raw analysis if structured parsing failed
                    st.warning("Could not parse structured news impact analysis. Displaying raw analysis:")
                    st.text(news_impact.get('raw_analysis', ''))
                
                else:
                    # Display error
                    st.error(f"Error analyzing news impact: {news_impact.get('message', 'Unknown error')}")