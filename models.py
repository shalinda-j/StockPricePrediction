import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    
    Args:
        X_train (numpy.ndarray): Training features (time index)
        y_train (numpy.ndarray): Training target (scaled prices)
        
    Returns:
        LinearRegression: Trained Linear Regression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=100):
    """
    Train a Random Forest model for time series prediction.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training target (scaled prices)
        n_estimators (int): Number of trees in the forest
        
    Returns:
        RandomForestRegressor: Trained Random Forest model
    """
    # Build the Random Forest model
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model

def predict_prices(model, X_test, y_test, df, scaler, model_type, prediction_days=7, sequence_length=30, last_sequence=None):
    """
    Make predictions using the trained model and prepare future predictions.
    
    Args:
        model: Trained model (LinearRegression or RandomForest)
        X_test: Test features
        y_test: Test targets
        df: DataFrame with historical data
        scaler: Trained scaler to transform data
        model_type (str): 'Linear Regression' or 'Random Forest'
        prediction_days (int): Number of days to predict into the future
        sequence_length (int): Length of input sequences (for future implementations)
        last_sequence: Last sequence from the dataset (for future implementations)
        
    Returns:
        tuple: (actual_prices, predicted_prices, future_dates, future_predictions)
    """
    # Make predictions on test data
    predictions = model.predict(X_test)
    
    # Inverse transform to get actual price values
    predictions_reshaped = predictions.reshape(-1, 1)
    predicted_prices = scaler.inverse_transform(predictions_reshaped).flatten()
    
    # Get actual prices from test data
    y_test_reshaped = y_test.values.reshape(-1, 1)
    actual_prices = scaler.inverse_transform(y_test_reshaped).flatten()
    
    # Generate future predictions
    last_idx = df.index[-1]
    future_dates = [last_idx + timedelta(days=i+1) for i in range(prediction_days)]
    
    # Create time indices for future predictions
    last_time_idx = X_test.iloc[-1]['Time_Index'] if len(X_test) > 0 else df.shape[0] - 1
    future_indices = np.array([last_time_idx + i + 1 for i in range(prediction_days)]).reshape(-1, 1)
    
    # Make predictions for future dates
    future_pred_scaled = model.predict(future_indices)
    future_pred_reshaped = future_pred_scaled.reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_pred_reshaped).flatten()
    
    return actual_prices, predicted_prices, future_dates, future_predictions
