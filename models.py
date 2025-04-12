import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

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

def train_lstm_model(X_train, y_train, sequence_length, epochs=50):
    """
    Train an LSTM model for time series prediction.
    
    Args:
        X_train (numpy.ndarray): Training features shaped for LSTM [samples, time steps, features]
        y_train (numpy.ndarray): Training target (scaled prices)
        sequence_length (int): Length of input sequences
        epochs (int): Number of training epochs
        
    Returns:
        Sequential: Trained LSTM model
    """
    # Build the LSTM model
    model = Sequential()
    
    # First LSTM layer with dropout
    model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(Dropout(0.2))
    
    # Second LSTM layer with dropout
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(units=1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    
    return model

def predict_prices(model, X_test, y_test, df, scaler, model_type, prediction_days=7, sequence_length=30, last_sequence=None):
    """
    Make predictions using the trained model and prepare future predictions.
    
    Args:
        model: Trained model (LinearRegression or LSTM)
        X_test: Test features
        y_test: Test targets
        df: DataFrame with historical data
        scaler: Trained scaler to transform data
        model_type (str): 'Linear Regression' or 'LSTM (Long Short-Term Memory)'
        prediction_days (int): Number of days to predict into the future
        sequence_length (int): Length of input sequences for LSTM
        last_sequence: Last sequence from the dataset for LSTM prediction
        
    Returns:
        tuple: (actual_prices, predicted_prices, future_dates, future_predictions)
    """
    # Make predictions on test data
    if model_type == "Linear Regression":
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
        
    else:  # LSTM model
        predictions = model.predict(X_test)
        
        # Inverse transform to get actual price values
        predicted_prices = scaler.inverse_transform(predictions).flatten()
        
        # Get actual prices from test data
        actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Generate future predictions
        last_idx = df.index[-1]
        future_dates = [last_idx + timedelta(days=i+1) for i in range(prediction_days)]
        
        # Make predictions for future dates
        future_predictions = []
        curr_sequence = last_sequence.copy()
        
        for _ in range(prediction_days):
            # Predict the next price
            next_pred = model.predict(curr_sequence)[0][0]
            future_predictions.append(next_pred)
            
            # Update the sequence for the next prediction
            curr_sequence = np.append(curr_sequence[:, 1:, :], 
                                      [[next_pred]], 
                                      axis=1)
        
        # Inverse transform the future predictions
        future_predictions = scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)
        ).flatten()
    
    return actual_prices, predicted_prices, future_dates, future_predictions
