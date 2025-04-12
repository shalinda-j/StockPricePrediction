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
    
    # Create future predictions
    if model_type == "Linear Regression":
        # For Linear Regression, we can simply use time indices
        last_time_idx = X_test.iloc[-1]['Time_Index'] if len(X_test) > 0 else df.shape[0] - 1
        future_indices = np.array([last_time_idx + i + 1 for i in range(prediction_days)]).reshape(-1, 1)
        future_pred_scaled = model.predict(future_indices)
        
    else:  # Random Forest
        # For Random Forest, we need to create a proper feature set with lag values
        # Get the column names from X_test to ensure we use the same features
        feature_columns = X_test.columns.tolist()
        
        # Create a DataFrame to store future predictions
        future_X = pd.DataFrame(columns=feature_columns)
        
        # Extract the last window_size values from the original data for initial lags
        window_size = len(feature_columns) - 1  # Subtract 1 for Time_Index
        
        # Get the last time index from X_test
        last_time_idx = X_test.iloc[-1]['Time_Index'] if len(X_test) > 0 else df.shape[0] - 1
        
        # Get the last set of scaled values
        last_values = X_test.iloc[-1].copy()
        
        # Make predictions day by day
        for i in range(prediction_days):
            # Update the time index for the new day
            current_time_idx = last_time_idx + i + 1
            current_row = last_values.copy()
            current_row['Time_Index'] = current_time_idx
            
            # Make prediction for the current day
            current_pred = model.predict(current_row.values.reshape(1, -1))[0]
            
            # Add the row to future_X
            future_X.loc[i] = current_row
            
            # Update lag values for the next prediction
            # Shift all lag values (assuming column names like 'Lag_1', 'Lag_2'...)
            for j in range(1, window_size):
                lag_col = f'Lag_{j}'
                next_lag_col = f'Lag_{j+1}'
                if next_lag_col in current_row:
                    current_row[lag_col] = current_row[next_lag_col]
            
            # Set the newest lag value to the prediction we just made
            if 'Lag_1' in current_row:
                current_row['Lag_1'] = current_pred
            
            # Update last_values for the next iteration
            last_values = current_row
        
        # Make predictions for all future days
        future_pred_scaled = model.predict(future_X)
    
    # Reshape and inverse transform to get actual price values
    future_pred_reshaped = future_pred_scaled.reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_pred_reshaped).flatten()
    
    return actual_prices, predicted_prices, future_dates, future_predictions
