import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    """
    Preprocess the data for model training.
    
    Args:
        df (pandas.DataFrame): DataFrame containing historical price data
        
    Returns:
        pandas.DataFrame: Preprocessed DataFrame
    """
    # Make a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Keep only the 'Close' price column
    if 'Close' in processed_df.columns:
        processed_df = processed_df[['Close']]
    else:
        raise ValueError("DataFrame does not contain 'Close' column")
    
    # Check for and handle NaN values
    if processed_df.isnull().any().any():
        processed_df = processed_df.fillna(method='ffill')  # Forward fill
        processed_df = processed_df.fillna(method='bfill')  # Backward fill if any remain
    
    return processed_df

def prepare_data_for_lr(df, test_size=0.2):
    """
    Prepare data for Linear Regression model.
    
    Args:
        df (pandas.DataFrame): DataFrame with 'Close' price
        test_size (float): Proportion of data to use for testing
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # Create a copy of the dataframe
    data = df.copy()
    
    # Create a feature representing the time index
    data['Time_Index'] = np.arange(len(data))
    
    # Scale the 'Close' price
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_price_scaled = scaler.fit_transform(data[['Close']])
    data['Close_Scaled'] = close_price_scaled
    
    # Prepare features (X) and target (y)
    X = data[['Time_Index']]
    y = data['Close_Scaled']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    return X_train, X_test, y_train, y_test, scaler

def prepare_data_for_lstm(df, sequence_length=30, test_size=0.2):
    """
    Prepare data for LSTM model.
    
    Args:
        df (pandas.DataFrame): DataFrame with 'Close' price
        sequence_length (int): Number of previous days to use for prediction
        test_size (float): Proportion of data to use for testing
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, last_sequence)
    """
    # Create a copy of the dataframe
    data = df.copy()
    
    # Extract only the 'Close' prices
    close_prices = data['Close'].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    # Create sequences for LSTM
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    # Convert to numpy arrays
    X, y = np.array(X), np.array(y)
    
    # Reshape for LSTM [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Split the data into training and testing sets
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Get the last sequence for future prediction
    last_sequence = scaled_data[-sequence_length:]
    last_sequence = np.reshape(last_sequence, (1, sequence_length, 1))
    
    return X_train, X_test, y_train, y_test, scaler, last_sequence
