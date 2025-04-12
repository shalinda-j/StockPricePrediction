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

def prepare_data_for_random_forest(df, test_size=0.2, window_size=5):
    """
    Prepare data for Random Forest model.
    
    Args:
        df (pandas.DataFrame): DataFrame with 'Close' price
        test_size (float): Proportion of data to use for testing
        window_size (int): Number of previous days to use for features
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # Create a copy of the dataframe
    data = df.copy()
    
    # Scale the 'Close' price
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_price_scaled = scaler.fit_transform(data[['Close']])
    data['Close_Scaled'] = close_price_scaled.flatten()
    
    # Create lag features
    for i in range(1, window_size + 1):
        data[f'Lag_{i}'] = data['Close_Scaled'].shift(i)
    
    # Drop rows with NaN values (from lag creation)
    data = data.dropna()
    
    # Create time index feature
    data['Time_Index'] = np.arange(len(data))
    
    # Create features (X) and target (y)
    lag_columns = [f'Lag_{i}' for i in range(1, window_size + 1)]
    X = data[['Time_Index'] + lag_columns]
    y = data['Close_Scaled']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    return X_train, X_test, y_train, y_test, scaler
