import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(actual, predicted):
    """
    Calculate performance metrics for model evaluation.
    
    Args:
        actual (numpy.ndarray): Actual prices
        predicted (numpy.ndarray): Predicted prices
        
    Returns:
        tuple: (MAE, MSE, RMSE, MAPE)
    """
    # Mean Absolute Error
    mae = mean_absolute_error(actual, predicted)
    
    # Mean Squared Error
    mse = mean_squared_error(actual, predicted)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return mae, mse, rmse, mape

def get_default_start_date():
    """
    Returns a default start date (1 year ago from today).
    
    Returns:
        datetime.date: Default start date
    """
    return (datetime.now() - timedelta(days=365)).date()

def get_default_end_date():
    """
    Returns a default end date (today).
    
    Returns:
        datetime.date: Default end date
    """
    return datetime.now().date()
