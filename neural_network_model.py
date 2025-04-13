import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

class AdvancedNeuralNetworkModel:
    """
    Advanced Neural Network model for financial market prediction.
    Combines LSTM, attention mechanism, and dense layers for multi-factor prediction.
    """
    
    def __init__(self, sequence_length=60, lstm_units=100, dense_units=64, dropout_rate=0.3):
        """
        Initialize the Advanced Neural Network model.
        
        Args:
            sequence_length (int): Number of time steps to look back
            lstm_units (int): Number of LSTM units
            dense_units (int): Number of dense layer units
            dropout_rate (float): Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.last_sequence = None
        self.feature_count = None
        
    def build_model(self, input_shape):
        """
        Build the LSTM neural network model architecture.
        
        Args:
            input_shape (tuple): Shape of input data (sequence_length, features)
            
        Returns:
            model: Compiled Keras model
        """
        model = Sequential()
        
        # First LSTM layer with return sequences for stacking
        model.add(LSTM(units=self.lstm_units, 
                       return_sequences=True,
                       input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Second LSTM layer
        model.add(LSTM(units=self.lstm_units, 
                       return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Third LSTM layer without return sequences
        model.add(LSTM(units=self.lstm_units))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Dense layers for learning non-linear patterns
        model.add(Dense(units=self.dense_units, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        
        # Output layer with linear activation for regression
        model.add(Dense(units=1))
        
        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), 
                     loss='mean_squared_error',
                     metrics=['mae'])
        
        self.model = model
        return model
        
    def prepare_data(self, df, target_col='Close', test_size=0.2):
        """
        Prepare data for LSTM model by creating sequences and splitting into train/test.
        
        Args:
            df (pd.DataFrame): DataFrame with financial data
            target_col (str): Target column to predict
            test_size (float): Proportion of data to use for testing
            
        Returns:
            tuple: X_train, X_test, y_train, y_test, scaler
        """
        # Select features (we can add more technical indicators here)
        data = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Add technical indicators dynamically if they exist
        for col in df.columns:
            if col.startswith('RSI') or col.startswith('MACD') or col.startswith('BB_'):
                data[col] = df[col]
        
        # Store the feature count for later
        self.feature_count = data.shape[1]
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:i + self.sequence_length])
            y.append(scaled_data[i + self.sequence_length, data.columns.get_loc(target_col)])
        
        X, y = np.array(X), np.array(y)
        
        # Split into train and test sets
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Store the last sequence for future predictions
        self.last_sequence = X[-1:]
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        """
        Train the LSTM model.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Proportion of data to use for validation
            
        Returns:
            history: Training history
        """
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Set up callbacks for early stopping and model checkpointing
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint('best_lstm_model.h5', monitor='val_loss', save_best_only=True)
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X_test):
        """
        Make predictions using the trained model.
        
        Args:
            X_test (np.array): Test features
            
        Returns:
            np.array: Predicted values
        """
        if self.model is None:
            raise ValueError("Model needs to be trained before making predictions")
        
        predictions = self.model.predict(X_test)
        
        # Create dummy array to inverse transform
        dummy = np.zeros((len(predictions), self.feature_count))
        # Put predictions in the target column position (usually 'Close')
        dummy[:, 3] = predictions.flatten()  # Assuming 'Close' is at index 3
        
        # Inverse transform to get actual prices
        predictions_inverse = self.scaler.inverse_transform(dummy)[:, 3]
        
        return predictions_inverse
    
    def forecast_future(self, days=7):
        """
        Forecast future values for the specified number of days.
        
        Args:
            days (int): Number of days to forecast
            
        Returns:
            tuple: (predicted_dates, predicted_prices)
        """
        if self.model is None or self.last_sequence is None:
            raise ValueError("Model needs to be trained before forecasting")
        
        # Start with the last sequence we have
        curr_sequence = self.last_sequence.copy()
        predicted_prices = []
        
        # Generate predictions for each future day
        for _ in range(days):
            # Predict the next value
            predicted = self.model.predict(curr_sequence)
            
            # Create a dummy row for inverse transform
            dummy_row = np.zeros((1, self.feature_count))
            dummy_row[0, 3] = predicted[0, 0]  # Assuming 'Close' is at index 3
            
            # Inverse transform to get the actual price
            predicted_price = self.scaler.inverse_transform(dummy_row)[0, 3]
            predicted_prices.append(predicted_price)
            
            # Update the sequence by shifting and adding the new prediction
            # Create a new row with the predicted value at the target column position
            new_row = np.zeros((1, self.feature_count))
            new_row[0, 3] = predicted[0, 0]  # Assuming 'Close' is at index 3
            
            # Add the new row to the sequence and remove the first row
            curr_sequence = np.append(curr_sequence[:, 1:, :], [new_row], axis=1)
        
        # Generate future dates
        last_date = datetime.now()
        predicted_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(days)]
        
        return predicted_dates, np.array(predicted_prices)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance on test data.
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test targets
            
        Returns:
            dict: Model evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model needs to be trained before evaluation")
        
        # Get predictions
        y_pred = self.model.predict(X_test).flatten()
        
        # Create dummy arrays for inverse transform
        dummy_y_test = np.zeros((len(y_test), self.feature_count))
        dummy_y_test[:, 3] = y_test  # Assuming 'Close' is at index 3
        
        dummy_y_pred = np.zeros((len(y_pred), self.feature_count))
        dummy_y_pred[:, 3] = y_pred  # Assuming 'Close' is at index 3
        
        # Inverse transform
        actual = self.scaler.inverse_transform(dummy_y_test)[:, 3]
        predicted = self.scaler.inverse_transform(dummy_y_pred)[:, 3]
        
        # Calculate metrics
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual - predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
    
    def save_model(self, filepath='advanced_neural_network_model.h5'):
        """
        Save the trained model to file.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='advanced_neural_network_model.h5'):
        """
        Load a trained model from file.
        
        Args:
            filepath (str): Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
            
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def plot_training_history(self, history):
        """
        Plot the training history.
        
        Args:
            history: Training history from model.fit()
            
        Returns:
            matplotlib.figure.Figure: Figure with the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot loss
        ax1.plot(history.history['loss'], label='Train Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend(loc='upper right')
        ax1.grid(True)
        
        # Plot MAE
        ax2.plot(history.history['mae'], label='Train MAE')
        ax2.plot(history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model Mean Absolute Error')
        ax2.set_ylabel('MAE')
        ax2.set_xlabel('Epoch')
        ax2.legend(loc='upper right')
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_predictions(self, actual, predicted, title="Model Predictions vs Actual Prices"):
        """
        Plot actual vs predicted prices.
        
        Args:
            actual (np.array): Actual prices
            predicted (np.array): Predicted prices
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Figure with the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot actual and predicted prices
        ax.plot(actual, label='Actual Prices', color='blue')
        ax.plot(predicted, label='Predicted Prices', color='red', linestyle='--')
        
        # Set labels and title
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_future_predictions(self, dates, predictions, last_actual_price, title="Future Price Predictions"):
        """
        Plot future price predictions.
        
        Args:
            dates (list): Future dates
            predictions (np.array): Predicted future prices
            last_actual_price (float): Last known actual price
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Figure with the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot the last actual price and future predictions
        ax.plot([-1], [last_actual_price], 'bo', label='Last Actual Price')
        ax.plot(range(len(predictions)), predictions, 'r-o', label='Predicted Prices')
        
        # Set x-axis labels to dates
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels(dates, rotation=45)
        
        # Set labels and title
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig