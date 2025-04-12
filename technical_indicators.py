import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import streamlit as st

def add_technical_indicators(df):
    """
    Add technical indicators to a dataframe containing OHLCV data
    
    Args:
        df (pandas.DataFrame): DataFrame with historical price data (must contain 'Close' column)
        
    Returns:
        pandas.DataFrame: DataFrame with added technical indicators
    """
    if df is None or df.empty:
        return df
    
    # Make sure we have a copy to avoid modifying the original
    df_with_indicators = df.copy()
    
    # Add indicators using the 'ta' library
    try:
        # Ensure we have pandas Series (not numpy arrays) for all inputs
        # The ta library needs pandas Series with the rolling method
        if not isinstance(df['Close'], pd.Series):
            close_values = pd.Series(df['Close'].values.flatten() if hasattr(df['Close'], 'values') and df['Close'].values.ndim > 1 else df['Close'])
        else:
            close_values = df['Close']
            
        if not isinstance(df['High'], pd.Series):
            high_values = pd.Series(df['High'].values.flatten() if hasattr(df['High'], 'values') and df['High'].values.ndim > 1 else df['High'])
        else:
            high_values = df['High']
            
        if not isinstance(df['Low'], pd.Series):
            low_values = pd.Series(df['Low'].values.flatten() if hasattr(df['Low'], 'values') and df['Low'].values.ndim > 1 else df['Low'])
        else:
            low_values = df['Low']
            
        if not isinstance(df['Volume'], pd.Series):
            volume_values = pd.Series(df['Volume'].values.flatten() if hasattr(df['Volume'], 'values') and df['Volume'].values.ndim > 1 else df['Volume'])
        else:
            volume_values = df['Volume']
        
        # Moving averages
        df_with_indicators['SMA_20'] = ta.trend.sma_indicator(close_values, window=20)
        df_with_indicators['SMA_50'] = ta.trend.sma_indicator(close_values, window=50)
        df_with_indicators['EMA_20'] = ta.trend.ema_indicator(close_values, window=20)
        
        # RSI (Relative Strength Index)
        df_with_indicators['RSI'] = ta.momentum.rsi(close_values, window=14)
        
        # MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(close_values)
        df_with_indicators['MACD'] = macd.macd()
        df_with_indicators['MACD_Signal'] = macd.macd_signal()
        df_with_indicators['MACD_Hist'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close_values)
        df_with_indicators['BB_High'] = bollinger.bollinger_hband()
        df_with_indicators['BB_Low'] = bollinger.bollinger_lband()
        df_with_indicators['BB_Middle'] = bollinger.bollinger_mavg()
        df_with_indicators['BB_Width'] = bollinger.bollinger_wband()
        
        # ATR (Average True Range) - volatility indicator
        df_with_indicators['ATR'] = ta.volatility.average_true_range(
            high_values, low_values, close_values)
        
        # OBV (On-Balance Volume)
        df_with_indicators['OBV'] = ta.volume.on_balance_volume(close_values, volume_values)
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(high_values, low_values, close_values)
        df_with_indicators['Stoch_k'] = stoch.stoch()
        df_with_indicators['Stoch_d'] = stoch.stoch_signal()
        
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        # Return original dataframe if something goes wrong
        return df
    
    return df_with_indicators

def get_technical_signals(df_with_indicators):
    """
    Generate trading signals based on technical indicators
    
    Args:
        df_with_indicators (pandas.DataFrame): DataFrame with technical indicators
        
    Returns:
        dict: Dictionary of trading signals and their explanations
    """
    if df_with_indicators is None or df_with_indicators.empty:
        return {}
    
    signals = {}
    
    try:
        # Get the latest data point
        latest = df_with_indicators.iloc[-1]
        
        # Check if RSI is available (not NaN)
        if 'RSI' in latest and pd.notna(latest['RSI']):
            # RSI Signals
            if latest['RSI'] < 30:
                signals['RSI'] = {
                    'signal': 'BUY',
                    'value': f"{latest['RSI']:.2f}",
                    'explanation': 'RSI below 30 indicates oversold conditions'
                }
            elif latest['RSI'] > 70:
                signals['RSI'] = {
                    'signal': 'SELL',
                    'value': f"{latest['RSI']:.2f}",
                    'explanation': 'RSI above 70 indicates overbought conditions'
                }
            else:
                signals['RSI'] = {
                    'signal': 'NEUTRAL',
                    'value': f"{latest['RSI']:.2f}",
                    'explanation': 'RSI between 30-70 indicates neutral conditions'
                }
        else:
            # If RSI is not available, add a placeholder
            signals['RSI'] = {
                'signal': 'NEUTRAL',
                'value': 'N/A',
                'explanation': 'RSI data not available'
            }
        
        # Check if MACD is available
        if 'MACD' in latest and 'MACD_Signal' in latest and pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']):
            # MACD Signals
            if latest['MACD'] > latest['MACD_Signal']:
                signals['MACD'] = {
                    'signal': 'BUY',
                    'value': f"{latest['MACD']:.2f}",
                    'explanation': 'MACD above signal line suggests bullish momentum'
                }
            else:
                signals['MACD'] = {
                    'signal': 'SELL',
                    'value': f"{latest['MACD']:.2f}",
                    'explanation': 'MACD below signal line suggests bearish momentum'
                }
        else:
            signals['MACD'] = {
                'signal': 'NEUTRAL',
                'value': 'N/A',
                'explanation': 'MACD data not available'
            }
        
        # Check if SMA_50 is available
        if 'SMA_50' in latest and pd.notna(latest['SMA_50']):
            # Moving Average Signals
            if latest['Close'] > latest['SMA_50']:
                signals['SMA'] = {
                    'signal': 'BUY',
                    'value': f"{latest['SMA_50']:.2f}",
                    'explanation': 'Price above 50-day SMA indicates uptrend'
                }
            else:
                signals['SMA'] = {
                    'signal': 'SELL',
                    'value': f"{latest['SMA_50']:.2f}",
                    'explanation': 'Price below 50-day SMA indicates downtrend'
                }
        else:
            signals['SMA'] = {
                'signal': 'NEUTRAL',
                'value': 'N/A',
                'explanation': 'SMA data not available'
            }
        
        # Check if Bollinger Bands are available
        if 'BB_Low' in latest and 'BB_High' in latest and 'BB_Width' in latest and pd.notna(latest['BB_Low']) and pd.notna(latest['BB_High']) and pd.notna(latest['BB_Width']):
            # Bollinger Bands Signals
            if latest['Close'] < latest['BB_Low']:
                signals['Bollinger'] = {
                    'signal': 'BUY',
                    'value': f"{latest['BB_Width']:.2f}",
                    'explanation': 'Price below lower Bollinger Band suggests oversold conditions'
                }
            elif latest['Close'] > latest['BB_High']:
                signals['Bollinger'] = {
                    'signal': 'SELL',
                    'value': f"{latest['BB_Width']:.2f}",
                    'explanation': 'Price above upper Bollinger Band suggests overbought conditions'
                }
            else:
                signals['Bollinger'] = {
                    'signal': 'NEUTRAL',
                    'value': f"{latest['BB_Width']:.2f}",
                    'explanation': 'Price within Bollinger Bands suggests neutral conditions'
                }
        else:
            signals['Bollinger'] = {
                'signal': 'NEUTRAL',
                'value': 'N/A',
                'explanation': 'Bollinger Bands data not available'
            }
        
        # Check if Stochastic is available
        if 'Stoch_k' in latest and pd.notna(latest['Stoch_k']):
            # Stochastic Oscillator Signals
            if latest['Stoch_k'] < 20:
                signals['Stochastic'] = {
                    'signal': 'BUY',
                    'value': f"{latest['Stoch_k']:.2f}",
                    'explanation': 'Stochastic below 20 indicates oversold conditions'
                }
            elif latest['Stoch_k'] > 80:
                signals['Stochastic'] = {
                    'signal': 'SELL',
                    'value': f"{latest['Stoch_k']:.2f}",
                    'explanation': 'Stochastic above 80 indicates overbought conditions'
                }
            else:
                signals['Stochastic'] = {
                    'signal': 'NEUTRAL',
                    'value': f"{latest['Stoch_k']:.2f}",
                    'explanation': 'Stochastic between 20-80 indicates neutral conditions'
                }
        else:
            signals['Stochastic'] = {
                'signal': 'NEUTRAL',
                'value': 'N/A',
                'explanation': 'Stochastic data not available'
            }
            
        # Overall signal (simple majority voting)
        buy_count = sum(1 for s in signals.values() if s['signal'] == 'BUY')
        sell_count = sum(1 for s in signals.values() if s['signal'] == 'SELL')
        neutral_count = sum(1 for s in signals.values() if s['signal'] == 'NEUTRAL')
        
        if buy_count > sell_count and buy_count > neutral_count:
            signals['OVERALL'] = {
                'signal': 'BUY',
                'value': f"{buy_count}/{len(signals)}",
                'explanation': f'{buy_count} out of {len(signals)} indicators suggest BUY'
            }
        elif sell_count > buy_count and sell_count > neutral_count:
            signals['OVERALL'] = {
                'signal': 'SELL',
                'value': f"{sell_count}/{len(signals)}",
                'explanation': f'{sell_count} out of {len(signals)} indicators suggest SELL'
            }
        else:
            signals['OVERALL'] = {
                'signal': 'NEUTRAL',
                'value': f"{neutral_count}/{len(signals)}",
                'explanation': 'Indicators show mixed signals or neutrality'
            }
        
    except Exception as e:
        st.error(f"Error generating technical signals: {str(e)}")
        return {}
    
    return signals

def plot_indicators(df_with_indicators, ticker_symbol):
    """
    Create plots of key technical indicators
    
    Args:
        df_with_indicators (pandas.DataFrame): DataFrame with technical indicators
        ticker_symbol (str): Symbol of the stock/crypto
        
    Returns:
        tuple: Tuple of matplotlib figures for different indicator plots
    """
    if df_with_indicators is None or df_with_indicators.empty:
        return None, None
    
    try:
        # Create figure for price and moving averages
        fig1, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
        
        # Price and moving averages
        axes[0].plot(df_with_indicators.index, df_with_indicators['Close'], label='Close Price', color='blue')
        
        # Only plot indicators if they exist
        if 'SMA_20' in df_with_indicators.columns and not df_with_indicators['SMA_20'].isna().all():
            axes[0].plot(df_with_indicators.index, df_with_indicators['SMA_20'], label='SMA (20)', color='orange', alpha=0.7)
        
        if 'SMA_50' in df_with_indicators.columns and not df_with_indicators['SMA_50'].isna().all():
            axes[0].plot(df_with_indicators.index, df_with_indicators['SMA_50'], label='SMA (50)', color='green', alpha=0.7)
        
        if ('BB_High' in df_with_indicators.columns and 'BB_Low' in df_with_indicators.columns and
            not df_with_indicators['BB_High'].isna().all() and not df_with_indicators['BB_Low'].isna().all()):
            axes[0].plot(df_with_indicators.index, df_with_indicators['BB_High'], label='Bollinger High', color='red', linestyle='--', alpha=0.5)
            axes[0].plot(df_with_indicators.index, df_with_indicators['BB_Low'], label='Bollinger Low', color='red', linestyle='--', alpha=0.5)
            axes[0].fill_between(df_with_indicators.index, df_with_indicators['BB_High'], df_with_indicators['BB_Low'], color='red', alpha=0.1)
        
        axes[0].set_title(f'Price and Moving Averages for {ticker_symbol}')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Volume
        if 'Volume' in df_with_indicators.columns:
            axes[1].bar(df_with_indicators.index, df_with_indicators['Volume'], color='blue', alpha=0.5)
            axes[1].set_ylabel('Volume')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Check if there are any indicators to plot
        has_rsi = 'RSI' in df_with_indicators.columns and not df_with_indicators['RSI'].isna().all()
        has_macd = ('MACD' in df_with_indicators.columns and 'MACD_Signal' in df_with_indicators.columns and 
                    not df_with_indicators['MACD'].isna().all())
        has_stoch = ('Stoch_k' in df_with_indicators.columns and 'Stoch_d' in df_with_indicators.columns and 
                     not df_with_indicators['Stoch_k'].isna().all())
        
        if has_rsi or has_macd or has_stoch:
            # Create figure for RSI, MACD and Stochastic
            fig2, axes = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 1, 1]})
            
            # RSI
            if has_rsi:
                axes[0].plot(df_with_indicators.index, df_with_indicators['RSI'], color='purple')
                axes[0].axhline(y=70, color='red', linestyle='--', alpha=0.5)
                axes[0].axhline(y=30, color='green', linestyle='--', alpha=0.5)
                axes[0].fill_between(df_with_indicators.index, df_with_indicators['RSI'], 70, 
                                    where=(df_with_indicators['RSI'] >= 70), color='red', alpha=0.3)
                axes[0].fill_between(df_with_indicators.index, df_with_indicators['RSI'], 30, 
                                    where=(df_with_indicators['RSI'] <= 30), color='green', alpha=0.3)
                axes[0].set_title('RSI (Relative Strength Index)')
                axes[0].set_ylabel('RSI')
                axes[0].set_ylim(0, 100)
                axes[0].grid(True, alpha=0.3)
            else:
                axes[0].set_title('RSI - Data Not Available')
                axes[0].grid(True, alpha=0.3)
            
            # MACD
            if has_macd:
                axes[1].plot(df_with_indicators.index, df_with_indicators['MACD'], label='MACD', color='blue')
                axes[1].plot(df_with_indicators.index, df_with_indicators['MACD_Signal'], label='Signal', color='red')
                if 'MACD_Hist' in df_with_indicators.columns:
                    axes[1].bar(df_with_indicators.index, df_with_indicators['MACD_Hist'], label='Histogram', color='green', alpha=0.5)
                axes[1].set_title('MACD (Moving Average Convergence Divergence)')
                axes[1].set_ylabel('MACD')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            else:
                axes[1].set_title('MACD - Data Not Available')
                axes[1].grid(True, alpha=0.3)
            
            # Stochastic
            if has_stoch:
                axes[2].plot(df_with_indicators.index, df_with_indicators['Stoch_k'], label='%K', color='blue')
                axes[2].plot(df_with_indicators.index, df_with_indicators['Stoch_d'], label='%D', color='red')
                axes[2].axhline(y=80, color='red', linestyle='--', alpha=0.5)
                axes[2].axhline(y=20, color='green', linestyle='--', alpha=0.5)
                axes[2].fill_between(df_with_indicators.index, df_with_indicators['Stoch_k'], 80, 
                                    where=(df_with_indicators['Stoch_k'] >= 80), color='red', alpha=0.3)
                axes[2].fill_between(df_with_indicators.index, df_with_indicators['Stoch_k'], 20, 
                                    where=(df_with_indicators['Stoch_k'] <= 20), color='green', alpha=0.3)
                axes[2].set_title('Stochastic Oscillator')
                axes[2].set_ylabel('Stochastic')
                axes[2].set_ylim(0, 100)
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
            else:
                axes[2].set_title('Stochastic Oscillator - Data Not Available')
                axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            return fig1, fig2
        else:
            return fig1, None
        
    except Exception as e:
        st.error(f"Error plotting technical indicators: {str(e)}")
        return None, None

def display_technical_dashboard(df_with_indicators, ticker_symbol):
    """
    Display a dashboard of technical indicators and signals in a Streamlit app
    
    Args:
        df_with_indicators (pandas.DataFrame): DataFrame with technical indicators
        ticker_symbol (str): Symbol of the stock/crypto
    """
    if df_with_indicators is None or df_with_indicators.empty:
        st.warning("No data available for technical analysis")
        return
    
    try:
        # Get technical signals
        signals = get_technical_signals(df_with_indicators)
        
        if not signals:
            st.warning("Unable to generate technical signals")
            return
        
        # Display overall signal
        st.subheader("Technical Analysis Summary")
        
        overall = signals.get('OVERALL', {'signal': 'NEUTRAL', 'explanation': 'No clear signal'})
        signal_color = "gray"
        if overall['signal'] == 'BUY':
            signal_color = "green"
        elif overall['signal'] == 'SELL':
            signal_color = "red"
        
        st.markdown(f"### Overall Signal: <span style='color:{signal_color}'>{overall['signal']}</span>", unsafe_allow_html=True)
        st.markdown(f"*{overall['explanation']}*")
        
        # Display individual signals
        st.subheader("Individual Technical Indicators")
        
        # Create columns for indicators
        cols = st.columns(2)
        
        # List of indicators to display (excluding OVERALL)
        indicators = [k for k in signals.keys() if k != 'OVERALL']
        
        # Display each indicator in a column
        for i, indicator in enumerate(indicators):
            col_idx = i % 2
            with cols[col_idx]:
                signal = signals[indicator]
                signal_color = "gray"
                if signal['signal'] == 'BUY':
                    signal_color = "green"
                elif signal['signal'] == 'SELL':
                    signal_color = "red"
                
                st.markdown(f"**{indicator}**: <span style='color:{signal_color}'>{signal['signal']}</span> (Value: {signal['value']})", unsafe_allow_html=True)
                st.markdown(f"*{signal['explanation']}*")
                st.markdown("---")
        
        # Plot technical indicators
        fig1, fig2 = plot_indicators(df_with_indicators, ticker_symbol)
        
        if fig1:
            st.subheader("Technical Charts")
            st.pyplot(fig1)
            
            if fig2:
                st.pyplot(fig2)
            
        # Display technical data table (last few rows)
        with st.expander("View Technical Data"):
            st.dataframe(df_with_indicators.tail(10))
            
    except Exception as e:
        st.error(f"Error displaying technical dashboard: {str(e)}")