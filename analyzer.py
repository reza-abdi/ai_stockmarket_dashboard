import warnings

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

class StockAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def fetch_stock_data(self, symbol, period="1y"):
        """Fetch stock data with error handling"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            info = stock.info
            return data, info
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None, None
    
    def calculate_technical_indicators(self, data):
        """Calculate comprehensive technical indicators using pure pandas/numpy"""
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price-based indicators
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Price_Change'] = df['Close'] - df['Open']
        df['Price_Change_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        
        # Volatility (Average True Range approximation)
        df['High_Low'] = df['High'] - df['Low']
        df['High_Close'] = np.abs(df['High'] - df['Close'].shift())
        df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift())
        df['True_Range'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
        df['ATR'] = df['True_Range'].rolling(window=14).mean()
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        return df
    
    def prepare_ml_features(self, data):
        """Prepare features for machine learning"""
        df = data.copy()
        
        # Returns and momentum
        df['Returns'] = df['Close'].pct_change()
        df['Returns_5d'] = df['Close'].pct_change(5)
        df['Returns_10d'] = df['Close'].pct_change(10)
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Returns_lag_{lag}'] = df['Returns'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20, 50]:
            df[f'Close_mean_{window}'] = df['Close'].rolling(window).mean()
            df[f'Close_std_{window}'] = df['Close'].rolling(window).std()
            df[f'Volume_mean_{window}'] = df['Volume'].rolling(window).mean()
            df[f'High_mean_{window}'] = df['High'].rolling(window).mean()
            df[f'Low_mean_{window}'] = df['Low'].rolling(window).mean()
        
        # Price position relative to moving averages
        df['Price_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20'] * 100
        df['Price_vs_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50'] * 100
        
        # Volatility features
        df['Price_volatility_10d'] = df['Returns'].rolling(10).std()
        df['Price_volatility_20d'] = df['Returns'].rolling(20).std()
        
        return df
    
    def train_prediction_model(self, data):
        """Train ML model for price prediction"""
        df = self.prepare_ml_features(data)
        df = df.dropna()
        
        if len(df) < 100:  # Need minimum data
            return None
        
        # Features for prediction (excluding target-related columns)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 
                       'Returns', 'Returns_5d', 'Returns_10d']
        feature_cols = [col for col in df.columns if not any(exc in col for exc in exclude_cols)]
        feature_cols = [col for col in feature_cols if 'lag' in col or 'mean' in col or 
                       'std' in col or col in ['RSI', 'MACD', 'Price_vs_SMA20', 'Price_vs_SMA50', 
                                              'Price_volatility_10d', 'Price_volatility_20d', 'ATR']]
        
        if len(feature_cols) < 5:
            return None
        
        X = df[feature_cols].fillna(method='ffill').fillna(method='bfill')
        y = df['Close'].shift(-1)  # Predict next day's close
        
        # Remove last row (no target) and any remaining NaN
        X = X[:-1]
        y = y[:-1]
        
        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 50:
            return None
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate accuracy
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': dict(zip(feature_cols, self.model.feature_importances_)),
            'last_features': X.iloc[-1:],
            'feature_cols': feature_cols
        }
    
    def predict_next_price(self, model_info):
        """Predict next trading day price"""
        if model_info is None:
            return None
        
        last_features_scaled = self.scaler.transform(model_info['last_features'])
        prediction = self.model.predict(last_features_scaled)[0]
        
        return prediction
    
    def generate_market_analysis(self, data, info, symbol):
        """Generate AI-powered market analysis"""
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        # Price movement
        price_change = latest['Close'] - prev['Close']
        price_change_pct = (price_change / prev['Close']) * 100
        
        # Technical analysis
        rsi = latest.get('RSI', 50)
        sma_20 = latest.get('SMA_20', latest['Close'])
        sma_50 = latest.get('SMA_50', latest['Close'])
        bb_upper = latest.get('BB_upper', latest['Close'])
        bb_lower = latest.get('BB_lower', latest['Close'])
        
        # Volume analysis
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        volume_ratio = latest['Volume'] / avg_volume if avg_volume > 0 else 1
        
        # MACD analysis
        macd = latest.get('MACD', 0)
        macd_signal = latest.get('MACD_signal', 0)
        
        # Generate analysis
        analysis = []
        
        # Price trend
        if price_change_pct > 3:
            analysis.append(f"🚀 {symbol} shows exceptional bullish momentum with a {price_change_pct:.2f}% surge")
        elif price_change_pct > 1:
            analysis.append(f"🟢 {symbol} demonstrates strong upward movement (+{price_change_pct:.2f}%)")
        elif price_change_pct > 0:
            analysis.append(f"🟡 {symbol} shows modest gains (+{price_change_pct:.2f}%)")
        elif price_change_pct > -1:
            analysis.append(f"🟡 {symbol} experiences slight decline ({price_change_pct:.2f}%)")
        elif price_change_pct > -3:
            analysis.append(f"🔴 {symbol} shows moderate bearish pressure ({price_change_pct:.2f}%)")
        else:
            analysis.append(f"🔻 {symbol} faces significant selling pressure ({price_change_pct:.2f}%)")
        
        # RSI analysis
        if rsi > 80:
            analysis.append(f"🚨 RSI at {rsi:.1f} indicates severely overbought conditions - potential reversal ahead")
        elif rsi > 70:
            analysis.append(f"⚠️ RSI at {rsi:.1f} shows overbought territory - exercise caution")
        elif rsi < 20:
            analysis.append(f"🛒 RSI at {rsi:.1f} signals severely oversold - strong buying opportunity")
        elif rsi < 30:
            analysis.append(f"💡 RSI at {rsi:.1f} suggests oversold conditions - potential buying opportunity")
        elif 40 <= rsi <= 60:
            analysis.append(f"⚖️ RSI at {rsi:.1f} indicates balanced momentum")
        else:
            analysis.append(f"📊 RSI at {rsi:.1f} shows {('bullish' if rsi > 50 else 'bearish')} bias")
        
        # Moving average analysis
        if latest['Close'] > sma_20 > sma_50:
            analysis.append("📈 Strong bullish alignment - price above both 20 and 50-day MAs")
        elif latest['Close'] < sma_20 < sma_50:
            analysis.append("📉 Bearish trend confirmed - price below key moving averages")
        elif latest['Close'] > sma_20 and sma_20 < sma_50:
            analysis.append("🔄 Mixed signals - short-term bullish but longer-term bearish")
        else:
            analysis.append("➡️ Consolidation phase - awaiting directional breakout")
        
        # Bollinger Bands analysis
        if latest['Close'] > bb_upper:
            analysis.append("📊 Price trading above upper Bollinger Band - potential overbought")
        elif latest['Close'] < bb_lower:
            analysis.append("📊 Price near lower Bollinger Band - potential oversold bounce")
        
        # MACD analysis
        if macd > macd_signal and macd > 0:
            analysis.append("⚡ MACD shows strong bullish momentum")
        elif macd < macd_signal and macd < 0:
            analysis.append("⚡ MACD indicates bearish momentum")
        elif macd > macd_signal:
            analysis.append("⚡ MACD bullish crossover - momentum improving")
        else:
            analysis.append("⚡ MACD bearish crossover - momentum weakening")
        
        # Volume analysis
        if volume_ratio > 2:
            analysis.append("🔥 Exceptional volume surge confirms strong conviction")
        elif volume_ratio > 1.5:
            analysis.append("📊 High volume validates price movement")
        elif volume_ratio < 0.5:
            analysis.append("📊 Below-average volume suggests weak conviction")
        else:
            analysis.append("📊 Normal volume levels")
        
        # Market cap context
        market_cap = info.get('marketCap', 0)
        if market_cap:
            if market_cap > 200e9:  # > 200B
                analysis.append("🏢 Large-cap stability with lower volatility expected")
            elif market_cap > 10e9:  # > 10B
                analysis.append("🏢 Mid-cap stock with balanced growth-stability profile")
            else:
                analysis.append("🏢 Small-cap stock with higher growth potential and volatility")
        
        return analysis