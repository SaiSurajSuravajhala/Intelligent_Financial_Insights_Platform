"""
Feature engineering module for financial data analysis.
Computes various financial indicators and metrics.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

import pandas as pd
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering for financial data."""
    
    def __init__(self):
        """Initialize feature engineer."""
        pass
    
    def add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic features to stock data.
        
        Args:
            df: DataFrame with stock price data (must have open, high, low, close, volume)
            
        Returns:
            DataFrame with added features
        """
        if df.empty:
            return df
        
        try:
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Daily returns
            result_df['return'] = result_df['close'].pct_change()
            
            # Log returns
            result_df['log_return'] = np.log(result_df['close'] / result_df['close'].shift(1))
            
            # Trading range
            result_df['range'] = result_df['high'] - result_df['low']
            
            # Gap (difference between current open and previous close)
            result_df['gap'] = result_df['open'] - result_df['close'].shift(1)
            
            # Body (difference between open and close)
            result_df['body'] = result_df['close'] - result_df['open']
            
            # Upper wick (difference between high and max of open/close)
            result_df['upper_wick'] = result_df['high'] - result_df[['open', 'close']].max(axis=1)
            
            # Lower wick (difference between min of open/close and low)
            result_df['lower_wick'] = result_df[['open', 'close']].min(axis=1) - result_df['low']
            
            # Volume change
            result_df['volume_change'] = result_df['volume'].pct_change()
            
            # Price-to-volume ratio
            result_df['price_volume_ratio'] = result_df['close'] / result_df['volume']
            
            return result_df
        
        except Exception as e:
            logger.error(f"Error adding basic features: {e}")
            return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to stock data.
        
        Args:
            df: DataFrame with stock price data (must have open, high, low, close, volume)
            
        Returns:
            DataFrame with added technical indicators
        """
        if df.empty:
            return df
        
        try:
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Moving averages
            for ma_period in [5, 10, 20, 50, 200]:
                result_df[f'ma_{ma_period}'] = result_df['close'].rolling(window=ma_period).mean()
            
            # Exponential moving averages
            for ema_period in [5, 10, 20, 50, 200]:
                result_df[f'ema_{ema_period}'] = result_df['close'].ewm(span=ema_period, adjust=False).mean()
            
            # Bollinger Bands (20-day SMA +/- 2 standard deviations)
            bb_period = 20
            result_df['bb_middle'] = result_df['close'].rolling(window=bb_period).mean()
            result_df['bb_std'] = result_df['close'].rolling(window=bb_period).std()
            result_df['bb_upper'] = result_df['bb_middle'] + 2 * result_df['bb_std']
            result_df['bb_lower'] = result_df['bb_middle'] - 2 * result_df['bb_std']
            result_df['bb_width'] = (result_df['bb_upper'] - result_df['bb_lower']) / result_df['bb_middle']
            
            # Relative Strength Index (RSI)
            # Calculate gains and losses
            delta = result_df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Calculate average gain and loss
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            result_df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Moving Average Convergence Divergence (MACD)
            result_df['macd'] = result_df['close'].ewm(span=12, adjust=False).mean() - \
                                result_df['close'].ewm(span=26, adjust=False).mean()
            result_df['macd_signal'] = result_df['macd'].ewm(span=9, adjust=False).mean()
            result_df['macd_histogram'] = result_df['macd'] - result_df['macd_signal']
            
            # Average True Range (ATR)
            tr1 = result_df['high'] - result_df['low']
            tr2 = abs(result_df['high'] - result_df['close'].shift(1))
            tr3 = abs(result_df['low'] - result_df['close'].shift(1))
            result_df['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            result_df['atr_14'] = result_df['true_range'].rolling(window=14).mean()
            
            # Stochastic Oscillator
            low_14 = result_df['low'].rolling(window=14).min()
            high_14 = result_df['high'].rolling(window=14).max()
            result_df['stoch_k'] = 100 * ((result_df['close'] - low_14) / (high_14 - low_14))
            result_df['stoch_d'] = result_df['stoch_k'].rolling(window=3).mean()
            
            # On-Balance Volume (OBV)
            obv = [0]
            for i in range(1, len(result_df)):
                if result_df['close'].iloc[i] > result_df['close'].iloc[i-1]:
                    obv.append(obv[-1] + result_df['volume'].iloc[i])
                elif result_df['close'].iloc[i] < result_df['close'].iloc[i-1]:
                    obv.append(obv[-1] - result_df['volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            result_df['obv'] = obv
            
            # Rate of Change (ROC)
            result_df['roc_10'] = result_df['close'].pct_change(periods=10) * 100
            
            # Commodity Channel Index (CCI)
            typical_price = (result_df['high'] + result_df['low'] + result_df['close']) / 3
            mean_deviation = abs(typical_price - typical_price.rolling(window=20).mean()).rolling(window=20).mean()
            result_df['cci_20'] = (typical_price - typical_price.rolling(window=20).mean()) / (0.015 * mean_deviation)
            
            # Williams %R
            result_df['williams_r_14'] = -100 * (high_14 - result_df['close']) / (high_14 - low_14)
            
            return result_df
        
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility indicators to stock data.
        
        Args:
            df: DataFrame with stock price data (must have close)
            
        Returns:
            DataFrame with added volatility indicators
        """
        if df.empty:
            return df
        
        try:
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Daily log returns
            log_returns = np.log(result_df['close'] / result_df['close'].shift(1))
            
            # Historical volatility (standard deviation of log returns)
            for window in [5, 10, 20, 30]:
                # Annualized volatility (assuming 252 trading days per year)
                result_df[f'volatility_{window}d'] = log_returns.rolling(window=window).std() * np.sqrt(252)
            
            # Parkinson volatility estimator (uses high-low range)
            hl_ratio = np.log(result_df['high'] / result_df['low'])
            result_df['parkinson_vol_10d'] = np.sqrt(
                1 / (4 * np.log(2)) * hl_ratio.pow(2).rolling(window=10).mean() * 252
            )
            
            # GARCH-like volatility (simple approximation)
            # Using exponentially weighted moving standard deviation
            result_df['garch_vol'] = log_returns.ewm(span=20).std() * np.sqrt(252)
            
            # Volatility ratio (short-term to long-term)
            result_df['vol_ratio'] = result_df['volatility_10d'] / result_df['volatility_30d']
            
            return result_df
        
        except Exception as e:
            logger.error(f"Error adding volatility indicators: {e}")
            return df
    
    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend indicators to stock data.
        
        Args:
            df: DataFrame with stock price data (must have close)
            
        Returns:
            DataFrame with added trend indicators
        """
        if df.empty:
            return df
        
        try:
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Simple trend indicators
            
            # Price position relative to moving averages
            for ma_period in [50, 200]:
                ma_col = f'ma_{ma_period}'
                if ma_col not in result_df.columns:
                    result_df[ma_col] = result_df['close'].rolling(window=ma_period).mean()
                # Percentage above/below moving average
                result_df[f'pct_diff_ma_{ma_period}'] = (result_df['close'] / result_df[ma_col] - 1) * 100
            
            # Moving average crossovers
            # Golden Cross / Death Cross (50-day vs 200-day MA)
            result_df['ma_50_200_ratio'] = result_df['ma_50'] / result_df['ma_200']
            result_df['golden_cross'] = (result_df['ma_50_200_ratio'] > 1) & (result_df['ma_50_200_ratio'].shift(1) <= 1)
            result_df['death_cross'] = (result_df['ma_50_200_ratio'] < 1) & (result_df['ma_50_200_ratio'].shift(1) >= 1)
            
            # ADX (Average Directional Index)
            # Simplified implementation
            high_diff = result_df['high'] - result_df['high'].shift(1)
            low_diff = result_df['low'].shift(1) - result_df['low']
            
            plus_dm = high_diff.copy()
            plus_dm[~((high_diff > 0) & (high_diff > low_diff))] = 0
            
            minus_dm = low_diff.copy()
            minus_dm[~((low_diff > 0) & (low_diff > high_diff))] = 0
            
            atr = result_df['true_range'].rolling(window=14).mean() if 'true_range' in result_df.columns else None
            if atr is None:
                # Calculate true range if not already present
                tr1 = result_df['high'] - result_df['low']
                tr2 = abs(result_df['high'] - result_df['close'].shift(1))
                tr3 = abs(result_df['low'] - result_df['close'].shift(1))
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = true_range.rolling(window=14).mean()
            
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            result_df['adx'] = dx.rolling(window=14).mean()
            result_df['plus_di'] = plus_di
            result_df['minus_di'] = minus_di
            
            # Aroon Indicator
            period = 25
            result_df['aroon_up'] = 100 * (period - result_df['high'].rolling(window=period+1).apply(lambda x: x.argmax(), raw=True)) / period
            result_df['aroon_down'] = 100 * (period - result_df['low'].rolling(window=period+1).apply(lambda x: x.argmin(), raw=True)) / period
            result_df['aroon_oscillator'] = result_df['aroon_up'] - result_df['aroon_down']
            
            return result_df
        
        except Exception as e:
            logger.error(f"Error adding trend indicators: {e}")
            return df
    
    def add_pattern_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add candlestick pattern indicators to stock data.
        
        Args:
            df: DataFrame with stock price data (must have open, high, low, close)
            
        Returns:
            DataFrame with added pattern indicators
        """
        if df.empty:
            return df
        
        try:
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Helper values
            body = result_df['close'] - result_df['open']
            body_abs = abs(body)
            range_day = result_df['high'] - result_df['low']
            
            # Doji pattern (open and close are almost equal)
            doji_threshold = 0.1  # 10% of range
            result_df['doji'] = body_abs < (range_day * doji_threshold)
            
            # Hammer pattern (small body near top, long lower shadow, small upper shadow)
            result_df['hammer'] = (
                (body_abs < (range_day * 0.3)) &  # Small body
                ((result_df[['open', 'close']].min(axis=1) - result_df['low']) > (2 * body_abs)) &  # Long lower shadow
                ((result_df['high'] - result_df[['open', 'close']].max(axis=1)) < (0.3 * body_abs))  # Small upper shadow
            )
            
            # Shooting star pattern (small body near bottom, long upper shadow, small lower shadow)
            result_df['shooting_star'] = (
                (body_abs < (range_day * 0.3)) &  # Small body
                ((result_df['high'] - result_df[['open', 'close']].max(axis=1)) > (2 * body_abs)) &  # Long upper shadow
                ((result_df[['open', 'close']].min(axis=1) - result_df['low']) < (0.3 * body_abs))  # Small lower shadow
            )
            
            # Bullish engulfing pattern
            result_df['bullish_engulfing'] = (
                (result_df['close'] > result_df['open']) &  # Current day is bullish
                (result_df['open'].shift(1) > result_df['close'].shift(1)) &  # Previous day is bearish
                (result_df['open'] < result_df['close'].shift(1)) &  # Current open is lower than previous close
                (result_df['close'] > result_df['open'].shift(1))  # Current close is higher than previous open
            )
            
            # Bearish engulfing pattern
            result_df['bearish_engulfing'] = (
                (result_df['close'] < result_df['open']) &  # Current day is bearish
                (result_df['open'].shift(1) < result_df['close'].shift(1)) &  # Previous day is bullish
                (result_df['open'] > result_df['close'].shift(1)) &  # Current open is higher than previous close
                (result_df['close'] < result_df['open'].shift(1))  # Current close is lower than previous open
            )
            
            # Morning star pattern (simplified)
            result_df['morning_star'] = (
                (result_df['close'].shift(2) < result_df['open'].shift(2)) &  # First day is bearish
                (body_abs.shift(1) < (range_day.shift(1) * 0.3)) &  # Second day has small body
                (result_df['close'] > result_df['open']) &  # Third day is bullish
                (result_df['close'] > (result_df['open'].shift(2) + result_df['close'].shift(2)) / 2)  # Third day closes above midpoint of first day
            )
            
            # Evening star pattern (simplified)
            result_df['evening_star'] = (
                (result_df['close'].shift(2) > result_df['open'].shift(2)) &  # First day is bullish
                (body_abs.shift(1) < (range_day.shift(1) * 0.3)) &  # Second day has small body
                (result_df['close'] < result_df['open']) &  # Third day is bearish
                (result_df['close'] < (result_df['open'].shift(2) + result_df['close'].shift(2)) / 2)  # Third day closes below midpoint of first day
            )
            
            return result_df
        
        except Exception as e:
            logger.error(f"Error adding pattern indicators: {e}")
            return df
    
    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all features to stock data.
        
        Args:
            df: DataFrame with stock price data
            
        Returns:
            DataFrame with all features added
        """
        if df.empty:
            return df
        
        try:
            # Add features in sequence
            result_df = self.add_basic_features(df)
            result_df = self.add_technical_indicators(result_df)
            result_df = self.add_volatility_indicators(result_df)
            result_df = self.add_trend_indicators(result_df)
            result_df = self.add_pattern_indicators(result_df)
            
            return result_df
        
        except Exception as e:
            logger.error(f"Error adding all features: {e}")
            return df
    
    def calculate_financial_metrics(self, financial_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Calculate financial metrics from financial statements.
        
        Args:
            financial_data: Dictionary with financial statements
            
        Returns:
            Dictionary with calculated metrics
        """
        metrics = {}
        
        try:
            income_statement = financial_data.get('income_statement', pd.DataFrame())
            balance_sheet = financial_data.get('balance_sheet', pd.DataFrame())
            cash_flow = financial_data.get('cash_flow', pd.DataFrame())
            
            if income_statement.empty or balance_sheet.empty:
                return metrics
            
            # Convert columns to numeric where possible
            for df in [income_statement, balance_sheet, cash_flow]:
                if not df.empty:
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='ignore')
            
            # Get latest data
            latest_income = income_statement.iloc[:, 0] if not income_statement.empty else pd.Series()
            latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series()
            latest_cash_flow = cash_flow.iloc[:, 0] if not cash_flow.empty else pd.Series()
            
            # Profitability metrics
            if 'totalRevenue' in latest_income and 'netIncome' in latest_income:
                metrics['net_profit_margin'] = latest_income['netIncome'] / latest_income['totalRevenue']
            
            if 'grossProfit' in latest_income and 'totalRevenue' in latest_income:
                metrics['gross_margin'] = latest_income['grossProfit'] / latest_income['totalRevenue']
            
            # Efficiency metrics
            if 'totalRevenue' in latest_income and 'totalAssets' in latest_balance:
                metrics['asset_turnover'] = latest_income['totalRevenue'] / latest_balance['totalAssets']
            
            # Leverage metrics
            if 'totalAssets' in latest_balance and 'totalLiabilities' in latest_balance:
                metrics['debt_to_assets'] = latest_balance['totalLiabilities'] / latest_balance['totalAssets']
            
            if 'totalShareholderEquity' in latest_balance and 'totalLiabilities' in latest_balance:
                metrics['debt_to_equity'] = latest_balance['totalLiabilities'] / latest_balance['totalShareholderEquity']
            
            # Liquidity metrics
            if 'totalCurrentAssets' in latest_balance and 'totalCurrentLiabilities' in latest_balance:
                metrics['current_ratio'] = latest_balance['totalCurrentAssets'] / latest_balance['totalCurrentLiabilities']
            
            # Cash flow metrics
            if not latest_cash_flow.empty and 'operatingCashflow' in latest_cash_flow and 'netIncome' in latest_income:
                metrics['cash_flow_to_income'] = latest_cash_flow['operatingCashflow'] / latest_income['netIncome']
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error calculating financial metrics: {e}")
            return metrics
