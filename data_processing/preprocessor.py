"""
Data preprocessing module for financial data.
Handles data cleaning, normalization, and standardization.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay

# Set up logging
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Data preprocessing for financial data."""
    
    def __init__(self):
        """Initialize preprocessor."""
        pass
    
    def clean_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean stock price data.
        
        Args:
            df: DataFrame with stock price data
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        try:
            # Convert index to datetime if it's not already
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Make column names lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]
            
            # Sort by date
            df = df.sort_index()
            
            # Handle missing values
            if df['close'].isna().any():
                # Forward fill for missing closing prices
                df['close'] = df['close'].fillna(method='ffill')
            
            if df['open'].isna().any():
                # Use previous close for missing open prices
                df['open'] = df['open'].fillna(df['close'].shift(1))
            
            if df['high'].isna().any():
                # Use max of open and close for missing high prices
                df['high'] = df.apply(
                    lambda row: max(row['open'], row['close']) if np.isnan(row['high']) 
                    else row['high'], 
                    axis=1
                )
            
            if df['low'].isna().any():
                # Use min of open and close for missing low prices
                df['low'] = df.apply(
                    lambda row: min(row['open'], row['close']) if np.isnan(row['low']) 
                    else row['low'], 
                    axis=1
                )
            
            if df['volume'].isna().any():
                # Use median volume for missing volume
                df['volume'] = df['volume'].fillna(df['volume'].median())
            
            # Check for outliers
            for col in ['open', 'high', 'low', 'close']:
                # Calculate z-scores
                z_scores = abs((df[col] - df[col].mean()) / df[col].std())
                # Replace extreme outliers (z-score > 3) with previous values
                outliers = z_scores > 3
                if outliers.any():
                    logger.warning(f"Found {outliers.sum()} outliers in {col}")
                    df.loc[outliers, col] = df[col].shift(1)
            
            return df
        
        except Exception as e:
            logger.error(f"Error cleaning stock data: {e}")
            return df
    
    def normalize_data(self, df: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """
        Normalize data to a standard scale.
        
        Args:
            df: DataFrame to normalize
            method: Normalization method ('minmax', 'zscore', 'log')
            
        Returns:
            Normalized DataFrame
        """
        if df.empty:
            return df
        
        try:
            # Create a copy to avoid modifying the original
            normalized_df = df.copy()
            
            # Columns to normalize (numeric columns only)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if method == 'minmax':
                # Min-max normalization (scale to 0-1)
                for col in numeric_cols:
                    min_val = normalized_df[col].min()
                    max_val = normalized_df[col].max()
                    if max_val > min_val:  # Avoid division by zero
                        normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
            
            elif method == 'zscore':
                # Z-score normalization (mean=0, std=1)
                for col in numeric_cols:
                    mean_val = normalized_df[col].mean()
                    std_val = normalized_df[col].std()
                    if std_val > 0:  # Avoid division by zero
                        normalized_df[col] = (normalized_df[col] - mean_val) / std_val
            
            elif method == 'log':
                # Log transformation (for positive skewed data)
                for col in numeric_cols:
                    # Add small constant to avoid log(0)
                    normalized_df[col] = np.log(normalized_df[col] + 1e-8)
            
            return normalized_df
        
        except Exception as e:
            logger.error(f"Error normalizing data: {e}")
            return df
    
    def fill_missing_dates(self, df: pd.DataFrame, freq: str = 'B') -> pd.DataFrame:
        """
        Fill missing dates in time series data.
        
        Args:
            df: DataFrame with time series data
            freq: Frequency ('B' for business days, 'D' for calendar days)
            
        Returns:
            DataFrame with complete date range
        """
        if df.empty:
            return df
        
        try:
            # Convert index to datetime if it's not already
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Create complete date range
            start_date = df.index.min()
            end_date = df.index.max()
            date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
            
            # Reindex with new date range
            df_reindexed = df.reindex(date_range)
            
            # Forward fill for missing dates
            df_reindexed = df_reindexed.fillna(method='ffill')
            
            return df_reindexed
        
        except Exception as e:
            logger.error(f"Error filling missing dates: {e}")
            return df
    
    def resample_data(self, df: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
        """
        Resample time series data to a different frequency.
        
        Args:
            df: DataFrame with time series data
            freq: Target frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
            
        Returns:
            Resampled DataFrame
        """
        if df.empty:
            return df
        
        try:
            # Convert index to datetime if it's not already
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Resample the data
            resampled = df.resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            return resampled
        
        except Exception as e:
            logger.error(f"Error resampling data: {e}")
            return df
    
    def prepare_stock_data_for_analysis(
        self, df: pd.DataFrame, fill_dates: bool = True
    ) -> pd.DataFrame:
        """
        Complete preparation of stock data for analysis.
        
        Args:
            df: DataFrame with stock data
            fill_dates: Whether to fill missing dates
            
        Returns:
            Prepared DataFrame
        """
        if df.empty:
            return df
        
        try:
            # Clean the data
            df = self.clean_stock_data(df)
            
            # Fill missing dates if requested
            if fill_dates:
                df = self.fill_missing_dates(df)
            
            # Add 'symbol' column if it doesn't exist
            if 'symbol' not in df.columns:
                df['symbol'] = df.get('ticker', None)
            
            # Convert to datetime if it's not already
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Sort by date
            df = df.sort_index()
            
            return df
        
        except Exception as e:
            logger.error(f"Error preparing stock data: {e}")
            return df
    
    def prepare_news_data(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare news data for analysis.
        
        Args:
            news_items: List of news item dictionaries
            
        Returns:
            Prepared news items
        """
        if not news_items:
            return []
        
        try:
            prepared_items = []
            
            for item in news_items:
                # Create a copy of the item
                prepared_item = item.copy()
                
                # Ensure published_at is datetime
                if isinstance(prepared_item.get('published_at'), str):
                    prepared_item['published_at'] = pd.to_datetime(prepared_item['published_at'])
                
                # Truncate overly long summaries
                if prepared_item.get('summary') and len(prepared_item['summary']) > 500:
                    prepared_item['summary'] = prepared_item['summary'][:497] + '...'
                
                # Remove any HTML tags from title and summary
                if prepared_item.get('title'):
                    prepared_item['title'] = self._strip_html(prepared_item['title'])
                
                if prepared_item.get('summary'):
                    prepared_item['summary'] = self._strip_html(prepared_item['summary'])
                
                prepared_items.append(prepared_item)
            
            # Sort by published date (newest first)
            prepared_items.sort(key=lambda x: x.get('published_at', datetime.now()), reverse=True)
            
            return prepared_items
        
        except Exception as e:
            logger.error(f"Error preparing news data: {e}")
            return news_items
    
    def _strip_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        import re
        return re.sub(r'<[^>]+>', '', text)
