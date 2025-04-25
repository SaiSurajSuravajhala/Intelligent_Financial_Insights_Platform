"""
Helper utility functions for the Intelligent Financial Insights Platform.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import os
import re

# Set up logging
logger = logging.getLogger(__name__)

def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to directory
    """
    os.makedirs(directory_path, exist_ok=True)

def format_currency(value: float, precision: int = 2) -> str:
    """
    Format a number as currency.
    
    Args:
        value: Numeric value
        precision: Decimal precision
        
    Returns:
        Formatted currency string
    """
    if abs(value) >= 1e9:
        return f"${value / 1e9:.{precision}f}B"
    elif abs(value) >= 1e6:
        return f"${value / 1e6:.{precision}f}M"
    elif abs(value) >= 1e3:
        return f"${value / 1e3:.{precision}f}K"
    else:
        return f"${value:.{precision}f}"

def format_percentage(value: float, precision: int = 2) -> str:
    """
    Format a number as percentage.
    
    Args:
        value: Numeric value (0.01 = 1%)
        precision: Decimal precision
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{precision}f}%"

def format_large_number(value: float, precision: int = 2) -> str:
    """
    Format a large number with K, M, B suffixes.
    
    Args:
        value: Numeric value
        precision: Decimal precision
        
    Returns:
        Formatted number string
    """
    if abs(value) >= 1e9:
        return f"{value / 1e9:.{precision}f}B"
    elif abs(value) >= 1e6:
        return f"{value / 1e6:.{precision}f}M"
    elif abs(value) >= 1e3:
        return f"{value / 1e3:.{precision}f}K"
    else:
        return f"{value:.{precision}f}"

def validate_symbol(symbol: str) -> bool:
    """
    Validate a stock symbol format.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        True if valid, False otherwise
    """
    # Basic validation - typically 1-5 uppercase letters
    pattern = re.compile(r'^[A-Z]{1,5}$')
    return bool(pattern.match(symbol))

def datetime_to_string(dt: datetime, format_str: str = "%Y-%m-%d") -> str:
    """
    Convert datetime to string.
    
    Args:
        dt: Datetime object
        format_str: Format string
        
    Returns:
        Formatted date string
    """
    return dt.strftime(format_str)

def string_to_datetime(date_str: str, format_str: str = "%Y-%m-%d") -> datetime:
    """
    Convert string to datetime.
    
    Args:
        date_str: Date string
        format_str: Format string
        
    Returns:
        Datetime object
    """
    return datetime.strptime(date_str, format_str)

def calculate_date_range(period: str) -> Tuple[datetime, datetime]:
    """
    Calculate start and end dates based on period string.
    
    Args:
        period: Period string (e.g., '1d', '1w', '1m', '1y')
        
    Returns:
        Tuple of (start_date, end_date)
    """
    end_date = datetime.now()
    
    if period == "1d":
        start_date = end_date - timedelta(days=1)
    elif period == "1w":
        start_date = end_date - timedelta(weeks=1)
    elif period == "1m":
        start_date = end_date - timedelta(days=30)
    elif period == "3m":
        start_date = end_date - timedelta(days=90)
    elif period == "6m":
        start_date = end_date - timedelta(days=180)
    elif period == "1y":
        start_date = end_date - timedelta(days=365)
    elif period == "2y":
        start_date = end_date - timedelta(days=730)
    elif period == "5y":
        start_date = end_date - timedelta(days=1825)
    elif period == "max":
        start_date = datetime(1970, 1, 1)  # Very old date
    else:
        # Default to 1 year
        start_date = end_date - timedelta(days=365)
    
    return start_date, end_date

def safe_json_serialize(obj: Any) -> Any:
    """
    Safely serialize objects to JSON, handling non-serializable types.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, (datetime, np.datetime64)):
        return obj.isoformat()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)

def save_to_json(data: Any, filepath: str) -> bool:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Path to JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Serialize to JSON with custom encoder
        with open(filepath, 'w') as f:
            json.dump(data, f, default=safe_json_serialize, indent=2)
        
        return True
    
    except Exception as e:
        logger.error(f"Error saving to JSON: {e}")
        return False

def load_from_json(filepath: str) -> Optional[Any]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data or None if error
    """
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return None
    
    except Exception as e:
        logger.error(f"Error loading from JSON: {e}")
        return None

def calculate_rolling_correlation(
    df1: pd.DataFrame, df2: pd.DataFrame, column: str, window: int = 30
) -> pd.Series:
    """
    Calculate rolling correlation between two DataFrames.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        column: Column to calculate correlation for
        window: Rolling window size
        
    Returns:
        Series with rolling correlation
    """
    # Ensure both DataFrames have the same index
    common_index = df1.index.intersection(df2.index)
    df1 = df1.loc[common_index]
    df2 = df2.loc[common_index]
    
    # Calculate rolling correlation
    corr = df1[column].rolling(window=window).corr(df2[column])
    
    return corr

def get_cache_filepath(key: str, subfolder: str = "cache") -> str:
    """
    Get filepath for a cache file.
    
    Args:
        key: Cache key
        subfolder: Cache subfolder
        
    Returns:
        Cache filepath
    """
    # Clean key for filesystem use
    clean_key = re.sub(r'[^\w\-_\.]', '_', key)
    
    # Create cache directory
    cache_dir = os.path.join("data", subfolder)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Return full path
    return os.path.join(cache_dir, f"{clean_key}.json")

def cache_data(key: str, data: Any, subfolder: str = "cache", ttl: int = 3600) -> bool:
    """
    Cache data with TTL.
    
    Args:
        key: Cache key
        data: Data to cache
        subfolder: Cache subfolder
        ttl: Time-to-live in seconds
        
    Returns:
        True if successful, False otherwise
    """
    filepath = get_cache_filepath(key, subfolder)
    
    # Add timestamp and TTL info
    cache_data = {
        "timestamp": datetime.now().timestamp(),
        "ttl": ttl,
        "data": data
    }
    
    return save_to_json(cache_data, filepath)

def get_cached_data(key: str, subfolder: str = "cache") -> Optional[Any]:
    """
    Get cached data if not expired.
    
    Args:
        key: Cache key
        subfolder: Cache subfolder
        
    Returns:
        Cached data or None if expired/not found
    """
    filepath = get_cache_filepath(key, subfolder)
    
    # Load cache data
    cache_data = load_from_json(filepath)
    
    if cache_data is None:
        return None
    
    # Check expiration
    timestamp = cache_data.get("timestamp", 0)
    ttl = cache_data.get("ttl", 0)
    
    if time.time() - timestamp > ttl:
        # Cache expired
        return None
    
    return cache_data.get("data")

def clean_html(text: str) -> str:
    """
    Remove HTML tags from text.
    
    Args:
        text: Text with potential HTML tags
        
    Returns:
        Cleaned text
    """
    return re.sub(r'<[^>]+>', '', text)

def truncate_text(text: str, max_length: int = 100, ellipsis: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        ellipsis: Ellipsis string to append
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(ellipsis)] + ellipsis

def calculate_cagr(initial_value: float, final_value: float, years: float) -> float:
    """
    Calculate Compound Annual Growth Rate.
    
    Args:
        initial_value: Initial value
        final_value: Final value
        years: Number of years
        
    Returns:
        CAGR as decimal (e.g., 0.1 for 10%)
    """
    if initial_value <= 0 or years <= 0:
        return 0
    
    return (final_value / initial_value) ** (1 / years) - 1

def calculate_drawdown(prices: pd.Series) -> pd.Series:
    """
    Calculate drawdown series for price data.
    
    Args:
        prices: Series of prices
        
    Returns:
        Series with drawdowns
    """
    # Calculate running maximum
    running_max = prices.cummax()
    
    # Calculate drawdown
    drawdown = (prices / running_max) - 1
    
    return drawdown

def calculate_sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Sharpe ratio
    """
    # Annualized return
    annual_return = returns.mean() * periods_per_year
    
    # Annualized volatility
    annual_volatility = returns.std() * np.sqrt(periods_per_year)
    
    # Daily risk-free rate
    daily_rf = risk_free_rate / periods_per_year
    
    # Sharpe ratio
    if annual_volatility == 0:
        return 0
    
    return (annual_return - daily_rf) / annual_volatility

def calculate_sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Sortino ratio
    """
    # Annualized return
    annual_return = returns.mean() * periods_per_year
    
    # Downside returns
    downside_returns = returns[returns < 0]
    
    # Annualized downside deviation
    if len(downside_returns) == 0:
        return np.inf  # No downside returns
    
    annual_downside_dev = downside_returns.std() * np.sqrt(periods_per_year)
    
    # Daily risk-free rate
    daily_rf = risk_free_rate / periods_per_year
    
    # Sortino ratio
    if annual_downside_dev == 0:
        return 0
    
    return (annual_return - daily_rf) / annual_downside_dev
