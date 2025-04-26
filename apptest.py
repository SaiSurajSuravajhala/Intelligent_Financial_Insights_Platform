"""
Enhanced Streamlit application for the Intelligent Financial Insights Platform.
Complete implementation with favorites, international stocks, learning resources, advanced analysis, and more.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import json
import time
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpha_vantage.timeseries import TimeSeries
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import finnhub
from anthropic import Anthropic
from sqlmodel import SQLModel, create_engine, Session, select, Field, delete
from typing import Optional, List, Dict, Any, Tuple, Union

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Initialize API clients
try:
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
except Exception as e:
    st.error(f"Error initializing Finnhub client: {e}")
    finnhub_client = None

try:
    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
except Exception as e:
    st.error(f"Error initializing Anthropic client: {e}")
    anthropic_client = None

# Configure the page
st.set_page_config(
    page_title="Intelligent Financial Insights Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create any needed directories
os.makedirs("data/cache", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Database setup
DATABASE_URL = "sqlite:///./data/finance_db.sqlite"
engine = create_engine(DATABASE_URL, echo=False)

# Currency configuration by country
CURRENCY_CONFIG = {
    "US": {"symbol": "$", "code": "USD"},
    "India": {"symbol": "₹", "code": "INR"},
    "UK": {"symbol": "£", "code": "GBP"},
    "Canada": {"symbol": "C$", "code": "CAD"},
    "Australia": {"symbol": "A$", "code": "AUD"},
    "Germany": {"symbol": "€", "code": "EUR"}
}

# Define database models
class StockData(SQLModel, table=True):
    """Model for storing historical stock data."""
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    created_at: datetime = Field(default_factory=datetime.utcnow)

class UserQuery(SQLModel, table=True):
    """Model for storing user queries and responses."""
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: Optional[str] = Field(default=None, index=True)
    query_text: str
    response_text: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Favorite(SQLModel, table=True):
    """Model for storing user favorites."""
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    name: Optional[str] = None
    added_at: datetime = Field(default_factory=datetime.utcnow)

# Create database and tables
def create_db_and_tables():
    """Create database and tables safely."""
    try:
        # Create all tables that don't exist yet
        SQLModel.metadata.create_all(engine)
        return True
    except Exception as e:
        st.error(f"Database initialization error: {e}")
        return False

# Call this function to create the tables
create_db_and_tables()

# Basic styling
st.markdown("""
<style>
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
h1, h2, h3, h4, h5, h6 {
    color: #4A90E2;
}
.favorite-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.remove-btn {
    color: red;
    cursor: pointer;
}
.metric-card {
    background-color: #1E2130;
    border-radius: 5px;
    padding: 15px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.custom-info-box {
    background-color: rgba(49, 51, 63, 0.7);
    border-left: 4px solid #4A90E2;
    padding: 10px;
    border-radius: 0 4px 4px 0;
}
</style>
""", unsafe_allow_html=True)

# Helper functions
def get_currency_for_symbol(symbol: str) -> Dict[str, str]:
    """Get currency info based on symbol's exchange suffix"""
    # Extract country from exchange suffix
    if ".BSE" in symbol or ".NSE" in symbol:
        return CURRENCY_CONFIG["India"]
    elif ".LON" in symbol:
        return CURRENCY_CONFIG["UK"]
    elif ".TSX" in symbol or ".TSXV" in symbol:
        return CURRENCY_CONFIG["Canada"]
    elif ".AX" in symbol:
        return CURRENCY_CONFIG["Australia"]
    elif ".FRA" in symbol or ".XETRA" in symbol:
        return CURRENCY_CONFIG["Germany"]
    else:
        return CURRENCY_CONFIG["US"]  # Default to USD

def format_currency(value: float, currency_info: Dict[str, str]) -> str:
    """Format currency with proper symbol and thousands separators"""
    if value >= 1000000000:  # Billions
        return f"{currency_info['symbol']}{value/1000000000:.2f}B"
    elif value >= 1000000:  # Millions
        return f"{currency_info['symbol']}{value/1000000:.2f}M"
    elif value >= 1000:  # Thousands
        return f"{currency_info['symbol']}{value:,.2f}"
    else:
        return f"{currency_info['symbol']}{value:.2f}"

# Favorites management functions using database
def save_favorite_to_db(symbol, name=None):
    """Save a favorite stock to the database."""
    try:
        with Session(engine) as session:
            # Check if already exists
            existing = session.exec(
                select(Favorite).where(Favorite.symbol == symbol)
            ).first()
            
            if not existing:
                favorite = Favorite(symbol=symbol, name=name)
                session.add(favorite)
                session.commit()
                return True
            return False
    except Exception as e:
        st.error(f"Error saving favorite to database: {e}")
        return False

def remove_favorite_from_db(symbol):
    """Remove a favorite stock from the database."""
    try:
        with Session(engine) as session:
            favorite = session.exec(
                select(Favorite).where(Favorite.symbol == symbol)
            ).first()
            
            if favorite:
                session.delete(favorite)
                session.commit()
                return True
            return False
    except Exception as e:
        st.error(f"Error removing favorite from database: {e}")
        return False

def load_favorites_from_db():
    """Load favorite stocks from the database."""
    try:
        with Session(engine) as session:
            favorites = session.exec(select(Favorite)).all()
            return [fav.symbol for fav in favorites]
    except Exception as e:
        st.error(f"Error loading favorites from database: {e}")
        # Default favorites if database fails
        return ["AAPL", "GOOGL", "META", "MSFT", "TSLA"]

# For backwards compatibility - file-based favorites
def save_favorites(favorites):
    """Save favorite stocks to a file."""
    try:
        with open("data/favorites.json", "w") as f:
            json.dump(favorites, f)
        
        # Also save to database
        for symbol in favorites:
            save_favorite_to_db(symbol)
    except Exception as e:
        st.error(f"Error saving favorites: {e}")

def load_favorites():
    """Load favorite stocks from database, fallback to file."""
    # Try database first
    db_favorites = load_favorites_from_db()
    if db_favorites:
        return db_favorites
    
    # Fallback to file
    try:
        with open("data/favorites.json", "r") as f:
            file_favorites = json.load(f)
            
        # Save to database for future
        for symbol in file_favorites:
            save_favorite_to_db(symbol)
            
        return file_favorites
    except:
        # Default favorites
        default_favorites = ["AAPL", "GOOGL", "META", "MSFT", "TSLA"]
        for symbol in default_favorites:
            save_favorite_to_db(symbol)
        return default_favorites

# Save query to database function
def save_query_to_database(query, response, symbol=None):
    """Save user query and response to database."""
    try:
        with Session(engine) as session:
            db_query = UserQuery(
                symbol=symbol,
                query_text=query,
                response_text=response,
                timestamp=datetime.now()
            )
            session.add(db_query)
            session.commit()
            return True
    except Exception as e:
        st.error(f"Error saving query to database: {e}")
        return False

# Exchange suffix guide by country
EXCHANGE_SUFFIXES = {
    # Removed empty value for US
    "India": {".BSE": "Bombay Stock Exchange", ".NSE": "National Stock Exchange of India"},
    "UK": {".LON": "London Stock Exchange"},
    "Canada": {".TSX": "Toronto Stock Exchange", ".TSXV": "TSX Venture Exchange"},
    "Australia": {".AX": "Australian Securities Exchange"},
    "Germany": {".FRA": "Frankfurt Stock Exchange", ".XETRA": "Xetra"}
}

# Function to display database debug information
def display_debug_page():
    st.title("Database Debug Information")
    
    # Show favorites in database
    st.header("Favorites in Database")
    try:
        with Session(engine) as session:
            favorites = session.exec(select(Favorite)).all()
            if favorites:
                for fav in favorites:
                    st.write(f"Symbol: {fav.symbol}, Added: {fav.added_at}")
            else:
                st.write("No favorites in database")
    except Exception as e:
        st.error(f"Error accessing favorites: {e}")
    
    # Show queries in database
    st.header("Recent Queries")
    try:
        with Session(engine) as session:
            queries = session.exec(select(UserQuery).order_by(UserQuery.timestamp.desc()).limit(10)).all()
            if queries:
                for query in queries:
                    st.write(f"Query: {query.query_text}")
                    st.write(f"Symbol: {query.symbol}")
                    st.write(f"Time: {query.timestamp}")
                    st.write("---")
            else:
                st.write("No queries in database")
    except Exception as e:
        st.error(f"Error accessing queries: {e}")
    
    # Show stock data in database
    st.header("Stock Data in Database")
    try:
        with Session(engine) as session:
            # Get all unique symbols
            symbols_result = session.exec(select(StockData.symbol).distinct()).all()
            symbols = list(set(symbols_result))  # Convert to a set to ensure uniqueness
            
            if symbols:
                st.write(f"Stored data for {len(symbols)} symbols:")
                for symbol in symbols:
                    # Count records for this symbol - using len instead of count
                    records = session.exec(
                        select(StockData)
                        .where(StockData.symbol == symbol)
                    ).all()
                    st.write(f"{symbol}: {len(records)} records")
            else:
                st.write("No stock data in database")
    except Exception as e:
        st.error(f"Error accessing stock data: {e}")
    
    # Database reset option
    st.header("Database Management")
    if st.button("Reset Database"):
        try:
            # Better approach: instead of dropping tables, delete records
            with Session(engine) as session:
                # Delete all records from all tables
                session.exec(delete(StockData))  # Delete all stock data records
                session.exec(delete(UserQuery))  # Delete all query records
                session.exec(delete(Favorite))   # Delete all favorite records
                session.commit()
                
                # Add default favorites
                default_favorites = ["AAPL", "GOOGL", "META", "MSFT", "TSLA"]
                for symbol in default_favorites:
                    session.add(Favorite(symbol=symbol))
                session.commit()
                
                st.success("Database reset successfully (all records deleted)")
                
                # Update session state
                if 'favorites' in st.session_state:
                    st.session_state.favorites = default_favorites
        except Exception as e:
            st.error(f"Error resetting database: {e}")
            st.error("Try manually deleting the database file at data/finance_db.sqlite")
    
    # Return button
    if st.button("Return to Dashboard"):
        st.session_state.page = "main"
        st.rerun()

# Advanced technical indicators function
def add_advanced_technical_indicators(df):
    """Add advanced technical indicators beyond the basics."""
    # Make sure we have the basic indicators first
    df = add_technical_indicators(df)
    
    try:
        # Add Stochastic Oscillator
        # Fast stochastic
        n = 14  # Standard lookback period
        df['stoch_k'] = 100 * ((df['close'] - df['low'].rolling(n).min()) / 
                               (df['high'].rolling(n).max() - df['low'].rolling(n).min()))
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()  # 3-day SMA of %K
        
        # Add Average Directional Index (ADX)
        high_diff = df['high'] - df['high'].shift(1)
        low_diff = df['low'].shift(1) - df['low']
        
        plus_dm = high_diff.copy()
        plus_dm[~((high_diff > 0) & (high_diff > low_diff))] = 0
        
        minus_dm = low_diff.copy()
        minus_dm[~((low_diff > 0) & (low_diff > high_diff))] = 0
        
        # Calculate True Range
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        
        true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Avoid division by zero issues
        atr_safe = df['atr'].replace(0, np.nan)
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_safe)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_safe)
        
        # Handle potential division by zero
        plus_di_safe = plus_di.fillna(0)
        minus_di_safe = minus_di.fillna(0)
        di_sum = plus_di_safe + minus_di_safe
        
        # Calculate DX avoiding division by zero
        dx = pd.Series(0, index=df.index)  # Default values
        mask = di_sum > 0
        dx[mask] = 100 * abs(plus_di_safe[mask] - minus_di_safe[mask]) / di_sum[mask]
        
        df['adx'] = dx.rolling(14).mean()
        
        # Add Ichimoku Cloud
        # Conversion line (Tenkan-sen) - 9-period moving average
        df['ichimoku_conversion'] = (df['high'].rolling(9).max() + 
                                    df['low'].rolling(9).min()) / 2
        
        # Base line (Kijun-sen) - 26-period moving average
        df['ichimoku_base'] = (df['high'].rolling(26).max() + 
                              df['low'].rolling(26).min()) / 2
        
        # Leading Span A (Senkou Span A)
        df['ichimoku_span_a'] = ((df['ichimoku_conversion'] + df['ichimoku_base']) / 2).shift(26)
        
        # Leading Span B (Senkou Span B)
        df['ichimoku_span_b'] = ((df['high'].rolling(52).max() + 
                                 df['low'].rolling(52).min()) / 2).shift(26)
                                 
        # Add Fibonacci Retracement levels
        min_price = df['close'].min()
        max_price = df['close'].max()
        diff = max_price - min_price
        
        df['fib_0'] = min_price  # 0% level (the low)
        df['fib_23.6'] = min_price + 0.236 * diff  # 23.6% level
        df['fib_38.2'] = min_price + 0.382 * diff  # 38.2% level
        df['fib_50'] = min_price + 0.5 * diff      # 50% level
        df['fib_61.8'] = min_price + 0.618 * diff  # 61.8% level
        df['fib_100'] = max_price  # 100% level (the high)
        
        # Calculate Money Flow Index (MFI)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        # Calculate positive and negative money flow
        pos_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        neg_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        # Calculate 14-day positive and negative money flow sum
        pos_flow_sum = pos_flow.rolling(window=14).sum()
        neg_flow_sum = neg_flow.rolling(window=14).sum()
        
        # Calculate money flow ratio
        mf_ratio = pos_flow_sum / neg_flow_sum
        
        # Calculate MFI
        df['mfi'] = 100 - (100 / (1 + mf_ratio))
        
        return df
    except Exception as e:
        st.warning(f"Could not add some advanced indicators: {e}")
        return df

# Function to detect patterns
def detect_patterns(df):
    """Detect common chart patterns in the price data."""
    patterns = {}
    
    try:
        # Detect support and resistance levels
        # Find significant highs and lows
        df['rolling_high'] = df['high'].rolling(10, center=True).max()
        df['rolling_low'] = df['low'].rolling(10, center=True).min()
        
        high_points = df[df['high'] == df['rolling_high']]
        low_points = df[df['low'] == df['rolling_low']]
        
        # Get current price
        current_price = df['close'].iloc[-1]
        
        # Find support levels (significant lows below current price)
        supports = []
        for idx, row in low_points.iterrows():
            # Only consider if it's a significant low (two lower lows on either side)
            price_level = row['low']
            if price_level < current_price:
                # Count how many times prices approached this level (within 1%)
                approach_count = ((df['low'] >= price_level * 0.99) & 
                                 (df['low'] <= price_level * 1.01)).sum()
                if approach_count >= 2:
                    supports.append(price_level)
        
        # Find resistance levels (significant highs above current price)
        resistances = []
        for idx, row in high_points.iterrows():
            # Only consider if it's a significant high (two higher highs on either side)
            price_level = row['high']
            if price_level > current_price:
                # Count how many times prices approached this level (within 1%)
                approach_count = ((df['high'] >= price_level * 0.99) & 
                                 (df['high'] <= price_level * 1.01)).sum()
                if approach_count >= 2:
                    resistances.append(price_level)
        
        # Sort and limit to top 3
        supports = sorted(set(supports), reverse=True)[:3]
        resistances = sorted(set(resistances))[:3]
        
        patterns['support_resistance'] = {
            'supports': [float(s) for s in supports],
            'resistances': [float(r) for r in resistances]
        }
    except Exception as e:
        st.warning(f"Error detecting support/resistance: {e}")
    
    try:
        # Detect trend direction and strength
        if 'ma_50' in df.columns and 'adx' in df.columns:
            current_price = df['close'].iloc[-1]
            ma_50 = df['ma_50'].iloc[-1]
            adx = df['adx'].iloc[-1]
            
            # Determine trend direction
            trend_direction = "bullish" if current_price > ma_50 else "bearish"
            
            # Determine trend strength based on ADX
            if adx > 25:
                trend_strength = "strong"
            elif adx > 15:
                trend_strength = "moderate"
            else:
                trend_strength = "weak"
            
            patterns['trend'] = {
                'direction': trend_direction,
                'strength': trend_strength,
                'adx': float(adx)
            }
    except Exception as e:
        st.warning(f"Error detecting trend: {e}")
    
    try:
        # Detect oversold/overbought conditions
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            if rsi > 70:
                patterns['overbought'] = {
                    'detected': True,
                    'indicator': 'RSI',
                    'value': float(rsi)
                }
            elif rsi < 30:
                patterns['oversold'] = {
                    'detected': True,
                    'indicator': 'RSI',
                    'value': float(rsi)
                }
    except Exception as e:
        st.warning(f"Error detecting overbought/oversold: {e}")
    
    return patterns

# Calculate performance metrics
def calculate_performance_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate various performance metrics for a stock.
    
    Args:
        df: DataFrame with at least close prices
        
    Returns:
        Dictionary of performance metrics
    """
    metrics = {}
    
    try:
        # Calculate returns
        returns = df['close'].pct_change().dropna()
        log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
        
        # Calculate simple period returns
        periods = {
            '1d': 1,
            '1w': 5,
            '1m': 21,
            '3m': 63,
            '6m': 126,
            '1y': 252
        }
        
        for period_name, days in periods.items():
            if len(df) > days:
                start_price = df['close'].iloc[-days-1] if days < len(df) else df['close'].iloc[0]
                end_price = df['close'].iloc[-1]
                period_return = (end_price - start_price) / start_price
                metrics[f"{period_name}_return"] = float(period_return)
                metrics[f"{period_name}_return_pct"] = float(period_return * 100)
        
        # Annualized return (if we have enough data)
        if len(df) >= 252:  # At least a year
            years = len(df) / 252
            total_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
            annualized_return = (1 + total_return) ** (1 / years) - 1
            metrics["annualized_return"] = float(annualized_return)
            metrics["annualized_return_pct"] = float(annualized_return * 100)
        
        # Volatility (annualized standard deviation)
        if len(returns) > 20:
            daily_std = returns.std()
            annualized_std = daily_std * np.sqrt(252)
            metrics["volatility"] = float(annualized_std)
            metrics["volatility_pct"] = float(annualized_std * 100)
        
        # Downside risk (semi-deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            annualized_downside = downside_std * np.sqrt(252)
            metrics["downside_risk"] = float(annualized_downside)
            metrics["downside_risk_pct"] = float(annualized_downside * 100)
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        max_return = cumulative_returns.cummax()
        drawdowns = (cumulative_returns / max_return) - 1
        max_drawdown = drawdowns.min()
        metrics["max_drawdown"] = float(max_drawdown)
        metrics["max_drawdown_pct"] = float(max_drawdown * 100)
        
        # Risk-adjusted returns
        risk_free_rate = 0.03 / 252  # Approximate daily risk-free rate (3% annual)
        if len(returns) > 20:
            # Sharpe Ratio
            sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)
            metrics["sharpe_ratio"] = float(sharpe_ratio)
            
            # Sortino Ratio (if we have downside returns)
            if len(downside_returns) > 0:
                sortino_ratio = (returns.mean() - risk_free_rate) / downside_std * np.sqrt(252)
                metrics["sortino_ratio"] = float(sortino_ratio)
        
        # Win rate (percentage of positive days)
        win_rate = (returns > 0).sum() / len(returns)
        metrics["win_rate"] = float(win_rate)
        metrics["win_rate_pct"] = float(win_rate * 100)
        
        return metrics
    
    except Exception as e:
        st.error(f"Error calculating performance metrics: {e}")
        return {}

# Cache function for stock data to avoid repeated API calls
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_data(symbol, period="1y"):
    try:
        # Get current date
        end_date = datetime.now()
        
        # Calculate start date based on period
        if period == "1 Week":
            start_date = end_date - timedelta(days=7)
        elif period == "1 Month":
            start_date = end_date - timedelta(days=30)
        elif period == "3 Months":
            start_date = end_date - timedelta(days=90)
        elif period == "6 Months":
            start_date = end_date - timedelta(days=180)
        else:  # 1 Year
            start_date = end_date - timedelta(days=365)
        
        # Format dates for cache key
        cache_key = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        cache_file = f"data/cache/{cache_key}.csv"
        
        # Check if we have cached data
        if os.path.exists(cache_file):
            data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return data
        
        # Fetch data from Alpha Vantage
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
        
        # Rename columns
        data.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Sort by date
        data = data.sort_index()
        
        # Filter for date range - ensure we're only getting recent data
        data = data[(data.index >= start_date) & (data.index <= end_date)]
        
        # Save to cache
        data.to_csv(cache_file)
        
        # Save to database
        try:
            with Session(engine) as session:
                # Check if we already have this data
                existing_records = session.exec(
                    select(StockData).where(StockData.symbol == symbol)
                ).all()
                
                existing_dates = [record.date for record in existing_records]
                
                # Prepare records
                db_records = []
                for date, row in data.iterrows():
                    if date not in existing_dates:
                        db_records.append(StockData(
                            symbol=symbol,
                            date=date,
                            open=float(row['open']),
                            high=float(row['high']),
                            low=float(row['low']),
                            close=float(row['close']),
                            volume=int(row['volume'])
                        ))
                
                # Add new records
                if db_records:
                    session.add_all(db_records)
                    session.commit()
        except Exception as e:
            st.error(f"Error saving stock data to database: {e}")
        
        return data
    
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        # Fall back to mock data if API fails
        days_map = {"1 Week": 7, "1 Month": 30, "3 Months": 90, 
                "6 Months": 180, "1 Year": 252}
        days = days_map.get(period, 252)
        
        # Generate dates going BACK from current date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, periods=days)
        
        # Create random data with a trend
        base_price = 100
        trend = np.linspace(0, 0.2, days)  # Slight upward trend
        noise = np.random.normal(0, 0.02, days)  # Daily noise
        
        price_mov = trend + noise
        closes = base_price * (1 + np.cumsum(price_mov))
        
        mock_data = pd.DataFrame({
            'open': closes * (1 - np.random.uniform(0, 0.01, days)),
            'high': closes * (1 + np.random.uniform(0, 0.02, days)),
            'low': closes * (1 - np.random.uniform(0, 0.02, days)),
            'close': closes,
            'volume': np.random.normal(1000000, 200000, days)
        }, index=dates)
        
        return mock_data

# Function to add technical indicators
def add_technical_indicators(df):
    # Add moving averages
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()
    df['ma_200'] = df['close'].rolling(window=200).mean()
    
    # Add Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Add MACD
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Add Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    
    # Calculate daily returns
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    return df

# Function to create candlestick chart
def create_candlestick_chart(df, title="Stock Price", patterns=None, currency_info=None):
    if currency_info is None:
        currency_info = CURRENCY_CONFIG["US"]
    
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Price"
    ))
    
    # Add moving averages
    if 'ma_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['ma_20'],
            name="20-day MA",
            line=dict(color='blue', width=1)
        ))
    
    if 'ma_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['ma_50'],
            name="50-day MA",
            line=dict(color='red', width=1)
        ))
    
    if 'ma_200' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['ma_200'],
            name="200-day MA",
            line=dict(color='purple', width=1, dash='dot')
        ))
    
    # Add Bollinger Bands if available
    if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['bb_upper'],
            name="BB Upper",
            line=dict(color='rgba(0,176,246,0.7)', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['bb_lower'],
            name="BB Lower",
            line=dict(color='rgba(0,176,246,0.7)', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(0,176,246,0.05)'
        ))
    
    # Add support and resistance levels if available
    if patterns and 'support_resistance' in patterns:
        sr_data = patterns['support_resistance']
        
        # Add support levels
        if 'supports' in sr_data and sr_data['supports']:
            for i, level in enumerate(sr_data['supports']):
                fig.add_hline(
                    y=level,
                    line=dict(color='green', width=1, dash='dash'),
                    annotation_text=f"Support {i+1}",
                    annotation_position="right"
                )
        
        # Add resistance levels
        if 'resistances' in sr_data and sr_data['resistances']:
            for i, level in enumerate(sr_data['resistances']):
                fig.add_hline(
                    y=level,
                    line=dict(color='red', width=1, dash='dash'),
                    annotation_text=f"Resistance {i+1}",
                    annotation_position="right"
                )
    
    # Update layout with currency symbol in the axis title
    currency_symbol = currency_info['symbol']
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_symbol})",
        height=500,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Remove rangeslider
    fig.update_layout(xaxis_rangeslider_visible=False)
    
    return fig

# Function to create RSI chart
def create_rsi_chart(df, title="Relative Strength Index (RSI)"):
    fig = go.Figure()
    
    # Add RSI line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['rsi'],
        name="RSI",
        line=dict(color='purple', width=1)
    ))
    
    # Add overbought and oversold lines
    fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green")
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="RSI",
        height=300,
        template="plotly_dark"
    )
    
    return fig

# Function to create MACD chart
def create_macd_chart(df, title="MACD"):
    fig = go.Figure()
    
    # Add MACD line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['macd'],
        name="MACD",
        line=dict(color='blue', width=1)
    ))
    
    # Add Signal line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['macd_signal'],
        name="Signal",
        line=dict(color='red', width=1)
    ))
    
    # Add Histogram
    colors = ['green' if val >= 0 else 'red' for val in df['macd_histogram']]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['macd_histogram'],
        name="Histogram",
        marker_color=colors
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="MACD",
        height=300,
        template="plotly_dark"
    )
    
    return fig

# Function to create performance metrics chart
def create_performance_chart(metrics: Dict[str, Any], title="Performance"):
    # Extract return metrics
    return_data = {}
    for key, value in metrics.items():
        if key.endswith('_return_pct') and key != 'annualized_return_pct':
            # Extract period name
            period = key.replace('_return_pct', '')
            return_data[period.upper()] = value
    
    if not return_data:
        return None
    
    # Sort periods chronologically
    period_order = {'1D': 0, '1W': 1, '1M': 2, '3M': 3, '6M': 4, '1Y': 5}
    periods = sorted(return_data.keys(), key=lambda x: period_order.get(x, 99))
    values = [return_data[p] for p in periods]
    
    # Create colors based on values
    colors = ['green' if val >= 0 else 'red' for val in values]
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=periods,
            y=values,
            marker_color=colors,
            text=[f"{v:.2f}%" for v in values],
            textposition='auto'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=300,
        width=None,
        template="plotly_dark",
        yaxis_title="Return (%)",
        xaxis_title="Time Period"
    )
    
    # Add reference line at 0
    fig.add_hline(y=0, line_dash="solid", line_color="white", line_width=1)
    
    return fig

# Create comparison chart for multiple stocks
def create_comparison_chart(data_dict, column='close', normalize=True, title="Comparison"):
    """Create a comparison chart for multiple symbols."""
    if not data_dict or len(data_dict) < 2:
        return None
    
    fig = go.Figure()
    
    # Add lines for each symbol
    for symbol, df in data_dict.items():
        if column in df.columns:
            y_values = df[column]
            
            # Normalize to percentage change if requested
            if normalize:
                first_value = y_values.iloc[0]
                if first_value != 0:
                    y_values = (y_values / first_value - 1) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=y_values,
                    name=symbol,
                    line=dict(width=2)
                )
            )
    
    # Update layout
    y_title = "% Change" if normalize else column.capitalize()
    fig.update_layout(
        title=title,
        height=400,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title=y_title,
        xaxis_title="Date"
    )
    
    return fig

# Function to get company news
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_company_news(symbol, count=5):
    try:
        if not finnhub_client:
            return []
            
        # Remove exchange suffix for news search
        base_symbol = symbol.split('.')[0]
        
        # Calculate date range (7 days)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        # Get news from Finnhub
        news = finnhub_client.company_news(base_symbol, _from=start_date, to=end_date)
        
        # Return the most recent articles
        return news[:count]
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

# Function to get AI response
# Function to get AI response
def get_ai_response(query, symbol, stock_data):
    try:
        if not anthropic_client:
            return "AI response unavailable. Please check your Anthropic API key."
        
        if stock_data is None or stock_data.empty:
            return f"I don't have enough data about {symbol} to provide a detailed answer. Please try another stock or question."
            
        # Create context information about the stock for Claude
        latest_data = stock_data.iloc[-1]
        current_price = latest_data['close']
        
        # Get currency for symbol
        currency_info = get_currency_for_symbol(symbol)
        
        rsi_info = "RSI not available"
        if 'rsi' in stock_data.columns:
            rsi_info = f"RSI: {latest_data['rsi']:.2f}"
        
        ma_info = ""
        if 'ma_20' in stock_data.columns and 'ma_50' in stock_data.columns:
            ma_info = f"20-day MA: {currency_info['symbol']}{latest_data['ma_20']:.2f}, 50-day MA: {currency_info['symbol']}{latest_data['ma_50']:.2f}"
        
        # Period change calculation
        first_price = stock_data['close'].iloc[0]
        period_change = ((current_price / first_price) - 1) * 100
        
        # MACD info
        macd_info = ""
        if 'macd' in stock_data.columns:
            macd_info = f"MACD: {latest_data['macd']:.3f}, Signal: {latest_data['macd_signal']:.3f}"
        
        # Create context for Claude
        context = f"""
        Current information about {symbol}:
        - Current Price: {currency_info['symbol']}{current_price:.2f} ({currency_info['code']})
        - {rsi_info}
        - {ma_info}
        - {macd_info}
        - Recent Performance: {period_change:.2f}% change
        
        You are an investment advisor assistant helping answer questions about {symbol} stock.
        Provide concise, informative answers about investment strategies, technical analysis, and market trends.
        
        If the query includes document context, consider this information when formulating your response.
        When referencing information from documents, clearly indicate which document the information comes from
        by stating "According to [Document Name]" or similar attribution.
        
        If asked for specific investment advice, include appropriate disclaimers about the risks involved.
        Focus on educational content rather than specific buy/sell instructions.
        """
        
        # Get response from Claude
        response = anthropic_client.messages.create(
            model="claude-3-7-sonnet-20250219",  # Use the model you have access to
            max_tokens=800,
            system=context,
            messages=[
                {"role": "user", "content": query}
            ]
        )
        
        return response.content[0].text
    
    except Exception as e:
        st.error(f"Error getting AI response: {e}")
        return f"I'm having trouble generating a response with the AI service. As a fallback: {symbol} is currently trading at {currency_info['symbol']}{latest_data['close']:.2f}. For detailed investment advice, please consult a financial advisor."

# Strategic advice generator
def generate_investment_strategy(symbol: str, stock_data: pd.DataFrame, patterns: Dict) -> Dict[str, Any]:
    """Generate a basic investment strategy based on technical indicators"""
    
    if stock_data.empty:
        return {
            "recommendation": "HOLD",
            "confidence": 0.5,
            "reasoning": "Insufficient data to make a recommendation."
        }
    
    try:
        # Get latest data
        current_data = stock_data.iloc[-1]
        
        # Initialize score (0.5 = neutral)
        buy_signals = 0
        sell_signals = 0
        total_signals = 0
        
        # Check trend
        if "trend" in patterns:
            total_signals += 2
            trend = patterns["trend"]
            if trend["direction"] == "bullish":
                buy_signals += 1
                # Extra point if trend is strong
                if trend["strength"] == "strong":
                    buy_signals += 1
                elif trend["strength"] == "moderate":
                    buy_signals += 0.5
            else:  # bearish
                sell_signals += 1
                # Extra point if trend is strong
                if trend["strength"] == "strong":
                    sell_signals += 1
                elif trend["strength"] == "moderate":
                    sell_signals += 0.5
        
        # Check RSI
        if "rsi" in current_data:
            total_signals += 1
            rsi = current_data["rsi"]
            if rsi > 70:
                sell_signals += 1  # Overbought
            elif rsi < 30:
                buy_signals += 1   # Oversold
            elif rsi > 50:
                buy_signals += 0.2  # Slight bullish bias
            elif rsi < 50:
                sell_signals += 0.2  # Slight bearish bias
        
        # Check MACD
        if all(x in current_data for x in ["macd", "macd_signal"]):
            total_signals += 1
            macd = current_data["macd"]
            signal = current_data["macd_signal"]
            
            if macd > signal:
                buy_signals += 1  # Bullish signal
            else:
                sell_signals += 1  # Bearish signal
        
        # Check moving averages
        if all(x in current_data for x in ["ma_20", "ma_50"]):
            total_signals += 1
            ma_20 = current_data["ma_20"]
            ma_50 = current_data["ma_50"]
            price = current_data["close"]
            
            if price > ma_20 and price > ma_50 and ma_20 > ma_50:
                buy_signals += 1  # Strong bullish
            elif price < ma_20 and price < ma_50 and ma_20 < ma_50:
                sell_signals += 1  # Strong bearish
        
        # Calculate confidence and recommendation
        if total_signals > 0:
            buy_score = buy_signals / total_signals
            sell_score = sell_signals / total_signals
            
            if buy_score > sell_score:
                if buy_score > 0.7:
                    recommendation = "BUY"
                    confidence = buy_score
                    reasoning = "Strong bullish signals from multiple indicators."
                else:
                    recommendation = "HOLD"
                    confidence = 0.5 + (buy_score - 0.5) * 0.5  # Scale between 0.5-0.75
                    reasoning = "Moderate bullish bias, but not strong enough for a buy recommendation."
            elif sell_score > buy_score:
                if sell_score > 0.7:
                    recommendation = "SELL"
                    confidence = sell_score
                    reasoning = "Strong bearish signals from multiple indicators."
                else:
                    recommendation = "HOLD"
                    confidence = 0.5 - (sell_score - 0.5) * 0.5  # Scale between 0.25-0.5
                    reasoning = "Moderate bearish bias, but not strong enough for a sell recommendation."
            else:
                recommendation = "HOLD"
                confidence = 0.5
                reasoning = "Mixed signals suggesting a neutral stance."
        else:
            recommendation = "HOLD"
            confidence = 0.5
            reasoning = "Insufficient technical signals to make a directional recommendation."
        
        # Add some nuance to the reasoning
        if "rsi" in current_data:
            rsi = current_data["rsi"]
            if rsi > 70:
                reasoning += f" RSI at {rsi:.1f} indicates overbought conditions."
            elif rsi < 30:
                reasoning += f" RSI at {rsi:.1f} indicates oversold conditions."
        
        if "trend" in patterns:
            trend = patterns["trend"]
            reasoning += f" The overall trend is {trend['direction']} with {trend['strength']} strength."
        
        # Add support/resistance context
        if "support_resistance" in patterns:
            sr = patterns["support_resistance"]
            current_price = current_data["close"]
            
            if "supports" in sr and sr["supports"] and current_price < sr["supports"][0] * 1.05:
                reasoning += f" Price is near a support level at {sr['supports'][0]:.2f}."
            elif "resistances" in sr and sr["resistances"] and current_price > sr["resistances"][0] * 0.95:
                reasoning += f" Price is approaching resistance at {sr['resistances'][0]:.2f}."
        
        return {
            "recommendation": recommendation,
            "confidence": float(confidence),
            "reasoning": reasoning
        }
    
    except Exception as e:
        st.error(f"Error generating strategy: {e}")
        return {
            "recommendation": "HOLD",
            "confidence": 0.5,
            "reasoning": f"Error generating recommendation: {str(e)}"
        }

# Simple Coordinator (from app.py)
class SimpleCoordinator:
    """Simplified version of the AgentCoordinator from app.py."""
    
    def __init__(self):
        """Initialize coordinator."""
        self.results_cache = {}
    
    def process_stock_request(self, symbol, period="1y"):
        """Process a stock request (simplified)."""
        # Get stock data
        stock_data = get_stock_data(symbol, period)
        
        # Add technical indicators
        stock_data_with_indicators = add_technical_indicators(stock_data)
        
        # Add advanced indicators
        try:
            stock_data_with_advanced = add_advanced_technical_indicators(stock_data_with_indicators)
        except Exception as e:
            st.warning(f"Could not add advanced indicators: {e}")
            stock_data_with_advanced = stock_data_with_indicators
        
        # Detect patterns
        patterns = {}
        try:
            patterns = detect_patterns(stock_data_with_advanced)
        except Exception as e:
            st.warning(f"Could not detect patterns: {e}")
        
        # Calculate performance metrics
        performance_metrics = {}
        try:
            performance_metrics = calculate_performance_metrics(stock_data_with_advanced)
        except Exception as e:
            st.warning(f"Could not calculate performance metrics: {e}")
        
        # Generate strategy
        strategy = {}
        try:
            strategy = generate_investment_strategy(symbol, stock_data_with_advanced, patterns)
        except Exception as e:
            st.warning(f"Could not generate strategy: {e}")
            strategy = {
                "recommendation": "HOLD",
                "confidence": 0.5,
                "reasoning": "Unable to generate a recommendation at this time."
            }
        
        # Get currency info for the symbol
        currency_info = get_currency_for_symbol(symbol)
        
        # Create a basic result structure
        results = {
            "symbol": symbol,
            "company_info": self._get_mock_company_info(symbol),
            "current_data": self._get_current_data(stock_data_with_advanced),
            "historical_data": stock_data_with_advanced,
            "analysis": self._perform_basic_analysis(stock_data_with_advanced),
            "performance": performance_metrics,
            "news": get_company_news(symbol),
            "patterns": patterns,
            "currency_info": currency_info,
            "insights": {
                "summary": f"Analysis for {symbol} based on {period} of data."
            },
            "strategy": {
                "overall": strategy
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache the results
        self.results_cache[symbol] = results
        
        return results
    
    def process_user_query(self, query, symbol=None):
        """Process a user query."""
        context = {}
        if symbol in self.results_cache:
            context = self.results_cache[symbol]
        
        # Get stock data if we have a symbol
        stock_data = None
        if symbol:
            stock_data = context.get("historical_data", None)
        
        # Generate response
        # Access uploaded_files from the session state
        uploaded_files = st.session_state.get("uploaded_files", None)
        response = get_ai_response(query, symbol, stock_data)
        
        # Save query to database
        save_query_to_database(query, response, symbol)
        
        return {
            "query": query,
            "symbol": symbol,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_mock_company_info(self, symbol):
        """Get basic company info."""
        # Extract base symbol without exchange suffix
        base_symbol = symbol.split('.')[0]
        
        # Set company name based on exchange
        if ".BSE" in symbol or ".NSE" in symbol:
            company_name = f"{base_symbol} Ltd."
            country = "India"
            if ".BSE" in symbol:
                exchange = "BSE"
            else:
                exchange = "NSE"
        elif ".LON" in symbol:
            company_name = f"{base_symbol} plc"
            country = "UK"
            exchange = "London Stock Exchange"
        elif ".TSX" in symbol or ".TSXV" in symbol:
            company_name = f"{base_symbol} Inc."
            country = "Canada"
            exchange = "Toronto Stock Exchange"
        elif ".AX" in symbol:
            company_name = f"{base_symbol} Limited"
            country = "Australia"
            exchange = "ASX"
        elif ".FRA" in symbol or ".XETRA" in symbol:
            company_name = f"{base_symbol} AG"
            country = "Germany"
            exchange = "Frankfurt Stock Exchange"
        else:
            company_name = f"{base_symbol} Inc."
            country = "US"
            exchange = "NASDAQ"
        
        # Set currency based on country
        currency = CURRENCY_CONFIG.get(country, CURRENCY_CONFIG["US"])["code"]
        
        return {
            "name": company_name,
            "symbol": symbol,
            "sector": "Technology",  # Mock sector
            "industry": "Software",   # Mock industry
            "country": country,
            "exchange": exchange,
            "currency": currency
        }
    
    def _get_current_data(self, stock_data):
        """Extract current data from stock data."""
        if stock_data.empty:
            return {}
        
        try:
            latest_data = stock_data.iloc[-1]
            previous_data = stock_data.iloc[-2] if len(stock_data) > 1 else latest_data
            
            return {
                "date": stock_data.index[-1].strftime("%Y-%m-%d"),
                "open": float(latest_data["open"]),
                "high": float(latest_data["high"]),
                "low": float(latest_data["low"]),
                "close": float(latest_data["close"]),
                "volume": int(latest_data["volume"]),
                "change_pct": float((latest_data["close"] - previous_data["close"]) / previous_data["close"] * 100),
                "change": float(latest_data["close"] - previous_data["close"])
            }
        except Exception as e:
            st.error(f"Error extracting current data: {e}")
            return {}
    
    def _perform_basic_analysis(self, stock_data):
        """Perform basic analysis of stock data."""
        if stock_data.empty:
            return {}
        
        try:
            # Calculate some basic statistics
            returns = stock_data["close"].pct_change().dropna()
            
            result = {
                "basic_stats": {
                    "mean_return": float(returns.mean()),
                    "std_return": float(returns.std()),
                    "min_price": float(stock_data["close"].min()),
                    "max_price": float(stock_data["close"].max()),
                    "current_price": float(stock_data["close"].iloc[-1])
                },
                "technical": {
                    "trend": "bullish" if stock_data["close"].iloc[-1] > stock_data["ma_50"].iloc[-1] else "bearish",
                    "rsi": float(stock_data["rsi"].iloc[-1]) if "rsi" in stock_data.columns else 0
                }
            }
            
            # Add more advanced technical analysis
            if "macd" in stock_data.columns and "macd_signal" in stock_data.columns:
                macd = stock_data["macd"].iloc[-1]
                signal = stock_data["macd_signal"].iloc[-1]
                
                result["technical"]["macd"] = {
                    "value": float(macd),
                    "signal": float(signal),
                    "histogram": float(macd - signal),
                    "trend": "bullish" if macd > signal else "bearish"
                }
            
            # Add Bollinger Band analysis
            if all(col in stock_data.columns for col in ["bb_upper", "bb_middle", "bb_lower"]):
                price = stock_data["close"].iloc[-1]
                upper = stock_data["bb_upper"].iloc[-1]
                lower = stock_data["bb_lower"].iloc[-1]
                middle = stock_data["bb_middle"].iloc[-1]
                
                # Calculate percentage of price relative to the BB range
                bb_range = upper - lower
                if bb_range > 0:
                    bb_position = (price - lower) / bb_range
                else:
                    bb_position = 0.5
                
                result["technical"]["bollinger_bands"] = {
                    "upper": float(upper),
                    "middle": float(middle),
                    "lower": float(lower),
                    "width": float(bb_range / middle if middle > 0 else 0),
                    "position": float(bb_position),
                    "status": "overbought" if price > upper else "oversold" if price < lower else "neutral"
                }
            
            return result
        
        except Exception as e:
            st.error(f"Error in basic analysis: {e}")
            return {}

# Function to display learning page
def display_learning_page():
    st.title("Stock Market Learning Center")
    
    st.markdown("""
    ## Stock Market Basics
    
    The stock market is a place where people can buy and sell ownership shares of public companies. 
    When you buy a stock, you're purchasing a small piece of that company.
    
    ### Key Concepts:
    
    **Stocks**: Represent ownership in a company
    
    **Exchange**: A marketplace where stocks are traded (e.g., NYSE, NASDAQ)
    
    **Bull Market**: When prices are rising or expected to rise
    
    **Bear Market**: When prices are falling or expected to fall
    
    ## Technical Analysis
    
    Technical analysis involves studying price movements and patterns to predict future price behavior.
    
    ### Common Technical Indicators:
    
    **Moving Averages**: Show the average price over a specific time period
    
    **Relative Strength Index (RSI)**: Measures the speed and change of price movements
    
    **MACD (Moving Average Convergence Divergence)**: Shows the relationship between two moving averages
    
    ## Fundamental Analysis
    
    Fundamental analysis examines a company's financial health and business model.
    
    ### Key Metrics:
    
    **P/E Ratio**: Price-to-Earnings ratio compares a company's share price to its earnings per share
    
    **Market Cap**: Total value of a company's outstanding shares
    
    **Revenue Growth**: Rate at which a company's sales are increasing
    
    ## Investment Strategies
    
    ### Different Approaches:
    
    **Value Investing**: Buying stocks that appear undervalued
    
    **Growth Investing**: Focusing on companies with high growth potential
    
    **Dividend Investing**: Investing in stocks that pay regular dividends
    
    **Index Investing**: Buying a diverse portfolio that mirrors a market index
    
    ## Risk Management
    
    Managing risk is crucial for long-term investment success.
    
    ### Key Principles:
    
    **Diversification**: Spreading investments across different assets
    
    **Position Sizing**: Limiting how much you invest in any single stock
    
    **Stop-Loss Orders**: Setting price points to automatically sell and limit losses
    """)
    
    # Return button
    if st.button("Return to Dashboard"):
        st.session_state.page = "main"
        st.rerun()

# Initialize coordinator
coordinator = SimpleCoordinator()

# Main app code
def main():
    # Check page state
    if 'page' not in st.session_state:
        st.session_state.page = "main"
    
    # Display appropriate page
    if st.session_state.page == "learning":
        display_learning_page()
    elif st.session_state.page == "debug":
        display_debug_page()
    else:
        # Main app interface
        st.title("Intelligent Financial Insights Platform")
        
        # Sidebar
        st.sidebar.title("Settings")
        
        # Load favorites
        if 'favorites' not in st.session_state:
            st.session_state.favorites = load_favorites()
        
        # Favorites section with removal option
        st.sidebar.header("Favorites")
        
        # Custom favorites selector with remove buttons
        selected_index = 0
        
        # Create a container for the favorites selector
        fav_container = st.sidebar.container()
        
        # Use columns to create a selector with remove buttons
        col1, col2 = fav_container.columns([3, 1])
        
        with col1:
            symbol = st.selectbox(
                "Select Stock", 
                options=st.session_state.favorites,
                key="favorite_selector"
            )
        
        # Implement remove button in second column
        with col2:
            if st.button("✕", key="remove_button"):
                if symbol in st.session_state.favorites:
                    st.session_state.favorites.remove(symbol)
                    save_favorites(st.session_state.favorites)
                    remove_favorite_from_db(symbol)
                    st.rerun()
        
        # Country and exchange selection for search
        st.sidebar.header("Search for Stock")
        country = st.sidebar.selectbox("Country", ["US", "India", "UK", "Canada", "Australia", "Germany", "Other"])
        
        # Show exchange suffixes for the selected country
        if country != "US" and country in EXCHANGE_SUFFIXES:
            suffixes = EXCHANGE_SUFFIXES[country]
            suffix_text = "Exchange suffixes for " + country + ":\n"
            for suffix, exchange in suffixes.items():
                suffix_display = suffix if suffix else "(none)"
                suffix_text += f"- {suffix_display}: {exchange}\n"
            st.sidebar.info(suffix_text)
        
        # Custom symbol input
        custom_symbol = st.sidebar.text_input("Enter symbol:", key="custom_symbol")
        
        # Format symbol with exchange if needed
        if custom_symbol:
            formatted_symbol = custom_symbol.upper()
            try:
                if country == "India" and not "." in formatted_symbol:
                    # For India, ask specifically which exchange
                    india_exchange = st.sidebar.radio(
                        "Select Indian exchange:",
                        ["BSE", "NSE"]
                    )
                    formatted_symbol = f"{formatted_symbol}.{india_exchange}"
                elif country == "UK" and not "." in formatted_symbol:
                    formatted_symbol = f"{formatted_symbol}.LON"
                elif country == "Canada" and not "." in formatted_symbol:
                    formatted_symbol = f"{formatted_symbol}.TSX"
                elif country == "Australia" and not "." in formatted_symbol:
                    formatted_symbol = f"{formatted_symbol}.AX"
                elif country == "Germany" and not "." in formatted_symbol:
                    formatted_symbol = f"{formatted_symbol}.FRA"
                
                symbol = formatted_symbol
            except Exception as e:
                st.sidebar.error(f"Error setting exchange: {e}")
                # Fallback to default (no change)
                symbol = formatted_symbol
            
            # Add to favorites button
            if st.sidebar.button("Add to Favorites"):
                if symbol not in st.session_state.favorites:
                    # Add to session state
                    st.session_state.favorites.append(symbol)
                    # Save to file (legacy)
                    save_favorites(st.session_state.favorites)
                    # Save to database
                    save_favorite_to_db(symbol)
                    st.sidebar.success(f"Added {symbol} to favorites!")
        
        # Time period selection
        st.sidebar.header("Select Time Period:")
        period_options = ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"]
        period = st.sidebar.select_slider(
            "",
            options=period_options,
            value="1 Week"
        )
        
        # Learning resources
        st.sidebar.markdown("---")
        st.sidebar.markdown("**New to stock market? [Learn More]()**")
        if st.sidebar.button("Stock Market Basics"):
            st.session_state.page = "learning"
            st.rerun()
        
        # Debug button 
        if st.sidebar.button("Debug Database"):
            st.session_state.page = "debug"
            st.rerun()

            # Document upload section



        
# Find this section in your main() function:
# After the Debug button but before the "Get analysis results using coordinator" section

# Document upload section
st.sidebar.header("Document Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload financial documents (PDF, Word)",
    type=['pdf', 'docx'],
    accept_multiple_files=True
)

# Process uploaded files
if uploaded_files:
    st.sidebar.write(f"{len(uploaded_files)} document(s) uploaded")
    
    # Process documents button
    if st.sidebar.button("Process Documents"):
        try:
            # Import the simple_rag_utils module
            from simple_rag_utils import process_and_add_document
            
            with st.spinner("Processing documents..."):
                for file in uploaded_files:
                    success, message = process_and_add_document(file)
                    if success:
                        st.sidebar.success(message)
                    else:
                        st.sidebar.error(message)
                
                st.sidebar.success("All documents processed!")
                
                # Store uploaded files in session state for later use
                if 'uploaded_files' not in st.session_state:
                    st.session_state.uploaded_files = []
                st.session_state.uploaded_files.extend(uploaded_files)
        except Exception as e:
            st.sidebar.error(f"Error processing documents: {e}")

# Get the currently selected symbol
            symbol = st.session_state.get("favorite_selector", "AAPL")  # Default to AAPL if nothing selected
        # Get analysis results using coordinator

        # Get the currently selected symbol
            symbol = st.session_state.get("favorite_selector", "AAPL")  # Default to AAPL if nothing selected

#            Add this line to fix the error:
            period = st.session_state.get("period", "1 Week")  # Default to 1 Week

            # Get the currently selected symbol
            symbol = st.session_state.get("favorite_selector", "AAPL")  # Default to AAPL if nothing selected

# Make sure period is defined
            period = st.session_state.get("period", "1 Week")  # Default to 1 Week if not set

            # Add these two lines right before line 1807
symbol = st.session_state.get("favorite_selector", "AAPL")  # Default to AAPL if nothing selected
period = st.session_state.get("period", "1 Week")  # Default to 1 Week if not set

# Then this line should work
with st.spinner(f"Analyzing {symbol}..."):

        with st.spinner(f"Analyzing {symbol}..."):
            try:
                # Process the stock request through coordinator
                results = coordinator.process_stock_request(symbol, period)
                
                # Get stock data
                stock_data = results.get("historical_data")
                
                # Get currency info
                currency_info = results.get("currency_info", CURRENCY_CONFIG["US"])
            except Exception as e:
                st.error(f"Error analyzing {symbol}: {e}")
                st.warning("Using mock data instead")
                # Generate mock data
                stock_data = pd.DataFrame()
                results = {}
                currency_info = CURRENCY_CONFIG["US"]
        
        # Main tabs
        tabs = st.tabs(["Analysis Dashboard", "Investment Assistant"])
        
        with tabs[1]:
            st.header(f"{symbol} Analysis")
            
            # Current price info
            if stock_data is not None and not stock_data.empty:
                current_data = results.get("current_data", {})
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    price_change = current_data.get("change", 0)
                    st.metric(
                        label="Current Price",
                        value=format_currency(current_data.get("close", 0), currency_info),
                        delta=f"{price_change:.2f}"
                    )
                
                with col2:
                    st.metric(
                        label="Volume",
                        value=f"{int(current_data.get('volume', 0)):,}"
                    )
                
                with col3:
                    period_change = current_data.get("change_pct", 0)
                    st.metric(
                        label=f"{period} Change",
                        value=f"{period_change:.2f}%"
                    )
                
                # Strategy recommendation
                if "strategy" in results and "overall" in results["strategy"]:
                    strategy = results["strategy"]["overall"]
                    
                    # Create a colored box based on recommendation
                    recommendation = strategy.get("recommendation", "HOLD")
                    confidence = strategy.get("confidence", 0.5)
                    reasoning = strategy.get("reasoning", "")
                    
                    # Set color based on recommendation
                    rec_color = "#4CAF50" if recommendation == "BUY" else "#F44336" if recommendation == "SELL" else "#2196F3"
                    
                    # Display recommendation card
                    st.markdown(f"""
                    <div style="background-color: rgba({rec_color.lstrip('#')[:2]}, {rec_color.lstrip('#')[2:4]}, {rec_color.lstrip('#')[4:]}, 0.2); 
                                border-left: 4px solid {rec_color};
                                padding: 10px; border-radius: 4px; margin-bottom: 20px;">
                        <h3 style="color: {rec_color};">{recommendation} Recommendation - {confidence*100:.0f}% Confidence</h3>
                        <p>{reasoning}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
                # Price History section
                st.header("Price History")
                candlestick_fig = create_candlestick_chart(
                    stock_data, 
                    f"{symbol} Price History",
                    patterns=results.get("patterns", {}),
                    currency_info=currency_info
                )
                st.plotly_chart(candlestick_fig, use_container_width=True)
                
                # Performance chart
                if "performance" in results and results["performance"]:
                    st.subheader("Performance Analysis")
                    perf_chart = create_performance_chart(
                        results["performance"], 
                        "Return by Time Period"
                    )
                    if perf_chart is not None:
                        st.plotly_chart(perf_chart, use_container_width=True)
                
                # Technical indicators section
                st.header("Technical Indicators")
                
                # Create two columns for indicator charts
                indicator_col1, indicator_col2 = st.columns(2)
                
                with indicator_col1:
                    # RSI Chart
                    if 'rsi' in stock_data.columns:
                        rsi_fig = create_rsi_chart(stock_data)
                        st.plotly_chart(rsi_fig, use_container_width=True)
                
                with indicator_col2:
                    # MACD Chart
                    if 'macd' in stock_data.columns:
                        macd_fig = create_macd_chart(stock_data)
                        st.plotly_chart(macd_fig, use_container_width=True)
                
                # Technical Analysis Summary
                st.header("Technical Analysis")
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    # Trend analysis
                    if "patterns" in results and "trend" in results["patterns"]:
                        trend_data = results["patterns"]["trend"]
                        st.subheader("Trend Analysis")
                        
                        trend_direction = trend_data.get("direction", "neutral")
                        trend_strength = trend_data.get("strength", "weak")
                        
                        # Display with appropriate color
                        direction_color = "green" if trend_direction == "bullish" else "red" if trend_direction == "bearish" else "gray"
                        st.markdown(f"**Direction:** <span style='color:{direction_color};'>{trend_direction.upper()}</span>", unsafe_allow_html=True)
                        st.write(f"**Strength:** {trend_strength.title()}")
                        st.write(f"**ADX:** {trend_data.get('adx', 0):.1f}")
                    
                    # Support & resistance
                    if "patterns" in results and "support_resistance" in results["patterns"]:
                        sr_data = results["patterns"]["support_resistance"]
                        st.subheader("Support & Resistance")
                        
                        if "supports" in sr_data and sr_data["supports"]:
                            st.write("**Support Levels:**")
                            for i, level in enumerate(sr_data["supports"][:3]):
                                st.write(f"{i+1}. {format_currency(level, currency_info)}")
                        
                        if "resistances" in sr_data and sr_data["resistances"]:
                            st.write("**Resistance Levels:**")
                            for i, level in enumerate(sr_data["resistances"][:3]):
                                st.write(f"{i+1}. {format_currency(level, currency_info)}")
                
                with summary_col2:
                    # Indicator summary
                    st.subheader("Key Indicators")
                    
                    # RSI
                    if "rsi" in stock_data.columns:
                        rsi_value = stock_data["rsi"].iloc[-1]
                        rsi_status = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
                        rsi_color = "red" if rsi_value > 70 else "green" if rsi_value < 30 else "white"
                        st.markdown(f"**RSI (14):** {rsi_value:.1f} - <span style='color:{rsi_color};'>{rsi_status}</span>", unsafe_allow_html=True)
                    
                    # MACD
                    if all(x in stock_data.columns for x in ["macd", "macd_signal"]):
                        macd = stock_data["macd"].iloc[-1]
                        signal = stock_data["macd_signal"].iloc[-1]
                        hist = stock_data["macd_histogram"].iloc[-1]
                        
                        macd_status = "Bullish" if macd > signal else "Bearish"
                        macd_color = "green" if macd > signal else "red"
                        
                        st.markdown(f"**MACD:** {macd:.3f} - <span style='color:{macd_color};'>{macd_status}</span>", unsafe_allow_html=True)
                        st.write(f"**Signal:** {signal:.3f}")
                        st.write(f"**Histogram:** {hist:.3f}")
                    
                    # Bollinger Bands
                    if all(x in stock_data.columns for x in ["bb_upper", "bb_middle", "bb_lower"]):
                        price = stock_data["close"].iloc[-1]
                        upper = stock_data["bb_upper"].iloc[-1]
                        lower = stock_data["bb_lower"].iloc[-1]
                        
                        bb_status = "Above Upper Band" if price > upper else "Below Lower Band" if price < lower else "Within Bands"
                        bb_color = "red" if price > upper else "green" if price < lower else "white"
                        
                        st.markdown(f"**Bollinger Bands:** <span style='color:{bb_color};'>{bb_status}</span>", unsafe_allow_html=True)
                        st.write(f"**Upper Band:** {format_currency(upper, currency_info)}")
                        st.write(f"**Lower Band:** {format_currency(lower, currency_info)}")
                
                # News section
                st.header("Recent News")
                news_items = results.get("news", [])
                
                if news_items:
                    for item in news_items:
                        with st.expander(item.get('headline', 'News item')):
                            st.write(f"**Source**: {item.get('source', 'Unknown')}")
                            st.write(f"**Date**: {datetime.fromtimestamp(item.get('datetime', 0)).strftime('%Y-%m-%d')}")
                            st.write(item.get('summary', 'No summary available'))
                            st.write(f"[Read more]({item.get('url', '#')})")
                else:
                    st.info(f"No recent news found for {symbol}")
                
                # Data table
                st.header("Recent Data")
                display_cols = ['open', 'high', 'low', 'close', 'volume', 'ma_20', 'ma_50', 'rsi']
                # Filter for columns that exist
                existing_cols = [col for col in display_cols if col in stock_data.columns]
                display_data = stock_data[existing_cols].copy() if existing_cols else stock_data
                
                # Format with appropriate currency symbol
                currency_symbol = currency_info['symbol']
                st.dataframe(display_data.tail(10).style.format({
                    'open': f'{currency_symbol}{{:.2f}}',
                    'high': f'{currency_symbol}{{:.2f}}',
                    'low': f'{currency_symbol}{{:.2f}}',
                    'close': f'{currency_symbol}{{:.2f}}',
                    'volume': '{:,.0f}',
                    'ma_20': f'{currency_symbol}{{:.2f}}',
                    'ma_50': f'{currency_symbol}{{:.2f}}',
                    'rsi': '{:.2f}'
                }))
            else:
                st.error("No data available for this stock symbol.")
                st.info("Please try another stock symbol or check your API keys.")
        
        



            


# Run the app
if __name__ == "__main__":
    main()