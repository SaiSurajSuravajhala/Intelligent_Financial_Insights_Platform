"""
Application settings and configuration management.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Application settings
APP_NAME = "Intelligent Financial Insights Platform"
APP_VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# API configuration
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/finance_db.sqlite")

# Web scraping configuration
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
REQUEST_TIMEOUT = 10  # seconds

# Data collection settings
DEFAULT_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
DEFAULT_TIME_PERIOD = "1y"  # Default time period for historical data
NEWS_DAYS_LOOKBACK = 7  # Days to look back for news articles

# Model parameters
SENTIMENT_THRESHOLD_POSITIVE = 0.2
SENTIMENT_THRESHOLD_NEGATIVE = -0.2

# UI Configuration
THEME_COLOR = "#4A90E2"
CHART_HEIGHT = 500
CHART_WIDTH = 800

# LLM Configuration
LLM_MODEL = "claude-3-7-sonnet-20250219"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2000
