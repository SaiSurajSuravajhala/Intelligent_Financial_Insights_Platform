"""
API keys and credentials.
NOTE: This file should be added to .gitignore to prevent exposing API keys.
For local development, create a .env file with these variables.
"""

# Example .env file format:
"""
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
FINNHUB_API_KEY=your_finnhub_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_alpha_vantage_api_key():
    """Get Alpha Vantage API key from environment variables."""
    key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not key:
        raise ValueError("ALPHA_VANTAGE_API_KEY not found in environment variables")
    return key

def get_finnhub_api_key():
    """Get Finnhub API key from environment variables."""
    key = os.getenv("FINNHUB_API_KEY")
    if not key:
        raise ValueError("FINNHUB_API_KEY not found in environment variables")
    return key

def get_anthropic_api_key():
    """Get Anthropic API key from environment variables."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    return key
