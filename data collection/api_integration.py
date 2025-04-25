"""
Financial API integration module.
Handles data collection from various financial APIs.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import requests
import finnhub
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData

from config.api_keys import get_alpha_vantage_api_key, get_finnhub_api_key
from config.settings import REQUEST_TIMEOUT

# Set up logging
logger = logging.getLogger(__name__)

class AlphaVantageAPI:
    """Alpha Vantage API integration."""
    
    def __init__(self):
        """Initialize with API key."""
        self.api_key = get_alpha_vantage_api_key()
        self.time_series = TimeSeries(key=self.api_key, output_format='pandas')
        self.fundamental_data = FundamentalData(key=self.api_key, output_format='pandas')
    
    def get_daily_stock_data(
        self, symbol: str, full: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Get daily stock data for a symbol.
        
        Args:
            symbol: Stock symbol
            full: If True, get full history; otherwise, get compact (100 data points)
            
        Returns:
            Tuple of dataframe and metadata
        """
        try:
            outputsize = 'full' if full else 'compact'
            data, meta_data = self.time_series.get_daily(
                symbol=symbol, outputsize=outputsize
            )
            # Convert index to datetime if it's not already
            data.index = pd.to_datetime(data.index)
            # Sort by date (ascending)
            data = data.sort_index()
            # Rename columns
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            return data, meta_data
        except Exception as e:
            logger.error(f"Error fetching data from Alpha Vantage: {e}")
            raise
    
    def get_intraday_stock_data(
        self, symbol: str, interval: str = '5min', full: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Get intraday stock data for a symbol.
        
        Args:
            symbol: Stock symbol
            interval: Time interval between data points (1min, 5min, 15min, 30min, 60min)
            full: If True, get full history; otherwise, get compact (100 data points)
            
        Returns:
            Tuple of dataframe and metadata
        """
        try:
            outputsize = 'full' if full else 'compact'
            data, meta_data = self.time_series.get_intraday(
                symbol=symbol, interval=interval, outputsize=outputsize
            )
            # Rename columns
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            return data, meta_data
        except Exception as e:
            logger.error(f"Error fetching intraday data from Alpha Vantage: {e}")
            raise
    
    def get_company_overview(self, symbol: str) -> pd.DataFrame:
        """
        Get company overview for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with company overview
        """
        try:
            data, _ = self.fundamental_data.get_company_overview(symbol=symbol)
            return data
        except Exception as e:
            logger.error(f"Error fetching company overview from Alpha Vantage: {e}")
            raise
    
    def get_income_statement(self, symbol: str) -> pd.DataFrame:
        """
        Get income statement for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with income statement
        """
        try:
            data, _ = self.fundamental_data.get_income_statement_annual(symbol=symbol)
            return data
        except Exception as e:
            logger.error(f"Error fetching income statement from Alpha Vantage: {e}")
            raise
    
    def get_balance_sheet(self, symbol: str) -> pd.DataFrame:
        """
        Get balance sheet for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with balance sheet
        """
        try:
            data, _ = self.fundamental_data.get_balance_sheet_annual(symbol=symbol)
            return data
        except Exception as e:
            logger.error(f"Error fetching balance sheet from Alpha Vantage: {e}")
            raise
    
    def get_cash_flow(self, symbol: str) -> pd.DataFrame:
        """
        Get cash flow statement for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with cash flow statement
        """
        try:
            data, _ = self.fundamental_data.get_cash_flow_annual(symbol=symbol)
            return data
        except Exception as e:
            logger.error(f"Error fetching cash flow from Alpha Vantage: {e}")
            raise


class FinnhubAPI:
    """Finnhub API integration."""
    
    def __init__(self):
        """Initialize with API key."""
        self.api_key = get_finnhub_api_key()
        self.client = finnhub.Client(api_key=self.api_key)
    
    def get_stock_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time stock quote for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with quote data
        """
        try:
            return self.client.quote(symbol)
        except Exception as e:
            logger.error(f"Error fetching stock quote from Finnhub: {e}")
            raise
    
    def get_company_news(
        self, symbol: str, from_date: str, to_date: str
    ) -> List[Dict[str, Any]]:
        """
        Get company news for a symbol and date range.
        
        Args:
            symbol: Stock symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            List of news items
        """
        try:
            return self.client.company_news(symbol, _from=from_date, to=to_date)
        except Exception as e:
            logger.error(f"Error fetching company news from Finnhub: {e}")
            raise
    
    def get_company_profile(self, symbol: str) -> Dict[str, Any]:
        """
        Get company profile for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with company profile
        """
        try:
            return self.client.company_profile2(symbol=symbol)
        except Exception as e:
            logger.error(f"Error fetching company profile from Finnhub: {e}")
            raise
    
    def get_recommendation_trends(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get recommendation trends for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of recommendation trends
        """
        try:
            return self.client.recommendation_trends(symbol)
        except Exception as e:
            logger.error(f"Error fetching recommendation trends from Finnhub: {e}")
            raise
    
    def get_earnings(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get earnings for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of earnings
        """
        try:
            return self.client.company_earnings(symbol)
        except Exception as e:
            logger.error(f"Error fetching earnings from Finnhub: {e}")
            raise
    
    def get_stock_targets(self, symbol: str) -> Dict[str, Any]:
        """
        Get stock price targets for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with price targets
        """
        try:
            return self.client.price_target(symbol)
        except Exception as e:
            logger.error(f"Error fetching price targets from Finnhub: {e}")
            raise


class FinancialDataCollector:
    """Unified financial data collector."""
    
    def __init__(self):
        """Initialize with API instances."""
        self.alpha_vantage = AlphaVantageAPI()
        self.finnhub = FinnhubAPI()
    
    def get_stock_historical_data(
        self, symbol: str, period: str = "1y"
    ) -> pd.DataFrame:
        """
        Get historical stock data for a symbol.
        
        Args:
            symbol: Stock symbol
            period: Time period ("1d", "1w", "1m", "3m", "6m", "1y", "2y", "5y", "max")
            
        Returns:
            DataFrame with historical data
        """
        # Determine if we need full or compact data
        full = period in ["1y", "2y", "5y", "max"]
        # Get data from Alpha Vantage
        data, _ = self.alpha_vantage.get_daily_stock_data(symbol, full=full)
        
        # Filter based on period
        if period != "max":
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
            
            data = data[data.index >= start_date]
        
        # Add symbol column
        data['symbol'] = symbol
        
        return data
    
    def get_company_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive company data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with company data
        """
        # Get data from Alpha Vantage
        try:
            company_overview = self.alpha_vantage.get_company_overview(symbol)
            # Convert Series to dict
            company_data = company_overview.iloc[0].to_dict() if not company_overview.empty else {}
        except Exception as e:
            logger.warning(f"Error fetching company overview from Alpha Vantage: {e}")
            company_data = {}
        
        # Get data from Finnhub
        try:
            finnhub_profile = self.finnhub.get_company_profile(symbol)
            # Update company_data with Finnhub data
            if finnhub_profile:
                company_data.update(finnhub_profile)
        except Exception as e:
            logger.warning(f"Error fetching company profile from Finnhub: {e}")
        
        return company_data
    
    def get_latest_company_news(
        self, symbol: str, days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Get latest company news for a symbol.
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            
        Returns:
            List of news items
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for Finnhub API
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        # Get news from Finnhub
        return self.finnhub.get_company_news(symbol, from_date, to_date)
    
    def get_financial_statements(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Get financial statements for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with financial statements
        """
        financial_data = {}
        
        # Get income statement
        try:
            financial_data['income_statement'] = self.alpha_vantage.get_income_statement(symbol)
        except Exception as e:
            logger.warning(f"Error fetching income statement: {e}")
            financial_data['income_statement'] = pd.DataFrame()
        
        # Get balance sheet
        try:
            financial_data['balance_sheet'] = self.alpha_vantage.get_balance_sheet(symbol)
        except Exception as e:
            logger.warning(f"Error fetching balance sheet: {e}")
            financial_data['balance_sheet'] = pd.DataFrame()
        
        # Get cash flow
        try:
            financial_data['cash_flow'] = self.alpha_vantage.get_cash_flow(symbol)
        except Exception as e:
            logger.warning(f"Error fetching cash flow: {e}")
            financial_data['cash_flow'] = pd.DataFrame()
        
        return financial_data
    
    def get_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with real-time data
        """
        # Get quote from Finnhub
        quote = self.finnhub.get_stock_quote(symbol)
        
        # Get price targets
        try:
            price_targets = self.finnhub.get_stock_targets(symbol)
            quote.update(price_targets)
        except Exception as e:
            logger.warning(f"Error fetching price targets: {e}")
        
        # Get recommendation trends
        try:
            recommendations = self.finnhub.get_recommendation_trends(symbol)
            if recommendations:
                quote['recommendations'] = recommendations[0]  # Latest recommendation
        except Exception as e:
            logger.warning(f"Error fetching recommendations: {e}")
        
        return quote
