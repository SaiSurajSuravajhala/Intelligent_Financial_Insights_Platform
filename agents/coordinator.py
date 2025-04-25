"""
Primary Agent Coordinator for orchestrating the hierarchical agent system.
"""
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, initialize_agent, Tool

from agents.data_analysis import DataAnalysisEngine
from agents.insight_generator import InsightGenerator
from agents.strategy_engine import StrategyRecommendationEngine
from agents.user_interaction import UserInteractionManager
from data_collection.api_integration import FinancialDataCollector
from data_collection.web_scraper import AlternativeDataCollector
from data_processing.preprocessor import DataPreprocessor
from data_processing.feature_engineering import FeatureEngineer
from database.db_manager import (
    StockDataManager, CompanyInfoManager, NewsItemManager,
    FinancialMetricManager, StrategyRecommendationManager
)

# Set up logging
logger = logging.getLogger(__name__)

class AgentCoordinator:
    """
    Primary Agent Coordinator that orchestrates the hierarchical agent system.
    Controls the flow of data and tasks between specialized agents.
    """
    
    def __init__(self):
        """Initialize the coordinator with required components."""
        # Initialize data collection components
        self.financial_data_collector = FinancialDataCollector()
        self.alternative_data_collector = AlternativeDataCollector()
        
        # Initialize data processing components
        self.data_preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        
        # Initialize database managers
        self.stock_data_manager = StockDataManager()
        self.company_info_manager = CompanyInfoManager()
        self.news_manager = NewsItemManager()
        self.metric_manager = FinancialMetricManager()
        self.strategy_manager = StrategyRecommendationManager()
        
        # Initialize specialized agents
        self.data_analysis_engine = DataAnalysisEngine()
        self.insight_generator = InsightGenerator()
        self.strategy_engine = StrategyRecommendationEngine()
        self.user_interaction_manager = UserInteractionManager()
        
        # Create a memory for the coordinator
        self.memory = ConversationBufferMemory(memory_key="coordinator_memory")
        
        # Create a task queue for handling sequential operations
        self.task_queue = []
        self.results_cache = {}
    
    def process_stock_request(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """
        Process a complete stock analysis request.
        
        Args:
            symbol: Stock symbol
            period: Time period for historical data
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        logger.info(f"Processing stock request for {symbol} with period {period}")
        
        try:
            # Step 1: Data Collection
            stock_data, company_info, news_data, financial_statements = self._collect_data(symbol, period)
            
            # Step 2: Data Processing
            processed_data = self._process_data(stock_data, news_data)
            
            # Step 3: Data Analysis
            analysis_results = self._analyze_data(processed_data, company_info, financial_statements)
            
            # Step 4: Generate Insights
            insights = self._generate_insights(analysis_results, news_data, symbol)
            
            # Step 5: Generate Strategy Recommendations
            strategy = self._generate_strategy(analysis_results, insights, symbol)
            
            # Combine all results
            comprehensive_results = {
                "symbol": symbol,
                "company_info": company_info,
                "current_data": self._get_current_data(processed_data),
                "historical_data": processed_data,
                "analysis": analysis_results,
                "news": news_data,
                "insights": insights,
                "strategy": strategy,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the results
            self.results_cache[symbol] = comprehensive_results
            
            return comprehensive_results
        
        except Exception as e:
            logger.error(f"Error processing stock request for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def process_user_query(self, query: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query about a stock or general investment.
        
        Args:
            query: User's natural language query
            symbol: Stock symbol (optional)
            
        Returns:
            Dictionary with response and context
        """
        logger.info(f"Processing user query: {query}")
        
        try:
            # Get context from cache if symbol is provided
            context = self.results_cache.get(symbol, {}) if symbol else {}
            
            # Process the query using the user interaction manager
            response = self.user_interaction_manager.process_query(query, context)
            
            return {
                "query": query,
                "symbol": symbol,
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error processing user query: {e}")
            return {
                "query": query,
                "symbol": symbol,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def update_data(self, symbol: str) -> Dict[str, Any]:
        """
        Update data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with update status
        """
        logger.info(f"Updating data for {symbol}")
        
        try:
            # Check if we have cached results
            if symbol in self.results_cache:
                # Get period from cached results
                period = "1y"  # Default
                
                # Re-process the stock request to get fresh data
                updated_results = self.process_stock_request(symbol, period)
                
                return {
                    "symbol": symbol,
                    "status": "updated",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Process new stock request
                self.process_stock_request(symbol)
                
                return {
                    "symbol": symbol,
                    "status": "new_data_collected",
                    "timestamp": datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Error updating data for {symbol}: {e}")
            return {
                "symbol": symbol,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _collect_data(self, symbol: str, period: str) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]], Dict[str, pd.DataFrame]]:
        """
        Collect all required data for a symbol.
        
        Args:
            symbol: Stock symbol
            period: Time period for historical data
            
        Returns:
            Tuple of (stock_data, company_info, news_data, financial_statements)
        """
        logger.info(f"Collecting data for {symbol}")
        
        # Check if we already have recent data in the database
        existing_data = self._get_existing_data(symbol)
        
        if existing_data.get("is_recent", False):
            logger.info(f"Using existing data for {symbol}")
            return (
                existing_data["stock_data"],
                existing_data["company_info"],
                existing_data["news_data"],
                existing_data["financial_statements"]
            )
        
        # Collect new data
        # Financial data
        stock_data = self.financial_data_collector.get_stock_historical_data(symbol, period)
        company_info = self.financial_data_collector.get_company_data(symbol)
        financial_statements = self.financial_data_collector.get_financial_statements(symbol)
        
        # Alternative data
        alternative_data = self.alternative_data_collector.get_all_alternative_data(symbol)
        news_data = alternative_data.get("news", [])
        
        # Save data to database
        self._save_data_to_database(symbol, stock_data, company_info, news_data, financial_statements)
        
        return stock_data, company_info, news_data, financial_statements
    
    def _get_existing_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get existing data for a symbol from the database.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with existing data and freshness indicator
        """
        result = {"is_recent": False}
        
        try:
            # Get latest stock data
            latest_stock = self.stock_data_manager.get_latest_price(symbol)
            
            # Check if data is fresh (less than 1 day old)
            if latest_stock and (datetime.now() - latest_stock.created_at).days < 1:
                # Get all recent stock data
                stock_data_list = self.stock_data_manager.get_by_symbol(symbol)
                
                if stock_data_list:
                    # Convert to DataFrame
                    stock_data = pd.DataFrame([
                        {
                            "date": item.date,
                            "open": item.open,
                            "high": item.high,
                            "low": item.low,
                            "close": item.close,
                            "volume": item.volume,
                            "symbol": item.symbol
                        } for item in stock_data_list
                    ])
                    stock_data.set_index("date", inplace=True)
                    
                    # Get company info
                    company_info_obj = self.company_info_manager.get_by_symbol(symbol)
                    company_info = {
                        "symbol": company_info_obj.symbol,
                        "name": company_info_obj.name,
                        "sector": company_info_obj.sector,
                        "industry": company_info_obj.industry,
                        "country": company_info_obj.country,
                        "market_cap": company_info_obj.market_cap,
                        "pe_ratio": company_info_obj.pe_ratio,
                        "dividend_yield": company_info_obj.dividend_yield,
                        "beta": company_info_obj.beta
                    } if company_info_obj else {}
                    
                    # Get news data
                    news_items = self.news_manager.get_by_symbol(symbol)
                    news_data = [
                        {
                            "title": item.title,
                            "summary": item.summary,
                            "url": item.url,
                            "source": item.source,
                            "published_at": item.published_at,
                            "sentiment_score": item.sentiment_score
                        } for item in news_items
                    ]
                    
                    # For financial statements, we'll collect new data as they change infrequently
                    financial_statements = self.financial_data_collector.get_financial_statements(symbol)
                    
                    result = {
                        "is_recent": True,
                        "stock_data": stock_data,
                        "company_info": company_info,
                        "news_data": news_data,
                        "financial_statements": financial_statements
                    }
        
        except Exception as e:
            logger.error(f"Error getting existing data for {symbol}: {e}")
        
        return result
    
    def _save_data_to_database(
        self, symbol: str, stock_data: pd.DataFrame, company_info: Dict[str, Any],
        news_data: List[Dict[str, Any]], financial_statements: Dict[str, pd.DataFrame]
    ) -> None:
        """
        Save collected data to the database.
        
        Args:
            symbol: Stock symbol
            stock_data: Historical stock price data
            company_info: Company information
            news_data: News articles data
            financial_statements: Financial statements data
        """
        try:
            # Save stock data
            stock_data_list = []
            for date, row in stock_data.iterrows():
                stock_data_list.append({
                    "symbol": symbol,
                    "date": date,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row["volume"])
                })
            self.stock_data_manager.bulk_create(stock_data_list)
            
            # Save company info
            if company_info:
                self.company_info_manager.create({
                    "symbol": symbol,
                    "name": company_info.get("name", ""),
                    "sector": company_info.get("sector", ""),
                    "industry": company_info.get("industry", ""),
                    "country": company_info.get("country", ""),
                    "market_cap": company_info.get("marketCapitalization", 0),
                    "pe_ratio": company_info.get("peRatio", 0),
                    "dividend_yield": company_info.get("dividendYield", 0),
                    "beta": company_info.get("beta", 0)
                })
            
            # Save news data
            news_items = []
            for item in news_data:
                news_items.append({
                    "symbol": symbol,
                    "title": item["title"],
                    "summary": item["summary"],
                    "url": item["url"],
                    "source": item["source"],
                    "published_at": item["published_at"],
                    "sentiment_score": item["sentiment_score"]
                })
            if news_items:
                self.news_manager.bulk_create(news_items)
            
            # Extract financial metrics and save
            metrics = self.feature_engineer.calculate_financial_metrics(financial_statements)
            metric_items = []
            for metric_name, metric_value in metrics.items():
                metric_items.append({
                    "symbol": symbol,
                    "date": datetime.now(),
                    "metric_name": metric_name,
                    "metric_value": float(metric_value) if metric_value is not None else 0.0
                })
            if metric_items:
                self.metric_manager.bulk_create(metric_items)
        
        except Exception as e:
            logger.error(f"Error saving data to database for {symbol}: {e}")
    
    def _process_data(self, stock_data: pd.DataFrame, news_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process collected data.
        
        Args:
            stock_data: Historical stock price data
            news_data: News articles data
            
        Returns:
            Processed stock data with features
        """
        logger.info("Processing data")
        
        try:
            # Preprocess stock data
            cleaned_data = self.data_preprocessor.prepare_stock_data_for_analysis(stock_data)
            
            # Add features
            featured_data = self.feature_engineer.add_all_features(cleaned_data)
            
            # Preprocess news data
            processed_news = self.data_preprocessor.prepare_news_data(news_data)
            
            # Calculate average sentiment from news and add to stock data
            if processed_news:
                sentiment_scores = [item["sentiment_score"] for item in processed_news 
                                   if "sentiment_score" in item and item["sentiment_score"] is not None]
                if sentiment_scores:
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                    # Add sentiment to the latest day in the stock data
                    latest_date = featured_data.index.max()
                    featured_data.loc[latest_date, "news_sentiment"] = avg_sentiment
            
            return featured_data
        
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return stock_data
    
    def _analyze_data(
        self, processed_data: pd.DataFrame, company_info: Dict[str, Any],
        financial_statements: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Analyze processed data using the Data Analysis Engine.
        
        Args:
            processed_data: Processed stock data with features
            company_info: Company information
            financial_statements: Financial statements data
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Analyzing data")
        
        try:
            analysis_results = self.data_analysis_engine.analyze_stock_data(
                processed_data, company_info, financial_statements
            )
            return analysis_results
        
        except Exception as e:
            logger.error(f"Error analyzing data: {e}")
            return {}
    
    def _generate_insights(
        self, analysis_results: Dict[str, Any], news_data: List[Dict[str, Any]], symbol: str
    ) -> Dict[str, Any]:
        """
        Generate insights using the Insight Generator.
        
        Args:
            analysis_results: Results from data analysis
            news_data: News articles data
            symbol: Stock symbol
            
        Returns:
            Dictionary with insights
        """
        logger.info(f"Generating insights for {symbol}")
        
        try:
            insights = self.insight_generator.generate_insights(
                analysis_results, news_data, symbol
            )
            return insights
        
        except Exception as e:
            logger.error(f"Error generating insights for {symbol}: {e}")
            return {}
    
    def _generate_strategy(
        self, analysis_results: Dict[str, Any], insights: Dict[str, Any], symbol: str
    ) -> Dict[str, Any]:
        """
        Generate strategy recommendations using the Strategy Recommendation Engine.
        
        Args:
            analysis_results: Results from data analysis
            insights: Generated insights
            symbol: Stock symbol
            
        Returns:
            Dictionary with strategy recommendations
        """
        logger.info(f"Generating strategy for {symbol}")
        
        try:
            strategy = self.strategy_engine.generate_strategy(
                analysis_results, insights, symbol
            )
            
            # Save strategy to database
            self.strategy_manager.create({
                "symbol": symbol,
                "strategy_type": strategy.get("type", ""),
                "recommendation": strategy.get("recommendation", ""),
                "confidence_score": strategy.get("confidence_score", 0.0),
                "reasoning": strategy.get("reasoning", ""),
                "suggested_entry_price": strategy.get("entry_price", 0.0),
                "suggested_exit_price": strategy.get("exit_price", 0.0),
                "risk_assessment": strategy.get("risk_assessment", ""),
                "time_horizon": strategy.get("time_horizon", "")
            })
            
            return strategy
        
        except Exception as e:
            logger.error(f"Error generating strategy for {symbol}: {e}")
            return {}
    
    def _get_current_data(self, processed_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract current data from processed data.
        
        Args:
            processed_data: Processed stock data
            
        Returns:
            Dictionary with current data
        """
        if processed_data.empty:
            return {}
        
        try:
            # Get the latest data point
            latest_data = processed_data.iloc[-1].to_dict()
            
            # Get key metrics
            current_data = {
                "date": processed_data.index[-1].strftime("%Y-%m-%d"),
                "open": latest_data.get("open", 0),
                "high": latest_data.get("high", 0),
                "low": latest_data.get("low", 0),
                "close": latest_data.get("close", 0),
                "volume": latest_data.get("volume", 0)
            }
            
            # Add percentage change
            if len(processed_data) > 1:
                prev_close = processed_data.iloc[-2]["close"]
                current_data["change_pct"] = (latest_data["close"] - prev_close) / prev_close * 100
            else:
                current_data["change_pct"] = 0
            
            # Add key technical indicators
            for indicator in ["rsi_14", "macd", "bb_upper", "bb_lower", "bb_middle", "volatility_30d"]:
                if indicator in latest_data:
                    current_data[indicator] = latest_data[indicator]
            
            return current_data
        
        except Exception as e:
            logger.error(f"Error extracting current data: {e}")
            return {}
