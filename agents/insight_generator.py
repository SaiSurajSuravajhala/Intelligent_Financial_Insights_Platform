"""
Insight Generator for financial analysis.
Converts analysis results into meaningful insights.
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from anthropic import Anthropic
import json

from config.settings import ANTHROPIC_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

# Set up logging
logger = logging.getLogger(__name__)

class InsightGenerator:
    """
    Specialized Agent for generating insights from financial analysis.
    Uses Claude API to generate narrative explanations of financial trends.
    """
    
    def __init__(self):
        """Initialize insight generator with Claude API client."""
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = LLM_MODEL
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
    
    def generate_insights(
        self, analysis_results: Dict[str, Any], news_data: List[Dict[str, Any]], symbol: str
    ) -> Dict[str, Any]:
        """
        Generate insights from analysis results and news data.
        
        Args:
            analysis_results: Results from data analysis
            news_data: News articles data
            symbol: Stock symbol
            
        Returns:
            Dictionary with insights
        """
        logger.info(f"Generating insights for {symbol}")
        
        insights = {}
        
        try:
            # Generate different types of insights
            
            # Technical insights
            insights["technical"] = self.generate_technical_insights(analysis_results, symbol)
            
            # Fundamental insights
            insights["fundamental"] = self.generate_fundamental_insights(analysis_results, symbol)
            
            # News and sentiment insights
            insights["news"] = self.generate_news_insights(news_data, symbol)
            
            # Risk assessment
            insights["risk"] = self.generate_risk_insights(analysis_results, symbol)
            
            # Overall market context
            insights["market_context"] = self.generate_market_context(analysis_results, news_data, symbol)
            
            # Summary insights (combining all insights)
            insights["summary"] = self.generate_summary_insights(insights, symbol)
            
            return insights
        
        except Exception as e:
            logger.error(f"Error generating insights for {symbol}: {e}")
            return {
                "error": str(e),
                "summary": f"Unable to generate complete insights for {symbol} due to an error."
            }
    
    def generate_technical_insights(self, analysis_results: Dict[str, Any], symbol: str) -> str:
        """
        Generate technical analysis insights.
        
        Args:
            analysis_results: Results from data analysis
            symbol: Stock symbol
            
        Returns:
            String with technical insights
        """
        try:
            # Extract technical analysis data
            technical = analysis_results.get("technical", {})
            patterns = analysis_results.get("patterns", {})
            basic_stats = analysis_results.get("basic_stats", {})
            
            # Current price and statistics
            current_data = {}
            if "all" in basic_stats:
                current_price = basic_stats["all"].get("current_price", "N/A")
                current_data["current_price"] = current_price
            
            # Create prompt for Claude
            prompt = f"""
            You are a financial analyst specializing in technical analysis. Generate detailed insights about {symbol} based on the following technical analysis data:
            
            Technical Analysis: {json.dumps(technical, indent=2)}
            
            Chart Patterns: {json.dumps(patterns, indent=2)}
            
            Current Price: {current_data.get('current_price', 'N/A')}
            
            Your analysis should include insights on:
            1. Current trend direction and strength
            2. Key technical indicators (moving averages, RSI, MACD, Bollinger Bands)
            3. Support and resistance levels
            4. Chart patterns and their implications
            5. Volume analysis and what it suggests
            6. Potential entry/exit points based on technical factors
            
            Keep your analysis factual, insightful, and actionable. Focus on what the technical indicators suggest about future price movements. Be specific and reference the data points.
            
            Format your response as a cohesive paragraph of technical analysis insights.
            """
            
            # Generate insights using Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system="You are a financial analyst specializing in technical analysis.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract insights from response
            insights = response.content[0].text
            
            return insights
        
        except Exception as e:
            logger.error(f"Error generating technical insights for {symbol}: {e}")
            return f"Unable to generate technical insights for {symbol} due to an error."
    
    def generate_fundamental_insights(self, analysis_results: Dict[str, Any], symbol: str) -> str:
        """
        Generate fundamental analysis insights.
        
        Args:
            analysis_results: Results from data analysis
            symbol: Stock symbol
            
        Returns:
            String with fundamental insights
        """
        try:
            # Extract fundamental analysis data
            fundamental = analysis_results.get("fundamental", {})
            
            # Create prompt for Claude
            prompt = f"""
            You are a financial analyst specializing in fundamental analysis. Generate detailed insights about {symbol} based on the following fundamental data:
            
            Fundamental Analysis: {json.dumps(fundamental, indent=2)}
            
            Your analysis should include insights on:
            1. Company's financial health and stability
            2. Profitability and growth metrics
            3. Key valuation metrics (P/E, P/B, etc.) and how they compare to industry averages
            4. Strengths and weaknesses based on financial statements
            5. Long-term investment potential based on fundamentals
            
            Keep your analysis factual, insightful, and actionable. Focus on what the fundamental data suggests about the company's intrinsic value and future prospects. Be specific and reference the data points.
            
            Format your response as a cohesive paragraph of fundamental analysis insights.
            """
            
            # Generate insights using Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system="You are a financial analyst specializing in fundamental analysis.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract insights from response
            insights = response.content[0].text
            
            return insights
        
        except Exception as e:
            logger.error(f"Error generating fundamental insights for {symbol}: {e}")
            return f"Unable to generate fundamental insights for {symbol} due to an error."
    
    def generate_news_insights(self, news_data: List[Dict[str, Any]], symbol: str) -> str:
        """
        Generate insights from news and sentiment data.
        
        Args:
            news_data: News articles data
            symbol: Stock symbol
            
        Returns:
            String with news insights
        """
        try:
            # Limit to recent news (top 10 articles)
            recent_news = news_data[:10] if news_data else []
            
            # Calculate average sentiment
            sentiment_scores = [item["sentiment_score"] for item in recent_news 
                               if "sentiment_score" in item and item["sentiment_score"] is not None]
            
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
            # Create prompt for Claude
            prompt = f"""
            You are a financial analyst specializing in news analysis and sentiment. Generate detailed insights about {symbol} based on the following recent news data:
            
            Recent News Articles: {json.dumps(recent_news, indent=2)}
            
            Average Sentiment Score: {avg_sentiment} (scale from -1 negative to +1 positive)
            
            Your analysis should include insights on:
            1. Key themes or trends in recent news coverage
            2. Potential market-moving events mentioned in the news
            3. Overall sentiment and how it might impact stock price
            4. Any notable news that investors should be aware of
            5. How news sentiment aligns with technical or fundamental indicators
            
            Keep your analysis factual, insightful, and actionable. Focus on what the news suggests about potential stock movements. Be specific and reference the data points.
            
            Format your response as a cohesive paragraph of news-based insights.
            """
            
            # Generate insights using Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system="You are a financial analyst specializing in news analysis and sentiment.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract insights from response
            insights = response.content[0].text
            
            return insights
        
        except Exception as e:
            logger.error(f"Error generating news insights for {symbol}: {e}")
            return f"Unable to generate news insights for {symbol} due to an error."
    
    def generate_risk_insights(self, analysis_results: Dict[str, Any], symbol: str) -> str:
        """
        Generate risk assessment insights.
        
        Args:
            analysis_results: Results from data analysis
            symbol: Stock symbol
            
        Returns:
            String with risk assessment insights
        """
        try:
            # Extract relevant data for risk assessment
            volatility = analysis_results.get("volatility", {})
            performance = analysis_results.get("performance", {})
            anomalies = analysis_results.get("anomalies", {})
            
            # Create prompt for Claude
            prompt = f"""
            You are a financial risk analyst. Generate a detailed risk assessment for {symbol} based on the following data:
            
            Volatility Analysis: {json.dumps(volatility, indent=2)}
            
            Performance Metrics: {json.dumps(performance, indent=2)}
            
            Market Anomalies: {json.dumps(anomalies, indent=2)}
            
            Your risk assessment should include insights on:
            1. Overall risk level (low, moderate, high) with clear justification
            2. Key risk factors specific to this stock
            3. Volatility analysis and what it implies for risk
            4. Potential downside scenarios
            5. Risk mitigation strategies for investors interested in this stock
            
            Keep your analysis factual, insightful, and actionable. Focus on helping investors understand the specific risks associated with this stock. Be specific and reference the data points.
            
            Format your response as a cohesive paragraph of risk assessment insights.
            """
            
            # Generate insights using Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system="You are a financial risk analyst.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract insights from response
            insights = response.content[0].text
            
            return insights
        
        except Exception as e:
            logger.error(f"Error generating risk insights for {symbol}: {e}")
            return f"Unable to generate risk assessment for {symbol} due to an error."
    
    def generate_market_context(
        self, analysis_results: Dict[str, Any], news_data: List[Dict[str, Any]], symbol: str
    ) -> str:
        """
        Generate market context insights.
        
        Args:
            analysis_results: Results from data analysis
            news_data: News articles data
            symbol: Stock symbol
            
        Returns:
            String with market context insights
        """
        try:
            # Extract any sector/industry info
            fundamental = analysis_results.get("fundamental", {})
            company_info = fundamental.get("company", {})
            sector = company_info.get("sector", "Unknown")
            industry = company_info.get("industry", "Unknown")
            
            # Create prompt for Claude
            prompt = f"""
            You are a market analyst with expertise in sector trends and macroeconomic factors. Generate insights about the market context for {symbol} (in the {sector} sector, {industry} industry) based on the available data.
            
            Your market context analysis should include insights on:
            1. How this stock might be positioned within its sector/industry
            2. Relevant macroeconomic factors that could impact this stock
            3. Sector-specific trends or challenges
            4. Competitive positioning based on available information
            5. How broader market conditions might affect this specific stock
            
            Keep your analysis factual, insightful, and actionable. Focus on helping investors understand how market context might influence investment decisions for this stock. Make reasonable inferences based on the sector and industry, even if specific competitor data isn't provided.
            
            Format your response as a cohesive paragraph of market context insights.
            """
            
            # Generate insights using Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system="You are a market analyst with expertise in sector trends and macroeconomic factors.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract insights from response
            insights = response.content[0].text
            
            return insights
        
        except Exception as e:
            logger.error(f"Error generating market context for {symbol}: {e}")
            return f"Unable to generate market context for {symbol} due to an error."
    
    def generate_summary_insights(self, insights: Dict[str, str], symbol: str) -> str:
        """
        Generate summary insights combining all individual insights.
        
        Args:
            insights: Dictionary with individual insights
            symbol: Stock symbol
            
        Returns:
            String with summary insights
        """
        try:
            # Combine all insights
            all_insights = {
                "technical": insights.get("technical", "No technical insights available."),
                "fundamental": insights.get("fundamental", "No fundamental insights available."),
                "news": insights.get("news", "No news insights available."),
                "risk": insights.get("risk", "No risk assessment available."),
                "market_context": insights.get("market_context", "No market context available.")
            }
            
            # Create prompt for Claude
            prompt = f"""
            You are a comprehensive financial analyst. Generate a concise summary of all insights for {symbol} based on the following detailed analyses:
            
            Technical Analysis Insights: {all_insights["technical"]}
            
            Fundamental Analysis Insights: {all_insights["fundamental"]}
            
            News Analysis Insights: {all_insights["news"]}
            
            Risk Assessment: {all_insights["risk"]}
            
            Market Context: {all_insights["market_context"]}
            
            Your summary should:
            1. Synthesize the key points from all analysis types
            2. Highlight the most important factors for investors to consider
            3. Identify any contradictions or alignments between different analyses
            4. Provide a balanced overall perspective on the stock
            5. Be comprehensive yet concise (around 250-300 words)
            
            Format your response as a well-organized, cohesive summary that could stand alone as a comprehensive analysis of the stock.
            """
            
            # Generate insights using Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=800,  # Allow for longer summary
                temperature=self.temperature,
                system="You are a comprehensive financial analyst providing balanced investment insights.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract insights from response
            summary = response.content[0].text
            
            return summary
        
        except Exception as e:
            logger.error(f"Error generating summary insights for {symbol}: {e}")
            return f"Unable to generate comprehensive summary for {symbol} due to an error."
