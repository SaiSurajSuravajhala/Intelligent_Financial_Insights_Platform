"""
Strategy Recommendation Engine for financial analysis.
Generates tailored investment recommendations.
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from anthropic import Anthropic
import json

from config.settings import ANTHROPIC_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

# Set up logging
logger = logging.getLogger(__name__)

class StrategyRecommendationEngine:
    """
    Specialized Agent for generating investment strategy recommendations.
    Uses Claude API to generate tailored investment advice.
    """
    
    def __init__(self):
        """Initialize strategy engine with Claude API client."""
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = LLM_MODEL
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
    
    def generate_strategy(
        self, analysis_results: Dict[str, Any], insights: Dict[str, Any], symbol: str
    ) -> Dict[str, Any]:
        """
        Generate investment strategy recommendations.
        
        Args:
            analysis_results: Results from data analysis
            insights: Generated insights
            symbol: Stock symbol
            
        Returns:
            Dictionary with strategy recommendations
        """
        logger.info(f"Generating strategy for {symbol}")
        
        strategy = {}
        
        try:
            # Generate strategies for different time horizons
            strategy["short_term"] = self.generate_short_term_strategy(analysis_results, insights, symbol)
            strategy["medium_term"] = self.generate_medium_term_strategy(analysis_results, insights, symbol)
            strategy["long_term"] = self.generate_long_term_strategy(analysis_results, insights, symbol)
            
            # Generate entry/exit points
            strategy["entry_exit"] = self.generate_entry_exit_points(analysis_results, symbol)
            
            # Generate risk management recommendations
            strategy["risk_management"] = self.generate_risk_management(analysis_results, insights, symbol)
            
            # Generate overall recommendation
            strategy["overall"] = self.generate_overall_recommendation(strategy, analysis_results, insights, symbol)
            
            return strategy
        
        except Exception as e:
            logger.error(f"Error generating strategy for {symbol}: {e}")
            return {
                "error": str(e),
                "overall": {
                    "recommendation": "HOLD",
                    "confidence": 0.5,
                    "reasoning": f"Unable to generate complete strategy for {symbol} due to an error."
                }
            }
    
    def generate_short_term_strategy(
        self, analysis_results: Dict[str, Any], insights: Dict[str, Any], symbol: str
    ) -> Dict[str, Any]:
        """
        Generate short-term strategy recommendation (days to weeks).
        
        Args:
            analysis_results: Results from data analysis
            insights: Generated insights
            symbol: Stock symbol
            
        Returns:
            Dictionary with short-term strategy
        """
        try:
            # Extract relevant data for short-term strategy
            technical = analysis_results.get("technical", {})
            patterns = analysis_results.get("patterns", {})
            volatility = analysis_results.get("volatility", {})
            
            # Get insights
            technical_insights = insights.get("technical", "")
            news_insights = insights.get("news", "")
            
            # Create prompt for Claude
            prompt = f"""
            You are an investment strategist specializing in short-term trading strategies. Generate a detailed short-term investment strategy (days to weeks) for {symbol} based on the following data:
            
            Technical Analysis: {json.dumps(technical, indent=2)}
            
            Chart Patterns: {json.dumps(patterns, indent=2)}
            
            Volatility Analysis: {json.dumps(volatility, indent=2)}
            
            Technical Insights: {technical_insights}
            
            News Insights: {news_insights}
            
            Your strategy recommendation should include:
            1. A clear BUY, SELL, or HOLD recommendation with confidence level (0-1)
            2. Specific reasoning based on technical indicators, patterns, and news
            3. Key technical levels to watch (support/resistance)
            4. Potential catalysts in the short term
            5. Risk factors specific to this short-term outlook
            
            Format your response as JSON with the following structure:
            {{
                "recommendation": "BUY/SELL/HOLD",
                "confidence": 0.X,
                "reasoning": "Detailed reasoning...",
                "key_levels": {{
                    "support": [level1, level2],
                    "resistance": [level1, level2]
                }},
                "catalysts": ["catalyst1", "catalyst2"],
                "risk_factors": ["risk1", "risk2"]
            }}
            
            Ensure your recommendation is well-justified and actionable for a short-term trader.
            """
            
            # Generate strategy using Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system="You are an investment strategist specializing in short-term trading strategies.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract strategy from response
            strategy_text = response.content[0].text
            
            # Parse JSON response
            # Find JSON content (it might be wrapped in ```json or similar)
            strategy_json = self._extract_json(strategy_text)
            
            return strategy_json
        
        except Exception as e:
            logger.error(f"Error generating short-term strategy for {symbol}: {e}")
            return {
                "recommendation": "HOLD",
                "confidence": 0.5,
                "reasoning": f"Unable to generate short-term strategy for {symbol} due to an error."
            }
    
    def generate_medium_term_strategy(
        self, analysis_results: Dict[str, Any], insights: Dict[str, Any], symbol: str
    ) -> Dict[str, Any]:
        """
        Generate medium-term strategy recommendation (weeks to months).
        
        Args:
            analysis_results: Results from data analysis
            insights: Generated insights
            symbol: Stock symbol
            
        Returns:
            Dictionary with medium-term strategy
        """
        try:
            # Extract relevant data for medium-term strategy
            technical = analysis_results.get("technical", {})
            performance = analysis_results.get("performance", {})
            fundamental = analysis_results.get("fundamental", {})
            
            # Get insights
            technical_insights = insights.get("technical", "")
            fundamental_insights = insights.get("fundamental", "")
            market_context = insights.get("market_context", "")
            
            # Create prompt for Claude
            prompt = f"""
            You are an investment strategist specializing in medium-term investment strategies. Generate a detailed medium-term investment strategy (weeks to months) for {symbol} based on the following data:
            
            Technical Analysis: {json.dumps(technical, indent=2)}
            
            Performance Metrics: {json.dumps(performance, indent=2)}
            
            Fundamental Analysis: {json.dumps(fundamental, indent=2)}
            
            Technical Insights: {technical_insights}
            
            Fundamental Insights: {fundamental_insights}
            
            Market Context: {market_context}
            
            Your strategy recommendation should include:
            1. A clear BUY, SELL, or HOLD recommendation with confidence level (0-1)
            2. Reasoning that balances both technical and fundamental factors
            3. Key price targets for the medium term
            4. Potential catalysts to watch for
            5. Sector trends that might impact this stock
            6. Risk factors for this medium-term outlook
            
            Format your response as JSON with the following structure:
            {{
                "recommendation": "BUY/SELL/HOLD",
                "confidence": 0.X,
                "reasoning": "Detailed reasoning...",
                "price_targets": {{
                    "upper": X.XX,
                    "lower": X.XX
                }},
                "catalysts": ["catalyst1", "catalyst2"],
                "sector_trends": ["trend1", "trend2"],
                "risk_factors": ["risk1", "risk2"]
            }}
            
            Ensure your recommendation is well-justified and actionable for a medium-term investor.
            """
            
            # Generate strategy using Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system="You are an investment strategist specializing in medium-term investment strategies.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract strategy from response
            strategy_text = response.content[0].text
            
            # Parse JSON response
            strategy_json = self._extract_json(strategy_text)
            
            return strategy_json
        
        except Exception as e:
            logger.error(f"Error generating medium-term strategy for {symbol}: {e}")
            return {
                "recommendation": "HOLD",
                "confidence": 0.5,
                "reasoning": f"Unable to generate medium-term strategy for {symbol} due to an error."
            }
    
    def generate_long_term_strategy(
        self, analysis_results: Dict[str, Any], insights: Dict[str, Any], symbol: str
    ) -> Dict[str, Any]:
        """
        Generate long-term strategy recommendation (months to years).
        
        Args:
            analysis_results: Results from data analysis
            insights: Generated insights
            symbol: Stock symbol
            
        Returns:
            Dictionary with long-term strategy
        """
        try:
            # Extract relevant data for long-term strategy
            fundamental = analysis_results.get("fundamental", {})
            performance = analysis_results.get("performance", {})
            
            # Get insights
            fundamental_insights = insights.get("fundamental", "")
            market_context = insights.get("market_context", "")
            
            # Create prompt for Claude
            prompt = f"""
            You are an investment strategist specializing in long-term investment strategies. Generate a detailed long-term investment strategy (months to years) for {symbol} based on the following data:
            
            Fundamental Analysis: {json.dumps(fundamental, indent=2)}
            
            Performance Metrics: {json.dumps(performance, indent=2)}
            
            Fundamental Insights: {fundamental_insights}
            
            Market Context: {market_context}
            
            Your strategy recommendation should include:
            1. A clear BUY, SELL, or HOLD recommendation with confidence level (0-1)
            2. Reasoning focused primarily on fundamental factors and long-term industry trends
            3. Long-term growth potential assessment
            4. Competitive positioning in the industry
            5. Key long-term catalysts or headwinds
            6. Valuation assessment for long-term investment
            
            Format your response as JSON with the following structure:
            {{
                "recommendation": "BUY/SELL/HOLD",
                "confidence": 0.X,
                "reasoning": "Detailed reasoning...",
                "growth_potential": "High/Medium/Low",
                "competitive_position": "Strong/Moderate/Weak",
                "long_term_catalysts": ["catalyst1", "catalyst2"],
                "long_term_headwinds": ["headwind1", "headwind2"],
                "valuation_assessment": "Overvalued/Fairly Valued/Undervalued"
            }}
            
            Ensure your recommendation is well-justified and actionable for a long-term investor.
            """
            
            # Generate strategy using Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system="You are an investment strategist specializing in long-term investment strategies.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract strategy from response
            strategy_text = response.content[0].text
            
            # Parse JSON response
            strategy_json = self._extract_json(strategy_text)
            
            return strategy_json
        
        except Exception as e:
            logger.error(f"Error generating long-term strategy for {symbol}: {e}")
            return {
                "recommendation": "HOLD",
                "confidence": 0.5,
                "reasoning": f"Unable to generate long-term strategy for {symbol} due to an error."
            }
    
    def generate_entry_exit_points(
        self, analysis_results: Dict[str, Any], symbol: str
    ) -> Dict[str, Any]:
        """
        Generate entry and exit points recommendation.
        
        Args:
            analysis_results: Results from data analysis
            symbol: Stock symbol
            
        Returns:
            Dictionary with entry/exit points
        """
        try:
            # Extract relevant data for entry/exit points
            technical = analysis_results.get("technical", {})
            basic_stats = analysis_results.get("basic_stats", {})
            volatility = analysis_results.get("volatility", {})
            
            # Get current price if available
            current_price = 0
            if "all" in basic_stats:
                current_price = basic_stats["all"].get("current_price", 0)
            
            # Create prompt for Claude
            prompt = f"""
            You are an investment strategist specializing in entry and exit points for stock trades. Generate detailed entry and exit point recommendations for {symbol} based on the following data:
            
            Technical Analysis: {json.dumps(technical, indent=2)}
            
            Basic Stats: {json.dumps(basic_stats, indent=2)}
            
            Volatility Analysis: {json.dumps(volatility, indent=2)}
            
            Current Price: {current_price}
            
            Your entry/exit recommendation should include:
            1. Specific price points for entry (buying)
            2. Specific price points for exit (selling for profit)
            3. Stop-loss recommendations
            4. Reasoning behind each price point
            5. Time frames associated with these recommendations
            
            Format your response as JSON with the following structure:
            {{
                "entry_points": [
                    {{
                        "price": X.XX,
                        "reasoning": "Reasoning...",
                        "time_frame": "short/medium/long"
                    }},
                    ...
                ],
                "exit_points": [
                    {{
                        "price": X.XX,
                        "reasoning": "Reasoning...",
                        "time_frame": "short/medium/long"
                    }},
                    ...
                ],
                "stop_loss": {{
                    "price": X.XX,
                    "reasoning": "Reasoning..."
                }}
            }}
            
            Ensure your recommendations are specific, well-justified, and based on technical analysis principles.
            """
            
            # Generate entry/exit points using Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system="You are an investment strategist specializing in entry and exit points for stock trades.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract entry/exit points from response
            entry_exit_text = response.content[0].text
            
            # Parse JSON response
            entry_exit_json = self._extract_json(entry_exit_text)
            
            return entry_exit_json
        
        except Exception as e:
            logger.error(f"Error generating entry/exit points for {symbol}: {e}")
            return {
                "entry_points": [{"price": 0, "reasoning": "Unable to determine entry points", "time_frame": "medium"}],
                "exit_points": [{"price": 0, "reasoning": "Unable to determine exit points", "time_frame": "medium"}],
                "stop_loss": {"price": 0, "reasoning": "Unable to determine stop loss"}
            }
    
    def generate_risk_management(
        self, analysis_results: Dict[str, Any], insights: Dict[str, Any], symbol: str
    ) -> Dict[str, Any]:
        """
        Generate risk management recommendations.
        
        Args:
            analysis_results: Results from data analysis
            insights: Generated insights
            symbol: Stock symbol
            
        Returns:
            Dictionary with risk management recommendations
        """
        try:
            # Extract relevant data for risk management
            volatility = analysis_results.get("volatility", {})
            performance = analysis_results.get("performance", {})
            
            # Get insights
            risk_insights = insights.get("risk", "")
            
            # Create prompt for Claude
            prompt = f"""
            You are a risk management specialist in investment strategies. Generate detailed risk management recommendations for {symbol} based on the following data:
            
            Volatility Analysis: {json.dumps(volatility, indent=2)}
            
            Performance Metrics: {json.dumps(performance, indent=2)}
            
            Risk Insights: {risk_insights}
            
            Your risk management recommendation should include:
            1. Overall risk assessment (low, medium, high)
            2. Appropriate position sizing based on risk level
            3. Diversification recommendations
            4. Hedging strategies if appropriate
            5. Risk mitigation tactics specific to this stock
            
            Format your response as JSON with the following structure:
            {{
                "risk_level": "low/medium/high",
                "risk_score": X.X (1-10 scale),
                "position_sizing": "X% of portfolio maximum",
                "diversification": "Recommendations...",
                "hedging_strategies": ["strategy1", "strategy2"],
                "risk_mitigation_tactics": ["tactic1", "tactic2"]
            }}
            
            Ensure your recommendations are specific, practical, and tailored to this particular stock's risk profile.
            """
            
            # Generate risk management using Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system="You are a risk management specialist in investment strategies.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract risk management from response
            risk_text = response.content[0].text
            
            # Parse JSON response
            risk_json = self._extract_json(risk_text)
            
            return risk_json
        
        except Exception as e:
            logger.error(f"Error generating risk management for {symbol}: {e}")
            return {
                "risk_level": "medium",
                "risk_score": 5.0,
                "position_sizing": "Standard position sizing recommended",
                "diversification": "Standard diversification recommended",
                "hedging_strategies": ["Unable to determine specific hedging strategies"],
                "risk_mitigation_tactics": ["Unable to determine specific risk mitigation tactics"]
            }
    
    def generate_overall_recommendation(
        self, strategies: Dict[str, Any], analysis_results: Dict[str, Any], 
        insights: Dict[str, Any], symbol: str
    ) -> Dict[str, Any]:
        """
        Generate overall investment recommendation.
        
        Args:
            strategies: Previously generated strategies
            analysis_results: Results from data analysis
            insights: Generated insights
            symbol: Stock symbol
            
        Returns:
            Dictionary with overall recommendation
        """
        try:
            # Extract individual strategies
            short_term = strategies.get("short_term", {})
            medium_term = strategies.get("medium_term", {})
            long_term = strategies.get("long_term", {})
            risk_management = strategies.get("risk_management", {})
            
            # Get insights summary
            summary_insights = insights.get("summary", "")
            
            # Create prompt for Claude
            prompt = f"""
            You are a senior investment strategist responsible for providing overall investment recommendations. Generate a comprehensive final recommendation for {symbol} based on the following strategies and insights:
            
            Short-term Strategy: {json.dumps(short_term, indent=2)}
            
            Medium-term Strategy: {json.dumps(medium_term, indent=2)}
            
            Long-term Strategy: {json.dumps(long_term, indent=2)}
            
            Risk Management: {json.dumps(risk_management, indent=2)}
            
            Summary Insights: {summary_insights}
            
            Your overall recommendation should include:
            1. A final BUY, SELL, or HOLD recommendation with confidence level (0-1)
            2. Comprehensive reasoning that synthesizes short, medium, and long-term perspectives
            3. For which type of investor this stock is most appropriate
            4. Key price targets (entry, exit, stop-loss)
            5. Time horizon recommendation
            6. Primary catalysts and risk factors
            
            Format your response as JSON with the following structure:
            {{
                "recommendation": "BUY/SELL/HOLD",
                "confidence": 0.X,
                "reasoning": "Detailed reasoning...",
                "investor_suitability": ["Growth", "Value", "Income", etc.],
                "time_horizon": "Short/Medium/Long",
                "price_targets": {{
                    "entry": X.XX,
                    "exit": X.XX,
                    "stop_loss": X.XX
                }},
                "primary_catalysts": ["catalyst1", "catalyst2"],
                "primary_risks": ["risk1", "risk2"],
                "strategy_summary": "Concise strategy summary..."
            }}
            
            Ensure your recommendation is comprehensive, balanced, and integrates all timeframes and analyses.
            """
            
            # Generate overall recommendation using Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system="You are a senior investment strategist responsible for providing overall investment recommendations.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract overall recommendation from response
            overall_text = response.content[0].text
            
            # Parse JSON response
            overall_json = self._extract_json(overall_text)
            
            return overall_json
        
        except Exception as e:
            logger.error(f"Error generating overall recommendation for {symbol}: {e}")
            return {
                "recommendation": "HOLD",
                "confidence": 0.5,
                "reasoning": f"Unable to generate complete recommendation for {symbol} due to an error.",
                "investor_suitability": ["Neutral"],
                "time_horizon": "Medium",
                "price_targets": {
                    "entry": 0,
                    "exit": 0,
                    "stop_loss": 0
                },
                "primary_catalysts": ["Unable to determine specific catalysts"],
                "primary_risks": ["Unable to determine specific risks"],
                "strategy_summary": "Insufficient data for detailed strategy"
            }
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from text response.
        
        Args:
            text: Text possibly containing JSON
            
        Returns:
            Parsed JSON as dict
        """
        try:
            # Try to find JSON content which might be wrapped in markdown code blocks
            if "```json" in text:
                json_start = text.find("```json") + 7
                json_end = text.find("```", json_start)
                json_str = text[json_start:json_end].strip()
            elif "```" in text:
                json_start = text.find("```") + 3
                json_end = text.find("```", json_start)
                json_str = text[json_start:json_end].strip()
            else:
                # Assume the entire text is JSON
                json_str = text
            
            # Parse JSON
            return json.loads(json_str)
        
        except Exception as e:
            logger.error(f"Error extracting JSON from response: {e}")
            # Try a more aggressive approach to find anything that looks like JSON
            try:
                # Look for { and } brackets
                json_start = text.find("{")
                json_end = text.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = text[json_start:json_end]
                    return json.loads(json_str)
            except:
                pass
            
            # Return an empty dict if no JSON could be extracted
            return {}
