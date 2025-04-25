"""
User Interaction Manager for handling natural language queries about investments.
"""
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from anthropic import Anthropic

from config.settings import ANTHROPIC_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from database.db_manager import UserQueryManager

# Set up logging
logger = logging.getLogger(__name__)

class UserInteractionManager:
    """
    Specialized Agent for handling user queries about investments.
    Uses Claude API to generate responses to natural language questions.
    """
    
    def __init__(self):
        """Initialize user interaction manager with Claude API client."""
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = LLM_MODEL
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
        self.query_manager = UserQueryManager()
    
    def process_query(self, query: str, context: Dict[str, Any]) -> str:
        """
        Process a user query about a stock or investment.
        
        Args:
            query: User's natural language query
            context: Context for the query (analysis results, etc.)
            
        Returns:
            Response to the query
        """
        logger.info(f"Processing user query: {query}")
        
        try:
            # Classify the query to determine how to handle it
            query_type = self._classify_query(query)
            
            # Generate response based on query type and context
            response = self._generate_response(query, query_type, context)
            
            # Save query and response to database
            symbol = context.get("symbol", None)
            self._save_query(query, response, symbol, context)
            
            return response
        
        except Exception as e:
            logger.error(f"Error processing user query: {e}")
            return f"I apologize, but I encountered an error processing your query: {str(e)}. Please try asking in a different way or check if the required data is available."
    
    def _classify_query(self, query: str) -> str:
        """
        Classify the type of user query.
        
        Args:
            query: User's natural language query
            
        Returns:
            Query type category
        """
        try:
            # Create prompt for Claude to classify the query
            prompt = f"""
            Classify the following investment query into one of these categories:
            - technical_analysis: Questions about technical indicators, chart patterns, price movements
            - fundamental_analysis: Questions about company financials, valuation, earnings
            - news_sentiment: Questions about news, market sentiment, or recent events
            - strategy_recommendation: Questions asking for specific investment advice or recommendations
            - risk_assessment: Questions about risk, volatility, or downside protection
            - general_information: General questions about the stock or company
            - comparison: Questions comparing this stock to others or to benchmarks
            - explanation: Questions asking to explain concepts or terms
            - other: Any other type of query
            
            Query: "{query}"
            
            Respond with only the category name.
            """
            
            # Generate classification using Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=20,  # Short response for classification
                temperature=0.1,  # Low temperature for consistency
                system="You are a financial query classifier. You categorize investment queries into predefined categories. Respond with only the category name.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract classification from response
            classification = response.content[0].text.strip().lower()
            
            # Ensure we get a valid category
            valid_categories = [
                "technical_analysis", "fundamental_analysis", "news_sentiment",
                "strategy_recommendation", "risk_assessment", "general_information",
                "comparison", "explanation", "other"
            ]
            
            if classification not in valid_categories:
                # Default to general_information if we get an invalid category
                classification = "general_information"
            
            return classification
        
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            return "general_information"  # Default if classification fails
    
    def _generate_response(
        self, query: str, query_type: str, context: Dict[str, Any]
    ) -> str:
        """
        Generate a response to the user query based on context.
        
        Args:
            query: User's natural language query
            query_type: Type of query (from classification)
            context: Context for the query (analysis results, etc.)
            
        Returns:
            Response to the query
        """
        try:
            # Extract relevant information from context based on query type
            relevant_context = self._extract_relevant_context(query_type, context)
            
            # Create prompt for Claude based on query type
            system_prompt = self._create_system_prompt(query_type)
            
            user_prompt = f"""
            The following is a user query about a stock or investment:
            
            User Query: "{query}"
            
            Query Type: {query_type}
            
            Here is the relevant context for answering the query:
            
            {json.dumps(relevant_context, indent=2)}
            
            Please provide a helpful, informative response to the user's query based on the context provided. Be conversational but precise, and stay focused on answering the specific question asked. If the context doesn't contain enough information to fully answer the query, state what isn't available and then provide the best possible answer with the information you have.
            
            If the query is asking for specific investment advice, include appropriate disclaimers about investment risk.
            """
            
            # Generate response using Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Extract response from Claude
            answer = response.content[0].text
            
            return answer
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error generating a response to your query. Please try asking in a different way or check if the required data is available."
    
    def _extract_relevant_context(
        self, query_type: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract relevant context for the query based on query type.
        
        Args:
            query_type: Type of query
            context: Full context
            
        Returns:
            Relevant subset of context
        """
        relevant_context = {}
        
        # Basic information common to all query types
        if "symbol" in context:
            relevant_context["symbol"] = context.get("symbol")
        
        if "company_info" in context:
            relevant_context["company_info"] = context.get("company_info")
        
        if "current_data" in context:
            relevant_context["current_data"] = context.get("current_data")
        
        # Add specific context based on query type
        if query_type == "technical_analysis":
            if "analysis" in context and "technical" in context["analysis"]:
                relevant_context["technical_analysis"] = context["analysis"]["technical"]
            
            if "analysis" in context and "patterns" in context["analysis"]:
                relevant_context["patterns"] = context["analysis"]["patterns"]
            
            if "insights" in context and "technical" in context["insights"]:
                relevant_context["technical_insights"] = context["insights"]["technical"]
        
        elif query_type == "fundamental_analysis":
            if "analysis" in context and "fundamental" in context["analysis"]:
                relevant_context["fundamental_analysis"] = context["analysis"]["fundamental"]
            
            if "insights" in context and "fundamental" in context["insights"]:
                relevant_context["fundamental_insights"] = context["insights"]["fundamental"]
        
        elif query_type == "news_sentiment":
            if "news" in context:
                relevant_context["news"] = context["news"]
            
            if "insights" in context and "news" in context["insights"]:
                relevant_context["news_insights"] = context["insights"]["news"]
        
        elif query_type == "strategy_recommendation":
            if "strategy" in context:
                relevant_context["strategy"] = context["strategy"]
            
            if "insights" in context and "summary" in context["insights"]:
                relevant_context["summary_insights"] = context["insights"]["summary"]
        
        elif query_type == "risk_assessment":
            if "analysis" in context and "volatility" in context["analysis"]:
                relevant_context["volatility"] = context["analysis"]["volatility"]
            
            if "insights" in context and "risk" in context["insights"]:
                relevant_context["risk_insights"] = context["insights"]["risk"]
            
            if "strategy" in context and "risk_management" in context["strategy"]:
                relevant_context["risk_management"] = context["strategy"]["risk_management"]
        
        elif query_type == "comparison":
            # Include comprehensive data for comparisons
            if "analysis" in context:
                relevant_context["analysis"] = context["analysis"]
            
            if "insights" in context:
                relevant_context["insights"] = context["insights"]
        
        # For general or other query types, include most of the context
        else:
            if "insights" in context and "summary" in context["insights"]:
                relevant_context["summary_insights"] = context["insights"]["summary"]
            
            if "strategy" in context and "overall" in context["strategy"]:
                relevant_context["overall_strategy"] = context["strategy"]["overall"]
        
        return relevant_context
    
    def _create_system_prompt(self, query_type: str) -> str:
        """
        Create a system prompt based on query type.
        
        Args:
            query_type: Type of query
            
        Returns:
            System prompt for Claude
        """
        base_prompt = "You are an intelligent financial assistant providing information and insights about stocks and investments."
        
        if query_type == "technical_analysis":
            return base_prompt + " You specialize in technical analysis, chart patterns, and price movements. Provide clear explanations of technical indicators and their implications."
        
        elif query_type == "fundamental_analysis":
            return base_prompt + " You specialize in fundamental analysis, company financials, and valuation metrics. Provide clear explanations of company performance and financial health."
        
        elif query_type == "news_sentiment":
            return base_prompt + " You specialize in analyzing news and market sentiment. Provide insights on how recent events might impact stock performance."
        
        elif query_type == "strategy_recommendation":
            return base_prompt + " You provide investment strategy recommendations based on comprehensive analysis. Always include appropriate disclaimers about investment risk."
        
        elif query_type == "risk_assessment":
            return base_prompt + " You specialize in risk assessment and volatility analysis. Provide clear explanations of risk factors and potential downside scenarios."
        
        elif query_type == "explanation":
            return base_prompt + " You excel at explaining financial concepts and terminology in clear, accessible language. Make complex ideas understandable without oversimplifying."
        
        elif query_type == "comparison":
            return base_prompt + " You specialize in comparative analysis of stocks, sectors, and benchmarks. Highlight key similarities and differences in a balanced way."
        
        else:  # general_information or other
            return base_prompt + " You provide helpful, informative responses to a wide range of investment queries. Balance technical accuracy with accessibility."
    
    def _save_query(
        self, query: str, response: str, symbol: Optional[str], context: Dict[str, Any]
    ) -> None:
        """
        Save query and response to database.
        
        Args:
            query: User's query
            response: Generated response
            symbol: Stock symbol
            context: Query context
        """
        try:
            # Convert context to JSON string (only save necessary context)
            query_context = {
                "timestamp": datetime.now().isoformat(),
                "query_type": self._classify_query(query)
            }
            
            if symbol:
                query_context["symbol"] = symbol
            
            context_json = json.dumps(query_context)
            
            # Save to database
            self.query_manager.create({
                "symbol": symbol,
                "query_text": query,
                "response_text": response,
                "query_context": context_json
            })
        
        except Exception as e:
            logger.error(f"Error saving query to database: {e}")
