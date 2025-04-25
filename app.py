"""
Main Streamlit application for the Intelligent Financial Insights Platform.
"""
import logging
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
import json
from typing import Dict, List, Any, Optional, Tuple

# Import components
from agents.coordinator import AgentCoordinator
from data_collection.api_integration import FinancialDataCollector
from database.db_manager import create_db_and_tables
from ui.components.sidebar import create_sidebar
from visualization.dashboard import display_dashboard
from config.settings import DEFAULT_STOCKS, DEFAULT_TIME_PERIOD, APP_NAME, THEME_COLOR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize coordinator
coordinator = AgentCoordinator()

# Initialize data collector (for quick data access)
data_collector = FinancialDataCollector()

# Create database and tables
create_db_and_tables()

# Set page config
st.set_page_config(
    page_title=APP_NAME,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App state
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_data(symbol: str, period: str = DEFAULT_TIME_PERIOD) -> pd.DataFrame:
    """
    Get stock data for a symbol.
    
    Args:
        symbol: Stock symbol
        period: Time period
        
    Returns:
        DataFrame with stock data
    """
    try:
        data = data_collector.get_stock_historical_data(symbol, period)
        return data
    except Exception as e:
        logger.error(f"Error fetching stock data for {symbol}: {e}")
        st.error(f"Error fetching stock data for {symbol}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_company_info(symbol: str) -> Dict[str, Any]:
    """
    Get company information for a symbol.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Dictionary with company information
    """
    try:
        info = data_collector.get_company_data(symbol)
        return info
    except Exception as e:
        logger.error(f"Error fetching company info for {symbol}: {e}")
        st.error(f"Error fetching company info for {symbol}: {e}")
        return {}

@st.cache_data(ttl=900)  # Cache for 15 minutes
def get_analysis_results(symbol: str, period: str = DEFAULT_TIME_PERIOD) -> Dict[str, Any]:
    """
    Get complete analysis results for a symbol.
    
    Args:
        symbol: Stock symbol
        period: Time period
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Check if we have results in the cache
        cache_file = f"data/cache/{symbol}_{period}_results.json"
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        if os.path.exists(cache_file):
            # Check if cache is fresh (less than 15 minutes old)
            mod_time = os.path.getmtime(cache_file)
            if time.time() - mod_time < 900:  # 15 minutes
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        # Process the request through the coordinator
        results = coordinator.process_stock_request(symbol, period)
        
        # Save results to cache
        with open(cache_file, 'w') as f:
            json.dump(results, f)
        
        return results
    except Exception as e:
        logger.error(f"Error getting analysis results for {symbol}: {e}")
        st.error(f"Error getting analysis results for {symbol}: {e}")
        return {}

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_multiple_stocks_data(symbols: List[str], period: str = DEFAULT_TIME_PERIOD) -> Dict[str, pd.DataFrame]:
    """
    Get data for multiple stocks.
    
    Args:
        symbols: List of stock symbols
        period: Time period
        
    Returns:
        Dictionary of {symbol: DataFrame}
    """
    result = {}
    for symbol in symbols:
        data = get_stock_data(symbol, period)
        if not data.empty:
            result[symbol] = data
    return result

@st.cache_data(ttl=3600)  # Cache for 1 hour
def process_user_query(query: str, symbol: Optional[str] = None) -> str:
    """
    Process a user query.
    
    Args:
        query: User query
        symbol: Stock symbol
        
    Returns:
        Response to the query
    """
    try:
        # Process the query through the coordinator
        response = coordinator.process_user_query(query, symbol)
        return response.get("response", "Sorry, I couldn't process your query.")
    except Exception as e:
        logger.error(f"Error processing user query: {e}")
        return f"Error processing your query: {e}"

def main():
    """Main application function."""
    # Custom CSS
    st.markdown(f"""
    <style>
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {THEME_COLOR};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {THEME_COLOR};
        color: white;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'current_symbol' not in st.session_state:
        st.session_state.current_symbol = DEFAULT_STOCKS[0]
    
    if 'time_period' not in st.session_state:
        st.session_state.time_period = DEFAULT_TIME_PERIOD
    
    if 'comparison_symbols' not in st.session_state:
        st.session_state.comparison_symbols = []
    
    if 'show_comparison' not in st.session_state:
        st.session_state.show_comparison = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar
    symbol, period, comparison_symbols, show_comparison, uploaded_file = create_sidebar()
    
    # Update session state
    st.session_state.current_symbol = symbol
    st.session_state.time_period = period
    st.session_state.comparison_symbols = comparison_symbols
    st.session_state.show_comparison = show_comparison
    
    # Main content
    st.title(f"{APP_NAME}")
    
    # Display tabs
    main_tab, chat_tab = st.tabs(["Analysis Dashboard", "Investment Assistant"])
    
    # Analysis Dashboard Tab
    with main_tab:
        with st.spinner(f"Analyzing {symbol}..."):
            # Get analysis results
            results = get_analysis_results(symbol, period)
            
            # Get comparison data if requested
            comparison_data = None
            if show_comparison and comparison_symbols:
                all_symbols = [symbol] + comparison_symbols
                comparison_data = get_multiple_stocks_data(all_symbols, period)
            
            # Display dashboard
            if results:
                display_dashboard(results, comparison_data)
            else:
                st.error(f"No analysis results available for {symbol}")
    
    # Investment Assistant Tab
    with chat_tab:
        st.header("Investment Strategy Assistant")
        
        # Display chat interface
        st.write("Ask me anything about investing or about the current stock analysis.")
        
        # Context information
        with st.expander("About the Investment Assistant", expanded=False):
            st.write("""
            The Investment Strategy Assistant can answer questions about:
            
            - Technical analysis and indicators
            - Fundamental analysis and company financials
            - Market news and sentiment
            - Investment strategies and recommendations
            - Risk assessment and portfolio management
            - General investment concepts and terminology
            
            You can ask specific questions about the currently selected stock or general investment questions.
            """)
        
        # Query input
        query = st.text_input("Enter your question:", key="query_input")
        
        if st.button("Submit", key="submit_button"):
            if query:
                with st.spinner("Processing your question..."):
                    # Process the query
                    response = process_user_query(query, symbol)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "query": query,
                        "response": response,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                
                # Clear the input
                st.session_state.query_input = ""
        
        # Display chat history
        if st.session_state.chat_history:
            st.write("### Conversation History")
            
            for i, chat in enumerate(st.session_state.chat_history):
                # User message
                st.markdown(f"**You ({chat['timestamp']}):** {chat['query']}")
                
                # Assistant response
                st.markdown(f"**Assistant:** {chat['response']}")
                
                # Separator
                if i < len(st.session_state.chat_history) - 1:
                    st.markdown("---")

if __name__ == "__main__":
    main()
