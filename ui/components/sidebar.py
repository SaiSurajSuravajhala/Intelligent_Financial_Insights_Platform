"""
Sidebar components for the Streamlit application.
"""
import streamlit as st
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
import io

from config.settings import DEFAULT_STOCKS, DEFAULT_TIME_PERIOD
from data_collection.api_integration import FinancialDataCollector

# Initialize data collector for quick lookups
data_collector = FinancialDataCollector()

def create_sidebar() -> Tuple[str, str, List[str], bool, Optional[pd.DataFrame]]:
    """
    Create the sidebar for the application.
    
    Returns:
        Tuple of (selected_symbol, time_period, comparison_symbols, show_comparison, uploaded_file)
    """
    # Set up sidebar
    st.sidebar.title("Settings")
    
    # Symbol selection
    symbol = st.sidebar.selectbox(
        "Select Stock Symbol", 
        options=DEFAULT_STOCKS, 
        index=DEFAULT_STOCKS.index(st.session_state.get('current_symbol', DEFAULT_STOCKS[0]))
    )
    
    # Custom symbol input
    st.sidebar.markdown("---")
    st.sidebar.subheader("Or Enter Custom Symbol")
    custom_symbol = st.sidebar.text_input("Enter Stock Symbol:", "")
    
    if custom_symbol and custom_symbol.upper() != symbol:
        symbol = custom_symbol.upper()
    
    # Time period selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("Time Period")
    period_options = {
        "1 Week": "1w",
        "1 Month": "1m",
        "3 Months": "3m",
        "6 Months": "6m",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
        "Max": "max"
    }
    period_display = st.sidebar.select_slider(
        "Select Time Period:",
        options=list(period_options.keys()),
        value=next((k for k, v in period_options.items() if v == st.session_state.get('time_period', DEFAULT_TIME_PERIOD)), "1 Year")
    )
    period = period_options[period_display]
    
    # Comparison stocks
    st.sidebar.markdown("---")
    st.sidebar.subheader("Comparison")
    show_comparison = st.sidebar.checkbox("Compare with other stocks", value=st.session_state.get('show_comparison', False))
    
    comparison_symbols = []
    if show_comparison:
        # Multi-select for comparison symbols
        all_symbols = set(DEFAULT_STOCKS)
        all_symbols.discard(symbol)  # Remove current symbol
        
        comparison_symbols = st.sidebar.multiselect(
            "Select symbols to compare:",
            options=sorted(list(all_symbols)),
            default=st.session_state.get('comparison_symbols', [])[:3]  # Limit to first 3
        )
        
        # Limit to 5 comparison symbols
        if len(comparison_symbols) > 5:
            st.sidebar.warning("Limited to 5 comparison stocks")
            comparison_symbols = comparison_symbols[:5]
        
        # Custom comparison symbol
        custom_comparison = st.sidebar.text_input("Add custom symbol for comparison:", "")
        if custom_comparison and custom_comparison.upper() != symbol and custom_comparison.upper() not in comparison_symbols:
            comparison_symbols.append(custom_comparison.upper())
    
    # Upload custom data
    st.sidebar.markdown("---")
    st.sidebar.subheader("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload custom stock data (CSV):", type=["csv"])
    
    processed_file = None
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            content = uploaded_file.read()
            
            # Convert to DataFrame
            data = pd.read_csv(io.BytesIO(content))
            
            # Check if data has the required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                st.sidebar.error(f"Missing required columns: {', '.join(missing_cols)}")
                st.sidebar.info("Required columns: date, open, high, low, close, volume")
            else:
                # Try to parse the date column
                try:
                    data['date'] = pd.to_datetime(data['date'])
                    data.set_index('date', inplace=True)
                    data.sort_index(inplace=True)
                    
                    # Add symbol column if not present
                    if 'symbol' not in data.columns:
                        data['symbol'] = symbol
                    
                    st.sidebar.success(f"Data loaded successfully: {len(data)} rows")
                    processed_file = data
                except Exception as e:
                    st.sidebar.error(f"Error parsing date column: {e}")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
    
    # Company information
    if symbol:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Company Information")
        
        try:
            company_info = data_collector.get_company_data(symbol)
            
            if company_info:
                st.sidebar.write(f"**Name:** {company_info.get('name', 'N/A')}")
                st.sidebar.write(f"**Sector:** {company_info.get('sector', 'N/A')}")
                st.sidebar.write(f"**Industry:** {company_info.get('industry', 'N/A')}")
            else:
                st.sidebar.info(f"No company information available for {symbol}")
        except Exception as e:
            st.sidebar.warning(f"Error fetching company information: {e}")
    
    # About section
    st.sidebar.markdown("---")
    with st.sidebar.expander("About", expanded=False):
        st.write("""
        **Intelligent Financial Insights Platform**
        
        A comprehensive tool for financial market analysis and investment strategy.
        
        Features:
        - Technical and fundamental analysis
        - News sentiment analysis
        - Investment strategy recommendations
        - Interactive Q&A with the Investment Assistant
        
        Built with Python, Streamlit, and LangChain.
        """)
    
    return symbol, period, comparison_symbols, show_comparison, processed_file
