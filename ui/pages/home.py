"""
Home page component for the Streamlit application.
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from visualization.charts import (
    create_candlestick_chart, 
    create_comparison_chart, 
    create_performance_chart
)
from ui.components.widgets import (
    metric_card, 
    info_card, 
    expandable_section, 
    alert
)
from utils.helpers import format_currency, format_percentage, format_large_number

def render_home_page(
    symbol: str,
    current_data: Dict[str, Any],
    company_info: Dict[str, Any],
    insights: Dict[str, Any],
    comparison_data: Optional[Dict[str, pd.DataFrame]] = None
) -> None:
    """
    Render the home page content.
    
    Args:
        symbol: Stock symbol
        current_data: Current price data
        company_info: Company information
        insights: Generated insights
        comparison_data: Optional comparison data for multiple symbols
    """
    # Welcome section
    st.markdown(f"# Welcome to the Intelligent Financial Insights Platform")
    
    # Current selection info
    name = company_info.get("name", symbol)
    st.markdown(f"## Currently Analyzing: **{name}** ({symbol})")
    
    # Quick stats section
    st.markdown("### Current Stats")
    
    # Row of metric cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        price = current_data.get("close", 0)
        change_pct = current_data.get("change_pct", 0)
        metric_card(
            title="Current Price",
            value=format_currency(price),
            delta=format_percentage(change_pct / 100) if change_pct else None,
            delta_color="normal" if change_pct >= 0 else "inverse"
        )
    
    with col2:
        # Show market cap if available
        market_cap = company_info.get("market_cap", 0)
        metric_card(
            title="Market Cap",
            value=format_large_number(market_cap),
            help_text="Market capitalization of the company"
        )
    
    with col3:
        # Show volume
        volume = current_data.get("volume", 0)
        metric_card(
            title="Volume",
            value=format_large_number(volume),
            help_text="Trading volume"
        )
    
    with col4:
        # Show sector and industry
        sector = company_info.get("sector", "Unknown")
        industry = company_info.get("industry", "Unknown")
        st.markdown(f"**Sector:** {sector}")
        st.markdown(f"**Industry:** {industry}")
    
    # Summary insights
    st.markdown("### Summary Insights")
    summary = insights.get("summary", "No summary insights available.")
    st.markdown(summary)
    
    # Price chart section
    st.markdown("### Price History")
    
    # Check if we have historical data for comparison
    if comparison_data and len(comparison_data) > 1:
        # Show a tab for single stock and comparison chart
        tab1, tab2 = st.tabs(["Price Chart", "Comparison"])
        
        with tab1:
            # Get the main stock data
            if symbol in comparison_data:
                main_data = comparison_data[symbol]
                fig = create_candlestick_chart(main_data, f"{symbol} Price History")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No historical data available for {symbol}")
        
        with tab2:
            # Comparison chart options
            chart_type = st.radio(
                "Comparison Type",
                options=["Price Change %", "Absolute Price"],
                horizontal=True
            )
            
            normalize = (chart_type == "Price Change %")
            
            # Create comparison chart
            comp_chart = create_comparison_chart(
                comparison_data, 
                column='close', 
                normalize=normalize,
                title="Price Comparison"
            )
            st.plotly_chart(comp_chart, use_container_width=True)
            
            # Add a small description
            st.markdown("**Note:** This chart shows how the selected stocks have performed over the same time period. Use the sidebar to add more stocks for comparison.")
    
    else:
        # Show only the main stock chart
        if comparison_data and symbol in comparison_data:
            main_data = comparison_data[symbol]
            fig = create_candlestick_chart(main_data, f"{symbol} Price History")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No historical data available for {symbol}")
    
    # Quick overview of sections
    st.markdown("### Quick Navigation")
    st.markdown("Use the tabs above to navigate to different analysis sections:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        info_card(
            title="Analysis Dashboard", 
            content="Comprehensive technical and fundamental analysis, news sentiment, and performance metrics.",
            icon="📊"
        )
        
        info_card(
            title="Strategy", 
            content="Investment strategy recommendations, risk assessment, and entry/exit points.",
            icon="🎯"
        )
    
    with col2:
        info_card(
            title="News & Sentiment", 
            content="Latest news articles, sentiment analysis, and market impact assessment.",
            icon="📰"
        )
        
        info_card(
            title="Investment Assistant", 
            content="Ask questions and get personalized insights about this stock or general investment topics.",
            icon="🤖"
        )
    
    # How to use section
    with st.expander("How to Use This Platform", expanded=False):
        st.markdown("""
        ## How to Use the Intelligent Financial Insights Platform
        
        1. **Select a Stock**: Use the sidebar to select a stock symbol or enter a custom symbol.
        
        2. **Choose Time Period**: Select the time period for analysis using the slider in the sidebar.
        
        3. **Compare Stocks**: Enable comparison mode in the sidebar to compare multiple stocks.
        
        4. **Explore Analysis**: Navigate through the tabs to view different types of analysis:
           - **Analysis Dashboard**: Technical and fundamental analysis
           - **News & Sentiment**: Latest news and sentiment analysis
           - **Strategy**: Investment strategy recommendations
           
        5. **Ask Questions**: Use the Investment Assistant tab to ask questions about the stock or general investment topics.
        
        6. **Upload Custom Data**: You can upload your own CSV file with stock data using the sidebar.
        
        Happy investing!
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("##### Intelligent Financial Insights Platform | Data updated as of " + 
               datetime.now().strftime("%Y-%m-%d"))
