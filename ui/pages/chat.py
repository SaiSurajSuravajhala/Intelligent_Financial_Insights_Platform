"""
Chat page component for the Streamlit application.
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable

from ui.components.widgets import chat_message
from agents.user_interaction import UserInteractionManager

# Initialize user interaction manager
user_interaction_manager = UserInteractionManager()

def render_chat_page(
    symbol: str,
    results: Dict[str, Any]
) -> None:
    """
    Render the chat interface for investment assistant.
    
    Args:
        symbol: Current stock symbol
        results: Analysis results for context
    """
    st.markdown(f"# Investment Strategy Assistant")
    
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
    
    # Initialize chat history in session state if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        chat_message(
            content=message["content"],
            is_user=message["is_user"]
        )
    
    # Chat input
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Ask a question:", key="chat_input", height=100)
        submit_button = st.form_submit_button("Send")
    
    # Process the message when the form is submitted
    if submit_button and user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "content": user_input,
            "is_user": True,
            "timestamp": datetime.now().isoformat()
        })
        
        # Get response
        with st.spinner("Processing your question..."):
            response = process_user_query(user_input, symbol, results)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({
            "content": response,
            "is_user": False,
            "timestamp": datetime.now().isoformat()
        })
        
        # Rerun to update the display
        st.rerun()

def process_user_query(
    query: str, symbol: str, context: Dict[str, Any]
) -> str:
    """
    Process a user query.
    
    Args:
        query: User query
        symbol: Stock symbol
        context: Context information
        
    Returns:
        Response to the query
    """
    try:
        # Use the user interaction manager to process the query
        response = user_interaction_manager.process_query(query, context)
        return response
    except Exception as e:
        # Handle exceptions
        st.error(f"Error processing query: {e}")
        return f"I'm sorry, but I encountered an error while processing your question: {str(e)}. Please try asking in a different way."

def format_query_suggestions() -> None:
    """Display query suggestions."""
    st.markdown("### Example Questions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - What do the technical indicators suggest for this stock?
        - Explain the current support and resistance levels.
        - What does the MACD indicator tell us right now?
        - How volatile is this stock compared to the market?
        - What's the risk assessment for this stock?
        """)
    
    with col2:
        st.markdown("""
        - What do the financial metrics indicate about company health?
        - How does the P/E ratio compare to industry average?
        - What's the recent news sentiment about this company?
        - Would you recommend buying this stock now?
        - What's a good entry point for this stock?
        """)
