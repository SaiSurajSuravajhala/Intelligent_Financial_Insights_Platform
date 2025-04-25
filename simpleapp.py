"""
Minimal Streamlit application for the Intelligent Financial Insights Platform.
This is a temporary file to help diagnose and fix the recursion error.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Configure the page
st.set_page_config(
    page_title="Intelligent Financial Insights Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create any needed directories
os.makedirs("data/cache", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Basic styling
st.markdown("""
<style>
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
h1, h2, h3, h4, h5, h6 {
    color: #4A90E2;
}
</style>
""", unsafe_allow_html=True)

# Main app layout
st.title("Intelligent Financial Insights Platform")

# Sidebar
st.sidebar.title("Settings")
symbol = st.sidebar.selectbox("Select Stock Symbol", ["AAPL", "MSFT", "GOOGL", "AMZN", "META"])
period = st.sidebar.select_slider(
    "Select Time Period:",
    options=["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"]
)

# Generate mock data for demonstration
days = {"1 Week": 7, "1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365}[period]
dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
mock_data = pd.DataFrame({
    'open': np.random.normal(100, 5, days),
    'high': np.random.normal(105, 5, days),
    'low': np.random.normal(95, 5, days),
    'close': np.random.normal(102, 5, days),
    'volume': np.random.normal(1000000, 200000, days)
}, index=dates)

# Main tabs
tabs = st.tabs(["Analysis Dashboard", "Investment Assistant"])

with tabs[0]:
    st.header(f"{symbol} Analysis")
    
    # Current price info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Current Price",
            value=f"${mock_data['close'].iloc[-1]:.2f}",
            delta=f"{(mock_data['close'].iloc[-1] - mock_data['close'].iloc[-2]):.2f}"
        )
    
    with col2:
        st.metric(
            label="Volume",
            value=f"{mock_data['volume'].iloc[-1]:.0f}"
        )
    
    with col3:
        st.metric(
            label="30-Day Change",
            value=f"{((mock_data['close'].iloc[-1] / mock_data['close'].iloc[0]) - 1) * 100:.2f}%"
        )
    
    # Price chart
    st.subheader("Price History")
    st.line_chart(mock_data['close'])
    
    # Data table
    st.subheader("Recent Data")
    st.dataframe(mock_data.tail(10))

with tabs[1]:
    st.header("Investment Assistant")
    st.write("This feature would allow you to ask questions about stocks and investments.")
    
    user_query = st.text_input("Ask a question about investing or about the current stock:")
    
    if st.button("Submit"):
        st.write(f"You asked: {user_query}")
        st.write(f"This is a placeholder response for: {user_query}")
        st.info("The full version would use AI to answer your questions.")

# Footer
st.markdown("---")
st.caption(f"Intelligent Financial Insights Platform | Data as of {datetime.now().strftime('%Y-%m-%d')}")