# Intelligent Financial Insights Platform

A comprehensive financial analysis tool that combines technical analysis, AI-powered insights, and document-aware investment recommendations.

![Platform Banner](https://example.com/banner.png)
-[Follow the link to test my application in detailed](https://intelligentfinancialinsightsplatform-azsg6qufwc2bzi2rszbrjz.streamlit.app/)

## Overview

The Intelligent Financial Insights Platform is a robust Streamlit-based web application designed to provide comprehensive stock analysis and investment insights. It combines powerful technical analysis tools with AI-driven recommendations and document-aware context to help users make better investment decisions.

## Key Features

### 1. Multi-Market Stock Analysis
- Supports stocks from multiple international markets:
  - US (default)
  - India (BSE/NSE)
  - UK (London Stock Exchange)
  - Canada (TSX)
  - Australia (ASX)
  - Germany (Frankfurt)

### 2. Advanced Technical Analysis
- **Interactive Price Charts**: Candlestick charts with adjustable time periods
- **Technical Indicators**:
  - Moving Averages (20, 50, 200-day)
  - Relative Strength Index (RSI)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - ADX (Average Directional Index)
  - Stochastic Oscillator
  - Ichimoku Cloud
- **Pattern Detection**:
  - Support and resistance levels
  - Trend direction and strength analysis
  - Overbought/Oversold conditions
- **Performance Metrics**:
  - Period returns (daily, weekly, monthly, quarterly, yearly)
  - Volatility analysis
  - Drawdown calculations
  - Risk-adjusted returns (Sharpe, Sortino)

### 3. AI-Powered Investment Assistant
- **Claude API Integration**: Leverages Anthropic's Claude for natural language understanding
- **Context-Aware Responses**: Considers both technical data and uploaded documents
- **Real-Time Data Analysis**: Processes the latest market data for accurate insights
- **Personalized Recommendations**: Investment advice tailored to the selected stock

### 4. Document Processing (RAG Implementation)
- **Document Upload**: Support for PDF and DOCX financial documents
- **Intelligent Analysis**: Extracts key information from financial reports
- **Context Integration**: Uses document insights to enhance investment recommendations
- **Financial Metrics Extraction**: Automatically identifies key metrics from documents

### 5. User-Friendly Interface
- **Personalized Favorites**: Save and manage favorite stocks
- **Multi-Tab Interface**: Separate tabs for analysis and investment assistance
- **Interactive Visualizations**: Dynamic charts and indicators
- **Learning Resources**: Built-in educational content on stock market fundamentals

## Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Database**: SQLite (via SQLModel)
- **APIs**:
  - Alpha Vantage (market data)
  - Finnhub (news and market information)
  - Anthropic Claude (AI assistant)
- **Document Processing**:
  - PyPDF2 (PDF processing)
  - python-docx (DOCX processing)
  - ChromaDB (vector database for RAG)
  - Custom RAG implementation

## Setup and Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Environment Setup
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/intelligent-financial-insights.git
   cd intelligent-financial-insights
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys:
   ```
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
   FINNHUB_API_KEY=your_finnhub_key
   ANTHROPIC_API_KEY=your_anthropic_key
   ```

### Running the Application
Launch the application using Streamlit:
```
streamlit run apptest.py
```

The application will be available at http://localhost:8501 by default.

## Usage Guide

### Getting Started
1. **Select a Stock**: Choose from favorites or search for a new one
2. **Choose Time Period**: Select the time frame for analysis (1 week to 1 year)
3. **Review Analysis**: Examine technical indicators, patterns, and recommendations
4. **Ask Questions**: Use the Investment Assistant to get AI-powered insights

### Advanced Features
1. **Document Upload**:
   - Click on Document Upload in the sidebar
   - Upload financial documents (PDF/DOCX)
   - Process the documents
   - Switch to Investment Assistant tab to ask document-aware questions

2. **Managing Favorites**:
   - Add stocks to favorites for quick access
   - Remove stocks from favorites when no longer needed

3. **Learning Resources**:
   - Access Stock Market Basics for educational content
   - Learn about different investment strategies and concepts

### Best Practices for Document Analysis
- Upload recent financial reports for the most relevant insights
- Include company annual reports, quarterly earnings, and analyst research
- Ask specific questions that reference both the technical data and document content
- Compare financial metrics with technical indicators for comprehensive analysis

## Architecture and Data Flow

The platform is built around a modular architecture:

1. **Data Collection Layer**: Interfaces with financial APIs to retrieve market data
2. **Processing Layer**: Computes technical indicators and performance metrics
3. **Storage Layer**: Manages favorites, query history, and stock data in SQLite
4. **RAG Processing Layer**: Handles document ingestion, processing, and retrieval
5. **AI Integration Layer**: Connects with Claude API for intelligent insights
6. **Presentation Layer**: Streamlit-based UI for interactive visualization and interaction

## Contributing

Contributions to improve the platform are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Licensing

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Alpha Vantage](https://www.alphavantage.co/) for financial market data
- [Finnhub](https://finnhub.io/) for additional market information and news
- [Anthropic](https://www.anthropic.com/) for Claude AI capabilities
- [Streamlit](https://streamlit.io/) for the interactive web framework
