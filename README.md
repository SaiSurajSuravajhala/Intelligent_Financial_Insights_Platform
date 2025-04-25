# Intelligent Financial Insights Platform

A comprehensive AI-powered platform for financial market analysis and investment strategy.

## Overview

The Intelligent Financial Insights Platform is a sophisticated tool that combines traditional financial analysis with AI-powered insights to help users make informed investment decisions. The platform features:

- **Technical & Fundamental Analysis**: Comprehensive analysis of stock price patterns, indicators, company financials, and valuation metrics.
- **News & Sentiment Analysis**: Real-time analysis of news articles and social media sentiment related to stocks.
- **Investment Strategy Recommendations**: AI-generated investment strategies based on multiple data sources and analysis techniques.
- **Interactive Q&A Capability**: A natural language interface for asking investment-related questions.

## Project Architecture

The platform follows a modular hierarchical agent architecture:

### Data Collection Layer
- Web scraping module for news and social sentiment
- Financial API integration for market data and company fundamentals

### Data Processing & Storage Layer
- Data preprocessing module for cleaning and normalization
- Feature engineering for technical indicators
- SQLite database with Model Context Protocol

### Hierarchical Agent System
- Primary Agent Coordinator for orchestration
- Specialized modules for data analysis, insight generation, strategy recommendations, and user interaction

### User Interface Layer
- Streamlit-based dashboard with interactive visualizations
- Natural language query interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/financial-insights-platform.git
cd financial-insights-platform
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your API keys:
```
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
FINNHUB_API_KEY=your_finnhub_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Use the sidebar to select a stock symbol and time period for analysis.

3. Navigate through the tabs to explore different aspects of the analysis:
   - **Analysis Dashboard**: Technical and fundamental analysis
   - **News & Sentiment**: News articles and sentiment analysis
   - **Strategy**: Investment strategy recommendations
   - **Investment Assistant**: Ask questions about the stock or general investment topics

## Feature Highlights

### Technical Analysis
- Price charts with technical indicators
- Support and resistance level identification
- Pattern recognition
- Volume analysis
- Volatility assessment

### Fundamental Analysis
- Company financial metrics
- Valuation ratios
- Growth indicators
- Sector comparison

### News & Sentiment Analysis
- Latest news articles with sentiment scores
- Overall sentiment trends
- Market-moving news identification

### Investment Strategy
- Time-horizon based recommendations (short, medium, long-term)
- Entry and exit point suggestions
- Risk assessment and management
- Portfolio considerations

### Natural Language Interface
- Ask questions about stocks in plain English
- Get detailed answers based on comprehensive analysis
- Explore investment concepts and terminology

## Data Sources

The platform integrates data from multiple sources:

- **Alpha Vantage**: Historical stock prices, company fundamentals
- **Finnhub**: Real-time quotes, company news, financials
- **Web Scraping**: News articles, social media sentiment
- **Claude API**: Natural language processing and insight generation

## Customization

Users can customize the platform by:

- Uploading custom stock data (CSV format)
- Comparing multiple stocks
- Adjusting time periods for analysis
- Selecting specific technical indicators

## Dependencies

- Python 3.9+
- Streamlit
- Pandas & NumPy
- Plotly
- SQLModel & SQLAlchemy
- requests & BeautifulSoup
- Anthropic Claude API
- LangChain

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Financial data provided by Alpha Vantage and Finnhub
- Natural language capabilities powered by Anthropic's Claude
- Agent orchestration powered by LangChain
- Interactive visualizations built with Plotly
