# Intelligent Financial Insights Platform - Project Documentation

## Project Overview

The Intelligent Financial Insights Platform is a comprehensive web application built with Streamlit that provides advanced stock analysis, technical indicators, and AI-powered investment insights. The platform combines traditional technical analysis with modern AI capabilities and document processing to deliver a unique investment research experience.

## Project Goals

1. Create an intuitive financial analysis platform accessible to both novice and experienced investors
2. Implement comprehensive technical analysis with multiple indicators and pattern detection
3. Integrate AI capabilities to provide personalized investment insights and recommendations
4. Develop document-aware context through Retrieval Augmented Generation (RAG) implementation
5. Support multiple international stock markets with appropriate currency handling
6. Provide educational resources to help users understand investment concepts

## Technologies Used

### Core Framework and Libraries
- **Streamlit**: Web application framework for the user interface
- **Pandas & NumPy**: Data manipulation and numerical calculations
- **Plotly**: Interactive data visualization for charts and graphs
- **SQLModel**: ORM for database interactions

### APIs and External Services
- **Alpha Vantage API**: Market data retrieval service for stock price information
- **Finnhub API**: Financial news and company information
- **Anthropic Claude API**: AI-powered investment insights and analysis

### Database and Storage
- **SQLite**: Lightweight database for storing stock data, user queries, and favorites
- **Chroma DB**: Vector database for storing document embeddings (in RAG implementation)

### Document Processing
- **PyPDF2**: PDF document processing
- **python-docx**: Word document processing
- **RAG (Retrieval Augmented Generation)**: Framework for document-aware AI responses

## System Architecture

The application follows a modular architecture with clear separation of concerns:

### 1. Data Layer
- **Database Models**: SQLModel-based tables for stock data, user queries, and favorites
- **API Integration**: Alpha Vantage and Finnhub API clients for data retrieval
- **Caching**: Streamlit's caching mechanisms to optimize API usage

### 2. Business Logic Layer
- **Technical Indicators**: Calculation of RSI, MACD, Bollinger Bands, etc.
- **Pattern Detection**: Algorithms for identifying support/resistance levels and trends
- **Strategy Generation**: Investment recommendation engine based on technical analysis
- **Document Processing**: RAG implementation for document understanding

### 3. Presentation Layer
- **Dashboard**: Interactive charts and technical analysis visualization
- **Investment Assistant**: AI-powered investment advice interface
- **Document Upload**: Interface for uploading and processing financial documents

## Key Features and Implementation Details

### 1. Multi-Market Stock Analysis
The platform supports stocks from various international markets, with appropriate handling of exchange suffixes and currency formatting.

```python
# Exchange suffix guide by country
EXCHANGE_SUFFIXES = {
    "India": {".BSE": "Bombay Stock Exchange", ".NSE": "National Stock Exchange of India"},
    "UK": {".LON": "London Stock Exchange"},
    "Canada": {".TSX": "Toronto Stock Exchange", ".TSXV": "TSX Venture Exchange"},
    "Australia": {".AX": "Australian Securities Exchange"},
    "Germany": {".FRA": "Frankfurt Stock Exchange", ".XETRA": "Xetra"}
}

# Currency configuration by country
CURRENCY_CONFIG = {
    "US": {"symbol": "$", "code": "USD"},
    "India": {"symbol": "₹", "code": "INR"},
    "UK": {"symbol": "£", "code": "GBP"},
    "Canada": {"symbol": "C$", "code": "CAD"},
    "Australia": {"symbol": "A$", "code": "AUD"},
    "Germany": {"symbol": "€", "code": "EUR"}
}
```

### 2. Advanced Technical Analysis
The application calculates numerous technical indicators and detects patterns to provide comprehensive market insights:

- **Moving Averages (MA)**: 20-day, 50-day, and 200-day moving averages
- **RSI (Relative Strength Index)**: Momentum oscillator measuring speed and change of price movements
- **MACD (Moving Average Convergence Divergence)**: Trend-following momentum indicator
- **Bollinger Bands**: Volatility bands placed above and below a moving average
- **ADX (Average Directional Index)**: Measures trend strength
- **Support & Resistance Levels**: Automatically detected key price levels
- **Performance Metrics**: Period returns, volatility, drawdown calculations

### 3. AI-Powered Investment Assistant
The platform integrates with Anthropic's Claude API to provide intelligent investment insights:

- **Context-Aware Responses**: Claude receives current stock data and technical indicators
- **Document Integration**: Uploaded documents provide additional context for analysis
- **Natural Language Interface**: Users can ask questions in plain English
- **Personalized Recommendations**: Insights tailored to the specific stock and time period

### 4. Document Processing via RAG
The platform implements a Retrieval Augmented Generation (RAG) system for document-aware AI responses:

- **Document Upload**: Support for PDF and DOCX financial documents
- **Text Extraction**: Parsing and cleaning document content
- **Document Storage**: Efficient storage and retrieval of document content
- **Context Integration**: Combining document insights with technical analysis

### 5. User Experience Enhancements
The application includes several features to improve user experience:

- **Favorites Management**: Save and quickly access frequently analyzed stocks
- **Interactive Charts**: Dynamic visualizations with hover information
- **Educational Resources**: Built-in learning center for financial concepts
- **Responsive Layout**: Adaptive design for different screen sizes

## Data Flow

1. **User Interaction**:
   - User selects a stock or searches for a new one
   - User chooses a time period for analysis
   - User can upload financial documents for context

2. **Data Retrieval**:
   - Application fetches stock data from Alpha Vantage API
   - Application retrieves news from Finnhub API
   - Data is cached to minimize API calls

3. **Data Processing**:
   - Technical indicators are calculated from price data
   - Patterns are detected based on indicators
   - Documents are processed for contextual information

4. **Analysis Generation**:
   - Investment strategy recommendations are generated
   - Charts and visualizations are created
   - AI insights are prepared with available context

5. **User Interface Updates**:
   - Dashboard displays updated charts and analysis
   - Investment Assistant provides AI-powered responses
   - Document insights are integrated with technical analysis

## RAG Implementation Details

The platform implements a flexible RAG (Retrieval Augmented Generation) system with two alternative implementations:

### Implementation 1: Simple RAG
A straightforward implementation that stores document text in files:

1. **Document Processing**: Extract text from PDF/DOCX
2. **Text Storage**: Save text to files with metadata
3. **Retrieval**: Basic text lookup without embeddings
4. **Integration**: Include document text in AI prompts

### Implementation 2: ChromaDB-Based RAG
A more advanced implementation using ChromaDB for vector storage:

1. **Document Processing**: Extract and clean text from PDF/DOCX
2. **Chunking**: Split text into manageable chunks
3. **Embedding**: Generate vector embeddings for chunks
4. **Storage**: Store in ChromaDB with metadata
5. **Retrieval**: Semantic search based on query relevance
6. **Integration**: Include relevant document chunks in AI prompts

A custom adapter module (`document_processor.py`) allows the system to work with either implementation seamlessly.

## Database Schema

The application uses SQLite with SQLModel ORM, implementing these key tables:

1. **StockData**: Historical stock price and volume data
   - Fields: symbol, date, open, high, low, close, volume, created_at

2. **UserQuery**: Stored user queries and AI responses
   - Fields: symbol, query_text, response_text, timestamp

3. **Favorite**: User's favorite stocks
   - Fields: symbol, name, added_at

## Project Output

### Analysis Dashboard
- Interactive candlestick chart with technical indicators
- Current price metrics and percentage changes
- Investment recommendation with confidence level
- Technical indicator charts (RSI, MACD)
- Support and resistance levels
- Recent price data table
- News section with recent articles

### Investment Assistant
- Natural language query interface
- Document context selection
- Document insights extraction
- AI-powered responses integrating technical and fundamental analysis

### Learning Center
- Educational content on stock market basics
- Technical analysis explanations
- Fundamental analysis concepts
- Investment strategies overview
- Risk management principles

## Setup Instructions

### Environment Setup
1. Create a Python virtual environment
2. Install required packages via requirements.txt
3. Set up API keys in .env file:
   - ALPHA_VANTAGE_API_KEY
   - FINNHUB_API_KEY
   - ANTHROPIC_API_KEY

### Running the Application
```bash
# Navigate to project directory
cd intelligent-financial-insights

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run apptest.py
```

### Using the Application
1. Select or search for a stock symbol
2. Choose a time period for analysis
3. Review technical analysis in the dashboard
4. Upload relevant financial documents (optional)
5. Ask questions in the Investment Assistant tab

## Future Enhancements

1. **Portfolio Analysis**: Track and analyze multiple stocks as a portfolio
2. **Backtesting Module**: Test strategies against historical data
3. **Advanced Screener**: Filter stocks based on technical and fundamental criteria
4. **Real-Time Alerts**: Notifications for significant price movements or indicator signals
5. **Social Integration**: Share insights and recommendations
6. **Custom Indicators**: Allow users to create and save custom technical indicators
7. **Mobile Optimization**: Enhanced mobile responsiveness

## Conclusion

The Intelligent Financial Insights Platform represents a powerful combination of traditional technical analysis, modern AI capabilities, and document-aware context to provide comprehensive investment insights. By bridging quantitative market data with qualitative document analysis, the platform offers a unique approach to investment research that can assist both novice and experienced investors.