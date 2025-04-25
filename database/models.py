"""
Database models using SQLModel for the Intelligent Financial Insights Platform.
These models define the structure of the database tables.
"""
from datetime import datetime
from typing import Optional, List
from sqlmodel import Field, SQLModel, Relationship


class StockData(SQLModel, table=True):
    """Model for storing historical stock data."""
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    date: datetime = Field(index=True)
    open: float
    high: float
    low: float
    close: float
    volume: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Composite index for faster lookups
    __table_args__ = (
        {"index": True, "name": "idx_symbol_date"},
    )


class CompanyInfo(SQLModel, table=True):
    """Model for storing company information."""
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True, unique=True)
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    news_items: List["NewsItem"] = Relationship(back_populates="company")


class NewsItem(SQLModel, table=True):
    """Model for storing news articles and sentiment."""
    id: Optional[int] = Field(default=None, primary_key=True)
    company_id: Optional[int] = Field(default=None, foreign_key="companyinfo.id")
    symbol: str = Field(index=True)
    title: str
    summary: str
    url: str
    source: str
    published_at: datetime
    sentiment_score: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationship
    company: Optional[CompanyInfo] = Relationship(back_populates="news_items")


class FinancialMetric(SQLModel, table=True):
    """Model for storing calculated financial metrics."""
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    date: datetime = Field(index=True)
    metric_name: str = Field(index=True)
    metric_value: float
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Composite index
    __table_args__ = (
        {"index": True, "name": "idx_symbol_date_metric"},
    )


class UserQuery(SQLModel, table=True):
    """Model for storing user queries and responses."""
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: Optional[str] = Field(default=None, index=True)
    query_text: str
    response_text: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    query_context: Optional[str] = None  # JSON string with context


class StrategyRecommendation(SQLModel, table=True):
    """Model for storing strategy recommendations."""
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    strategy_type: str  # e.g., "Short-term", "Long-term", "Value", "Growth"
    recommendation: str  # Buy, Sell, Hold
    confidence_score: float  # 0-1
    reasoning: str
    suggested_entry_price: Optional[float] = None
    suggested_exit_price: Optional[float] = None
    risk_assessment: str
    time_horizon: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class UserPortfolio(SQLModel, table=True):
    """Model for storing user portfolio information."""
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True)  # For future multi-user support
    symbol: str = Field(index=True)
    quantity: float
    purchase_price: float
    purchase_date: datetime
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
