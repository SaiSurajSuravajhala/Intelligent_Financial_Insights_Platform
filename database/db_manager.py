"""
Database management module for the Intelligent Financial Insights Platform.
Handles database connections, session management, and CRUD operations.
"""
import os
from datetime import datetime, timedelta
from typing import List, Optional, Type, TypeVar, Dict, Any, Generic

from sqlmodel import SQLModel, Session, create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from config.settings import DATABASE_URL
from database.models import (
    StockData, 
    CompanyInfo, 
    NewsItem, 
    FinancialMetric, 
    UserQuery, 
    StrategyRecommendation,
    UserPortfolio
)

# Create database directory if it doesn't exist
os.makedirs(os.path.dirname(DATABASE_URL.replace('sqlite:///', '')), exist_ok=True)

# Create engine
engine = create_engine(DATABASE_URL, echo=False)

# Create all tables
def create_db_and_tables():
    """Create database and tables."""
    SQLModel.metadata.create_all(engine)

# Generic type for SQLModel
T = TypeVar('T', bound=SQLModel)

class DatabaseManager(Generic[T]):
    """Generic database manager for CRUD operations."""
    
    def __init__(self, model_class: Type[T]):
        """Initialize with model class."""
        self.model_class = model_class
    
    def get_session(self) -> Session:
        """Create a new database session."""
        return Session(engine)
    
    def create(self, obj_data: Dict[str, Any]) -> T:
        """Create a new record."""
        with self.get_session() as session:
            db_obj = self.model_class(**obj_data)
            session.add(db_obj)
            session.commit()
            session.refresh(db_obj)
            return db_obj
    
    def get_by_id(self, id: int) -> Optional[T]:
        """Get record by ID."""
        with self.get_session() as session:
            statement = select(self.model_class).where(self.model_class.id == id)
            result = session.exec(statement).first()
            return result
    
    def get_all(self) -> List[T]:
        """Get all records."""
        with self.get_session() as session:
            statement = select(self.model_class)
            results = session.exec(statement).all()
            return results
    
    def update(self, id: int, obj_data: Dict[str, Any]) -> Optional[T]:
        """Update a record."""
        with self.get_session() as session:
            statement = select(self.model_class).where(self.model_class.id == id)
            db_obj = session.exec(statement).first()
            if db_obj:
                for key, value in obj_data.items():
                    setattr(db_obj, key, value)
                session.add(db_obj)
                session.commit()
                session.refresh(db_obj)
            return db_obj
    
    def delete(self, id: int) -> bool:
        """Delete a record."""
        with self.get_session() as session:
            statement = select(self.model_class).where(self.model_class.id == id)
            db_obj = session.exec(statement).first()
            if db_obj:
                session.delete(db_obj)
                session.commit()
                return True
            return False
    
    def bulk_create(self, items: List[Dict[str, Any]]) -> List[T]:
        """Create multiple records in bulk."""
        with self.get_session() as session:
            db_objects = [self.model_class(**item) for item in items]
            session.add_all(db_objects)
            session.commit()
            for obj in db_objects:
                session.refresh(obj)
            return db_objects


# Specialized database managers for each model
class StockDataManager(DatabaseManager[StockData]):
    """Manager for StockData model."""
    
    def __init__(self):
        """Initialize with StockData model."""
        super().__init__(StockData)
    
    def get_by_symbol(self, symbol: str) -> List[StockData]:
        """Get stock data by symbol."""
        with self.get_session() as session:
            statement = select(StockData).where(StockData.symbol == symbol)
            results = session.exec(statement).all()
            return results
    
    def get_by_symbol_and_date_range(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> List[StockData]:
        """Get stock data by symbol and date range."""
        with self.get_session() as session:
            statement = (
                select(StockData)
                .where(StockData.symbol == symbol)
                .where(StockData.date >= start_date)
                .where(StockData.date <= end_date)
                .order_by(StockData.date)
            )
            results = session.exec(statement).all()
            return results
    
    def get_latest_price(self, symbol: str) -> Optional[StockData]:
        """Get latest stock price."""
        with self.get_session() as session:
            statement = (
                select(StockData)
                .where(StockData.symbol == symbol)
                .order_by(StockData.date.desc())
            )
            result = session.exec(statement).first()
            return result


class CompanyInfoManager(DatabaseManager[CompanyInfo]):
    """Manager for CompanyInfo model."""
    
    def __init__(self):
        """Initialize with CompanyInfo model."""
        super().__init__(CompanyInfo)
    
    def get_by_symbol(self, symbol: str) -> Optional[CompanyInfo]:
        """Get company info by symbol."""
        with self.get_session() as session:
            statement = select(CompanyInfo).where(CompanyInfo.symbol == symbol)
            result = session.exec(statement).first()
            return result
    
    def get_by_sector(self, sector: str) -> List[CompanyInfo]:
        """Get companies by sector."""
        with self.get_session() as session:
            statement = select(CompanyInfo).where(CompanyInfo.sector == sector)
            results = session.exec(statement).all()
            return results


class NewsItemManager(DatabaseManager[NewsItem]):
    """Manager for NewsItem model."""
    
    def __init__(self):
        """Initialize with NewsItem model."""
        super().__init__(NewsItem)
    
    def get_by_symbol(self, symbol: str, limit: int = 10) -> List[NewsItem]:
        """Get news items by symbol."""
        with self.get_session() as session:
            statement = (
                select(NewsItem)
                .where(NewsItem.symbol == symbol)
                .order_by(NewsItem.published_at.desc())
                .limit(limit)
            )
            results = session.exec(statement).all()
            return results
    
    def get_recent_news(self, days: int = 7, limit: int = 20) -> List[NewsItem]:
        """Get recent news items."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        with self.get_session() as session:
            statement = (
                select(NewsItem)
                .where(NewsItem.published_at >= cutoff_date)
                .order_by(NewsItem.published_at.desc())
                .limit(limit)
            )
            results = session.exec(statement).all()
            return results


class FinancialMetricManager(DatabaseManager[FinancialMetric]):
    """Manager for FinancialMetric model."""
    
    def __init__(self):
        """Initialize with FinancialMetric model."""
        super().__init__(FinancialMetric)
    
    def get_by_symbol_and_metric(
        self, symbol: str, metric_name: str, limit: int = 30
    ) -> List[FinancialMetric]:
        """Get financial metrics by symbol and metric name."""
        with self.get_session() as session:
            statement = (
                select(FinancialMetric)
                .where(FinancialMetric.symbol == symbol)
                .where(FinancialMetric.metric_name == metric_name)
                .order_by(FinancialMetric.date.desc())
                .limit(limit)
            )
            results = session.exec(statement).all()
            return results
    
    def get_latest_metric(self, symbol: str, metric_name: str) -> Optional[FinancialMetric]:
        """Get latest financial metric by symbol and metric name."""
        with self.get_session() as session:
            statement = (
                select(FinancialMetric)
                .where(FinancialMetric.symbol == symbol)
                .where(FinancialMetric.metric_name == metric_name)
                .order_by(FinancialMetric.date.desc())
            )
            result = session.exec(statement).first()
            return result


class UserQueryManager(DatabaseManager[UserQuery]):
    """Manager for UserQuery model."""
    
    def __init__(self):
        """Initialize with UserQuery model."""
        super().__init__(UserQuery)
    
    def get_by_symbol(self, symbol: str, limit: int = 10) -> List[UserQuery]:
        """Get user queries by symbol."""
        with self.get_session() as session:
            statement = (
                select(UserQuery)
                .where(UserQuery.symbol == symbol)
                .order_by(UserQuery.timestamp.desc())
                .limit(limit)
            )
            results = session.exec(statement).all()
            return results


class StrategyRecommendationManager(DatabaseManager[StrategyRecommendation]):
    """Manager for StrategyRecommendation model."""
    
    def __init__(self):
        """Initialize with StrategyRecommendation model."""
        super().__init__(StrategyRecommendation)
    
    def get_by_symbol(self, symbol: str) -> List[StrategyRecommendation]:
        """Get strategy recommendations by symbol."""
        with self.get_session() as session:
            statement = (
                select(StrategyRecommendation)
                .where(StrategyRecommendation.symbol == symbol)
                .order_by(StrategyRecommendation.timestamp.desc())
            )
            results = session.exec(statement).all()
            return results
    
    def get_latest_recommendation(self, symbol: str) -> Optional[StrategyRecommendation]:
        """Get latest strategy recommendation by symbol."""
        with self.get_session() as session:
            statement = (
                select(StrategyRecommendation)
                .where(StrategyRecommendation.symbol == symbol)
                .order_by(StrategyRecommendation.timestamp.desc())
            )
            result = session.exec(statement).first()
            return result


class UserPortfolioManager(DatabaseManager[UserPortfolio]):
    """Manager for UserPortfolio model."""
    
    def __init__(self):
        """Initialize with UserPortfolio model."""
        super().__init__(UserPortfolio)
    
    def get_by_user_id(self, user_id: str) -> List[UserPortfolio]:
        """Get portfolio items by user ID."""
        with self.get_session() as session:
            statement = (
                select(UserPortfolio)
                .where(UserPortfolio.user_id == user_id)
                .order_by(UserPortfolio.symbol)
            )
            results = session.exec(statement).all()
            return results
    
    def get_by_user_and_symbol(self, user_id: str, symbol: str) -> Optional[UserPortfolio]:
        """Get portfolio item by user ID and symbol."""
        with self.get_session() as session:
            statement = (
                select(UserPortfolio)
                .where(UserPortfolio.user_id == user_id)
                .where(UserPortfolio.symbol == symbol)
            )
            result = session.exec(statement).first()
            return result
