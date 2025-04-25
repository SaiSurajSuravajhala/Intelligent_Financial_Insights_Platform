"""
Web scraping module for financial news and sentiment data.
"""
import logging
import time
import random
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from textblob import TextBlob

from config.settings import USER_AGENT, REQUEST_TIMEOUT

# Set up logging
logger = logging.getLogger(__name__)

class WebScraper:
    """Base web scraper class."""
    
    def __init__(self):
        """Initialize with headers."""
        self.headers = {
            'User-Agent': USER_AGENT,
            'Accept-Language': 'en-US,en;q=0.9',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_page(self, url: str) -> Optional[str]:
        """
        Get HTML content from a URL.
        
        Args:
            url: URL to get
            
        Returns:
            HTML content as string or None if error
        """
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching page {url}: {e}")
            return None
    
    def parse_html(self, html: str) -> BeautifulSoup:
        """
        Parse HTML content.
        
        Args:
            html: HTML content as string
            
        Returns:
            BeautifulSoup object
        """
        return BeautifulSoup(html, 'html.parser')
    
    def calculate_sentiment(self, text: str) -> float:
        """
        Calculate sentiment score for text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score (-1 to 1)
        """
        blob = TextBlob(text)
        return blob.sentiment.polarity


class SeleniumScraper(WebScraper):
    """Selenium-based scraper for JavaScript-heavy sites."""
    
    def __init__(self):
        """Initialize with Selenium driver."""
        super().__init__()
        self.driver = None
    
    def initialize_driver(self):
        """Initialize Selenium WebDriver."""
        if self.driver is None:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument(f"user-agent={USER_AGENT}")
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
    
    def close_driver(self):
        """Close Selenium WebDriver."""
        if self.driver is not None:
            self.driver.quit()
            self.driver = None
    
    def get_page_selenium(self, url: str, wait_time: int = 5) -> Optional[str]:
        """
        Get HTML content from a URL using Selenium.
        
        Args:
            url: URL to get
            wait_time: Time to wait for page to load (seconds)
            
        Returns:
            HTML content as string or None if error
        """
        try:
            self.initialize_driver()
            self.driver.get(url)
            # Wait for the page to load
            time.sleep(wait_time)
            return self.driver.page_source
        except Exception as e:
            logger.error(f"Error fetching page with Selenium {url}: {e}")
            return None
    
    def scroll_to_bottom(self, scroll_pause_time: float = 1.0):
        """
        Scroll to the bottom of the page to load lazy-loaded content.
        
        Args:
            scroll_pause_time: Time to pause between scrolls (seconds)
        """
        if self.driver is None:
            return
        
        # Get scroll height
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        
        while True:
            # Scroll down to bottom
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            # Wait to load page
            time.sleep(scroll_pause_time)
            
            # Calculate new scroll height and compare with last scroll height
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height


class FinancialNewsScraper(WebScraper):
    """Scraper for financial news websites."""
    
    def __init__(self):
        """Initialize with parent class."""
        super().__init__()
        self.selenium_scraper = SeleniumScraper()
    
    def get_news_from_cnbc(self, symbol: str, max_articles: int = 5) -> List[Dict[str, Any]]:
        """
        Get news from CNBC for a symbol.
        
        Args:
            symbol: Stock symbol
            max_articles: Maximum number of articles to get
            
        Returns:
            List of news items with title, url, summary, and sentiment
        """
        url = f"https://www.cnbc.com/quotes/{symbol.lower()}?tab=news"
        html = self.selenium_scraper.get_page_selenium(url)
        
        if html is None:
            return []
        
        soup = self.parse_html(html)
        news_items = []
        
        try:
            # Find news article elements
            articles = soup.select('.Card-standardBreakerCard')[:max_articles]
            
            for article in articles:
                headline_elem = article.select_one('.Card-title')
                summary_elem = article.select_one('.Card-description')
                link_elem = article.select_one('a.Card-title')
                time_elem = article.select_one('.Card-time')
                
                if headline_elem and link_elem:
                    title = headline_elem.text.strip()
                    url = link_elem['href']
                    summary = summary_elem.text.strip() if summary_elem else ""
                    published_str = time_elem.text.strip() if time_elem else ""
                    
                    # Parse published date
                    try:
                        if "hours ago" in published_str.lower():
                            hours = int(published_str.split()[0])
                            published_at = datetime.now() - timedelta(hours=hours)
                        elif "mins ago" in published_str.lower():
                            minutes = int(published_str.split()[0])
                            published_at = datetime.now() - timedelta(minutes=minutes)
                        else:
                            published_at = datetime.now()  # Default to now if can't parse
                    except:
                        published_at = datetime.now()
                    
                    # Calculate sentiment
                    text = f"{title} {summary}"
                    sentiment_score = self.calculate_sentiment(text)
                    
                    news_items.append({
                        'title': title,
                        'summary': summary,
                        'url': url,
                        'source': 'CNBC',
                        'published_at': published_at,
                        'sentiment_score': sentiment_score
                    })
        
        except Exception as e:
            logger.error(f"Error parsing CNBC news for {symbol}: {e}")
        
        return news_items
    
    def get_news_from_yahoo_finance(self, symbol: str, max_articles: int = 5) -> List[Dict[str, Any]]:
        """
        Get news from Yahoo Finance for a symbol.
        
        Args:
            symbol: Stock symbol
            max_articles: Maximum number of articles to get
            
        Returns:
            List of news items with title, url, summary, and sentiment
        """
        url = f"https://finance.yahoo.com/quote/{symbol}/news"
        html = self.selenium_scraper.get_page_selenium(url)
        
        if html is None:
            return []
        
        soup = self.parse_html(html)
        news_items = []
        
        try:
            # Find news article elements
            articles = soup.select('li.js-stream-content')[:max_articles]
            
            for article in articles:
                headline_elem = article.select_one('h3')
                summary_elem = article.select_one('p')
                link_elem = article.select_one('a')
                source_elem = article.select_one('.C(#959595)')
                
                if headline_elem and link_elem:
                    title = headline_elem.text.strip()
                    url = link_elem['href']
                    if not url.startswith('http'):
                        # Make relative URL absolute
                        url = f"https://finance.yahoo.com{url}"
                    
                    summary = summary_elem.text.strip() if summary_elem else ""
                    source_text = source_elem.text.strip() if source_elem else ""
                    
                    # Parse source and published date
                    try:
                        source_parts = source_text.split('·')
                        source = source_parts[0].strip()
                        published_at = datetime.now()  # Default
                        
                        if len(source_parts) > 1:
                            time_str = source_parts[1].strip()
                            if "hours ago" in time_str:
                                hours = int(time_str.split()[0])
                                published_at = datetime.now() - timedelta(hours=hours)
                            elif "minutes ago" in time_str:
                                minutes = int(time_str.split()[0])
                                published_at = datetime.now() - timedelta(minutes=minutes)
                            elif "days ago" in time_str:
                                days = int(time_str.split()[0])
                                published_at = datetime.now() - timedelta(days=days)
                    except:
                        source = "Yahoo Finance"
                        published_at = datetime.now()
                    
                    # Calculate sentiment
                    text = f"{title} {summary}"
                    sentiment_score = self.calculate_sentiment(text)
                    
                    news_items.append({
                        'title': title,
                        'summary': summary,
                        'url': url,
                        'source': source,
                        'published_at': published_at,
                        'sentiment_score': sentiment_score
                    })
        
        except Exception as e:
            logger.error(f"Error parsing Yahoo Finance news for {symbol}: {e}")
        
        return news_items
    
    def get_news_from_seeking_alpha(self, symbol: str, max_articles: int = 5) -> List[Dict[str, Any]]:
        """
        Get news from Seeking Alpha for a symbol.
        
        Args:
            symbol: Stock symbol
            max_articles: Maximum number of articles to get
            
        Returns:
            List of news items with title, url, summary, and sentiment
        """
        url = f"https://seekingalpha.com/symbol/{symbol}/news"
        html = self.selenium_scraper.get_page_selenium(url)
        
        if html is None:
            return []
        
        soup = self.parse_html(html)
        news_items = []
        
        try:
            # Find news article elements
            articles = soup.select('.media-heading')[:max_articles]
            
            for article in articles:
                headline_elem = article.select_one('a')
                
                if headline_elem:
                    title = headline_elem.text.strip()
                    relative_url = headline_elem['href']
                    url = f"https://seekingalpha.com{relative_url}"
                    
                    # Get article page to extract summary
                    article_html = self.get_page(url)
                    if article_html:
                        article_soup = self.parse_html(article_html)
                        summary_elem = article_soup.select_one('.bullets_li')
                        summary = summary_elem.text.strip() if summary_elem else ""
                    else:
                        summary = ""
                    
                    # Get published date (approximate)
                    published_at = datetime.now() - timedelta(days=random.randint(0, 3))
                    
                    # Calculate sentiment
                    text = f"{title} {summary}"
                    sentiment_score = self.calculate_sentiment(text)
                    
                    news_items.append({
                        'title': title,
                        'summary': summary,
                        'url': url,
                        'source': 'Seeking Alpha',
                        'published_at': published_at,
                        'sentiment_score': sentiment_score
                    })
        
        except Exception as e:
            logger.error(f"Error parsing Seeking Alpha news for {symbol}: {e}")
        
        return news_items
    
    def get_all_news(self, symbol: str, max_per_source: int = 3) -> List[Dict[str, Any]]:
        """
        Get news from all sources for a symbol.
        
        Args:
            symbol: Stock symbol
            max_per_source: Maximum number of articles per source
            
        Returns:
            List of news items with title, url, summary, and sentiment
        """
        all_news = []
        
        # Get news from each source
        try:
            yahoo_news = self.get_news_from_yahoo_finance(symbol, max_per_source)
            all_news.extend(yahoo_news)
            
            # Add a small delay between requests
            time.sleep(random.uniform(1, 3))
            
            cnbc_news = self.get_news_from_cnbc(symbol, max_per_source)
            all_news.extend(cnbc_news)
            
            # Add a small delay between requests
            time.sleep(random.uniform(1, 3))
            
            seeking_alpha_news = self.get_news_from_seeking_alpha(symbol, max_per_source)
            all_news.extend(seeking_alpha_news)
        except Exception as e:
            logger.error(f"Error getting all news for {symbol}: {e}")
        
        # Clean up Selenium driver
        self.selenium_scraper.close_driver()
        
        # Sort by published date (newest first)
        all_news.sort(key=lambda x: x['published_at'], reverse=True)
        
        return all_news


class SocialMediaScraper(WebScraper):
    """Scraper for social media sentiment."""
    
    def __init__(self):
        """Initialize with parent class."""
        super().__init__()
        self.selenium_scraper = SeleniumScraper()
    
    def get_twitter_sentiment(self, symbol: str, max_tweets: int = 20) -> Dict[str, Any]:
        """
        Get Twitter sentiment for a symbol.
        
        Args:
            symbol: Stock symbol
            max_tweets: Maximum number of tweets to analyze
            
        Returns:
            Dictionary with sentiment stats
        """
        # This is a simplified version as direct Twitter scraping is challenging
        # In a real implementation, you would use Twitter API or a specialized service
        
        # Using StockTwits as an alternative (more accessible for stock discussions)
        url = f"https://stocktwits.com/symbol/{symbol}"
        html = self.selenium_scraper.get_page_selenium(url)
        
        if html is None:
            return {
                'average_sentiment': 0,
                'sentiment_count': {'positive': 0, 'negative': 0, 'neutral': 0},
                'total_count': 0
            }
        
        soup = self.parse_html(html)
        sentiments = []
        
        try:
            # Find tweet/message elements
            messages = soup.select('.MessageStreamView__message')[:max_tweets]
            
            for message in messages:
                content_elem = message.select_one('.MessageStreamView__content')
                
                if content_elem:
                    text = content_elem.text.strip()
                    sentiment_score = self.calculate_sentiment(text)
                    sentiments.append(sentiment_score)
            
            # Calculate sentiment statistics
            sentiment_count = {
                'positive': sum(1 for s in sentiments if s > 0.2),
                'negative': sum(1 for s in sentiments if s < -0.2),
                'neutral': sum(1 for s in sentiments if -0.2 <= s <= 0.2)
            }
            
            average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            
            return {
                'average_sentiment': average_sentiment,
                'sentiment_count': sentiment_count,
                'total_count': len(sentiments)
            }
        
        except Exception as e:
            logger.error(f"Error getting social media sentiment for {symbol}: {e}")
            return {
                'average_sentiment': 0,
                'sentiment_count': {'positive': 0, 'negative': 0, 'neutral': 0},
                'total_count': 0
            }
        finally:
            # Clean up Selenium driver
            self.selenium_scraper.close_driver()


class AlternativeDataCollector:
    """Unified alternative data collector."""
    
    def __init__(self):
        """Initialize with scrapers."""
        self.news_scraper = FinancialNewsScraper()
        self.social_scraper = SocialMediaScraper()
    
    def get_all_alternative_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get all alternative data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with alternative data
        """
        data = {}
        
        # Get news
        data['news'] = self.news_scraper.get_all_news(symbol)
        
        # Get social media sentiment
        data['social_sentiment'] = self.social_scraper.get_twitter_sentiment(symbol)
        
        # Calculate overall sentiment
        news_sentiments = [item['sentiment_score'] for item in data['news']]
        social_sentiment = data['social_sentiment']['average_sentiment']
        
        if news_sentiments:
            news_avg_sentiment = sum(news_sentiments) / len(news_sentiments)
            # Weighted average (70% news, 30% social)
            overall_sentiment = 0.7 * news_avg_sentiment + 0.3 * social_sentiment
        else:
            overall_sentiment = social_sentiment
        
        data['overall_sentiment'] = overall_sentiment
        
        return data
