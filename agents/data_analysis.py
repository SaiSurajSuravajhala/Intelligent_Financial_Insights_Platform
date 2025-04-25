"""
Data Analysis Engine Agent for financial analysis.
"""
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from scipy import stats

# Set up logging
logger = logging.getLogger(__name__)

class DataAnalysisEngine:
    """
    Specialized Agent for financial data analysis.
    Performs calculations, identifies patterns, and detects anomalies.
    """
    
    def __init__(self):
        """Initialize data analysis engine."""
        pass
    
    def analyze_stock_data(
        self, data: pd.DataFrame, company_info: Dict[str, Any],
        financial_statements: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of stock data.
        
        Args:
            data: Processed stock data with features
            company_info: Company information
            financial_statements: Financial statements data
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Analyzing stock data")
        
        results = {}
        
        try:
            # Skip analysis if data is empty
            if data.empty:
                return {"error": "No data available for analysis"}
            
            # Basic statistics
            results["basic_stats"] = self.calculate_basic_statistics(data)
            
            # Performance analysis
            results["performance"] = self.analyze_performance(data)
            
            # Volatility analysis
            results["volatility"] = self.analyze_volatility(data)
            
            # Technical analysis
            results["technical"] = self.perform_technical_analysis(data)
            
            # Pattern detection
            results["patterns"] = self.detect_patterns(data)
            
            # Anomaly detection
            results["anomalies"] = self.detect_anomalies(data)
            
            # Fundamental analysis
            results["fundamental"] = self.analyze_fundamentals(company_info, financial_statements)
            
            return results
        
        except Exception as e:
            logger.error(f"Error in stock data analysis: {e}")
            return {"error": str(e)}
    
    def calculate_basic_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate basic statistics from stock data.
        
        Args:
            data: Stock price data
            
        Returns:
            Dictionary with basic statistics
        """
        stats_results = {}
        
        try:
            # Last 30, 90, and all days stats
            for period, days in [("30d", 30), ("90d", 90), ("all", None)]:
                period_data = data.iloc[-days:] if days else data
                
                if period_data.empty:
                    continue
                
                close_prices = period_data["close"]
                daily_returns = period_data["return"] if "return" in period_data.columns else period_data["close"].pct_change()
                
                stats_results[period] = {
                    "start_date": period_data.index[0].strftime("%Y-%m-%d"),
                    "end_date": period_data.index[-1].strftime("%Y-%m-%d"),
                    "trading_days": len(period_data),
                    "price_min": float(close_prices.min()),
                    "price_max": float(close_prices.max()),
                    "price_mean": float(close_prices.mean()),
                    "price_median": float(close_prices.median()),
                    "price_std": float(close_prices.std()),
                    "return_min": float(daily_returns.min()),
                    "return_max": float(daily_returns.max()),
                    "return_mean": float(daily_returns.mean()),
                    "return_median": float(daily_returns.median()),
                    "return_std": float(daily_returns.std()),
                    "positive_days": int((daily_returns > 0).sum()),
                    "negative_days": int((daily_returns < 0).sum()),
                    "current_price": float(close_prices.iloc[-1])
                }
            
            return stats_results
        
        except Exception as e:
            logger.error(f"Error calculating basic statistics: {e}")
            return {"error": str(e)}
    
    def analyze_performance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze performance metrics.
        
        Args:
            data: Stock price data
            
        Returns:
            Dictionary with performance metrics
        """
        performance_results = {}
        
        try:
            # Calculate returns for different time periods
            current_price = data["close"].iloc[-1]
            
            # Define periods in trading days (approximate)
            periods = {
                "1d": 1,
                "1w": 5,
                "1m": 21,
                "3m": 63,
                "6m": 126,
                "1y": 252,
                "3y": 756,
                "5y": 1260
            }
            
            for period_name, period_days in periods.items():
                if len(data) > period_days:
                    start_price = data["close"].iloc[-period_days-1]
                    period_return = (current_price - start_price) / start_price
                    performance_results[f"{period_name}_return"] = float(period_return)
                    performance_results[f"{period_name}_return_pct"] = float(period_return * 100)
            
            # Calculate annualized returns if we have enough data
            if len(data) >= 252:  # At least 1 year of data
                years = (data.index[-1] - data.index[0]).days / 365.25
                total_return = (data["close"].iloc[-1] / data["close"].iloc[0]) - 1
                annualized_return = (1 + total_return) ** (1 / years) - 1
                performance_results["annualized_return"] = float(annualized_return)
                performance_results["annualized_return_pct"] = float(annualized_return * 100)
            
            # Calculate risk-adjusted returns
            if "return" in data.columns and len(data) >= 252:
                daily_returns = data["return"].dropna()
                risk_free_rate = 0.03 / 252  # Approximate daily risk-free rate (3% annual)
                
                # Sharpe ratio
                excess_return = daily_returns - risk_free_rate
                sharpe_ratio = np.sqrt(252) * excess_return.mean() / excess_return.std()
                performance_results["sharpe_ratio"] = float(sharpe_ratio)
                
                # Sortino ratio (downside risk only)
                downside_returns = daily_returns[daily_returns < 0]
                sortino_ratio = np.sqrt(252) * excess_return.mean() / downside_returns.std() if len(downside_returns) > 0 else np.nan
                performance_results["sortino_ratio"] = float(sortino_ratio) if not np.isnan(sortino_ratio) else None
                
                # Maximum drawdown
                cumulative_returns = (1 + daily_returns).cumprod()
                running_max = cumulative_returns.cummax()
                drawdown = (cumulative_returns / running_max) - 1
                max_drawdown = drawdown.min()
                performance_results["max_drawdown"] = float(max_drawdown)
                performance_results["max_drawdown_pct"] = float(max_drawdown * 100)
            
            # Compare to market benchmark (if we had benchmark data)
            # This would normally compare to an index like S&P 500
            performance_results["benchmark_comparison"] = {
                "note": "Benchmark comparison requires market index data which is not available in this analysis"
            }
            
            return performance_results
        
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return {"error": str(e)}
    
    def analyze_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze volatility metrics.
        
        Args:
            data: Stock price data
            
        Returns:
            Dictionary with volatility metrics
        """
        volatility_results = {}
        
        try:
            # Extract pre-calculated volatility if available
            for period in [5, 10, 20, 30]:
                vol_col = f"volatility_{period}d"
                if vol_col in data.columns and not data[vol_col].isna().all():
                    current_vol = data[vol_col].iloc[-1]
                    volatility_results[vol_col] = float(current_vol)
                    volatility_results[f"{vol_col}_annualized_pct"] = float(current_vol * 100)
            
            # Calculate historical volatility for different periods if not already available
            if "volatility_30d" not in volatility_results and "return" in data.columns:
                returns = data["return"].dropna()
                
                for period in [30, 90, 180, 360]:
                    if len(returns) >= period:
                        period_returns = returns.iloc[-period:]
                        # Annualized volatility (std of returns * sqrt(trading days per year))
                        volatility = period_returns.std() * np.sqrt(252)
                        volatility_results[f"historical_volatility_{period}d"] = float(volatility)
                        volatility_results[f"historical_volatility_{period}d_pct"] = float(volatility * 100)
            
            # Calculate relative volatility compared to recent history
            if "volatility_30d" in data.columns and not data["volatility_30d"].isna().all():
                current_vol = data["volatility_30d"].iloc[-1]
                vol_series = data["volatility_30d"].dropna()
                
                if len(vol_series) > 60:  # Need sufficient history
                    historical_vol = vol_series.iloc[:-30].mean()  # Excluding most recent 30 days
                    vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
                    volatility_results["volatility_ratio_current_to_historical"] = float(vol_ratio)
            
            # Extreme moves analysis
            if "return" in data.columns:
                returns = data["return"].dropna()
                std_dev = returns.std()
                
                # Count days with moves greater than 1, 2, and 3 standard deviations
                volatility_results["extreme_moves"] = {
                    "moves_gt_1std": int((abs(returns) > std_dev).sum()),
                    "moves_gt_2std": int((abs(returns) > 2 * std_dev).sum()),
                    "moves_gt_3std": int((abs(returns) > 3 * std_dev).sum()),
                    "moves_gt_1std_pct": float((abs(returns) > std_dev).mean() * 100),
                    "moves_gt_2std_pct": float((abs(returns) > 2 * std_dev).mean() * 100),
                    "moves_gt_3std_pct": float((abs(returns) > 3 * std_dev).mean() * 100)
                }
            
            # Up/down day sequences
            if "return" in data.columns:
                returns = data["return"].dropna()
                up_days = returns > 0
                
                # Longest sequence of consecutive up/down days
                up_streak = self._longest_streak(up_days)
                down_streak = self._longest_streak(~up_days)
                
                volatility_results["streaks"] = {
                    "longest_up_streak": int(up_streak),
                    "longest_down_streak": int(down_streak)
                }
                
                # Current streak
                current_streak = 0
                for i in range(len(returns) - 1, -1, -1):
                    if (returns.iloc[i] > 0 and returns.iloc[i-1] > 0) or \
                       (returns.iloc[i] < 0 and returns.iloc[i-1] < 0):
                        current_streak += 1
                    else:
                        break
                
                streak_type = "up" if returns.iloc[-1] > 0 else "down"
                volatility_results["streaks"]["current_streak"] = int(current_streak)
                volatility_results["streaks"]["current_streak_type"] = streak_type
            
            return volatility_results
        
        except Exception as e:
            logger.error(f"Error analyzing volatility: {e}")
            return {"error": str(e)}
    
    def perform_technical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform technical analysis on stock data.
        
        Args:
            data: Stock price data with calculated indicators
            
        Returns:
            Dictionary with technical analysis results
        """
        technical_results = {}
        
        try:
            # Trend analysis
            technical_results["trend"] = self._analyze_trend(data)
            
            # Support and resistance levels
            technical_results["support_resistance"] = self._identify_support_resistance(data)
            
            # Moving average analysis
            technical_results["moving_averages"] = self._analyze_moving_averages(data)
            
            # RSI analysis
            if "rsi_14" in data.columns:
                technical_results["rsi"] = self._analyze_rsi(data)
            
            # MACD analysis
            if "macd" in data.columns and "macd_signal" in data.columns:
                technical_results["macd"] = self._analyze_macd(data)
            
            # Bollinger Bands analysis
            if "bb_upper" in data.columns and "bb_lower" in data.columns:
                technical_results["bollinger_bands"] = self._analyze_bollinger_bands(data)
            
            # Volume analysis
            technical_results["volume"] = self._analyze_volume(data)
            
            return technical_results
        
        except Exception as e:
            logger.error(f"Error performing technical analysis: {e}")
            return {"error": str(e)}
    
    def detect_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect chart patterns in stock data.
        
        Args:
            data: Stock price data
            
        Returns:
            Dictionary with detected patterns
        """
        pattern_results = {}
        
        try:
            # Check for candlestick patterns if they've been identified
            candlestick_patterns = ["doji", "hammer", "shooting_star", "bullish_engulfing", 
                                    "bearish_engulfing", "morning_star", "evening_star"]
            
            recent_patterns = {}
            for pattern in candlestick_patterns:
                if pattern in data.columns:
                    # Check for pattern in the last 5 days
                    recent_data = data.iloc[-5:]
                    pattern_days = recent_data[recent_data[pattern] == True]
                    
                    if not pattern_days.empty:
                        recent_patterns[pattern] = [date.strftime("%Y-%m-%d") for date in pattern_days.index]
            
            pattern_results["candlestick_patterns"] = recent_patterns
            
            # Check for technical chart patterns
            pattern_results["chart_patterns"] = self._identify_chart_patterns(data)
            
            # Check for trading ranges
            pattern_results["trading_range"] = self._identify_trading_range(data)
            
            # Check for gaps
            pattern_results["gaps"] = self._identify_price_gaps(data)
            
            return pattern_results
        
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return {"error": str(e)}
    
    def detect_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect anomalies in stock data.
        
        Args:
            data: Stock price data
            
        Returns:
            Dictionary with detected anomalies
        """
        anomaly_results = {}
        
        try:
            # Price anomalies (significant deviations from moving averages)
            anomaly_results["price_anomalies"] = self._detect_price_anomalies(data)
            
            # Volume anomalies
            anomaly_results["volume_anomalies"] = self._detect_volume_anomalies(data)
            
            # Volatility anomalies
            anomaly_results["volatility_anomalies"] = self._detect_volatility_anomalies(data)
            
            # Correlation anomalies (breakdown in normal price-volume relationship)
            anomaly_results["correlation_anomalies"] = self._detect_correlation_anomalies(data)
            
            return anomaly_results
        
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return {"error": str(e)}
    
    def analyze_fundamentals(
        self, company_info: Dict[str, Any], financial_statements: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Analyze fundamental data.
        
        Args:
            company_info: Company information
            financial_statements: Financial statements data
            
        Returns:
            Dictionary with fundamental analysis results
        """
        fundamental_results = {}
        
        try:
            # Basic company info
            fundamental_results["company"] = {
                "name": company_info.get("name", ""),
                "sector": company_info.get("sector", ""),
                "industry": company_info.get("industry", ""),
                "country": company_info.get("country", "")
            }
            
            # Key metrics
            fundamental_results["key_metrics"] = {
                "market_cap": company_info.get("marketCapitalization", 0),
                "pe_ratio": company_info.get("peRatio", 0),
                "price_to_book": company_info.get("priceToBookRatio", 0),
                "dividend_yield": company_info.get("dividendYield", 0),
                "beta": company_info.get("beta", 0),
                "52_week_high": company_info.get("52WeekHigh", 0),
                "52_week_low": company_info.get("52WeekLow", 0)
            }
            
            # Analysis of financial statements if available
            if financial_statements:
                fundamental_results["financial_analysis"] = self._analyze_financial_statements(financial_statements)
            
            return fundamental_results
        
        except Exception as e:
            logger.error(f"Error analyzing fundamentals: {e}")
            return {"error": str(e)}
    
    # Helper methods for technical analysis
    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price trend."""
        result = {}
        
        try:
            # Determine overall trend using moving averages
            if "ma_50" in data.columns and "ma_200" in data.columns:
                current_ma_50 = data["ma_50"].iloc[-1]
                current_ma_200 = data["ma_200"].iloc[-1]
                
                # Simple trend determination
                if current_ma_50 > current_ma_200:
                    result["overall_trend"] = "bullish"
                elif current_ma_50 < current_ma_200:
                    result["overall_trend"] = "bearish"
                else:
                    result["overall_trend"] = "neutral"
                
                # Check for golden/death cross
                if "golden_cross" in data.columns and "death_cross" in data.columns:
                    last_30_days = data.iloc[-30:]
                    golden_cross = last_30_days["golden_cross"].any()
                    death_cross = last_30_days["death_cross"].any()
                    
                    if golden_cross:
                        result["recent_signal"] = "golden_cross"
                    elif death_cross:
                        result["recent_signal"] = "death_cross"
            
            # Short-term trend using short moving averages
            if "ma_20" in data.columns:
                current_price = data["close"].iloc[-1]
                current_ma_20 = data["ma_20"].iloc[-1]
                
                if current_price > current_ma_20:
                    result["short_term_trend"] = "bullish"
                elif current_price < current_ma_20:
                    result["short_term_trend"] = "bearish"
                else:
                    result["short_term_trend"] = "neutral"
            
            # Trend strength using ADX if available
            if "adx" in data.columns:
                current_adx = data["adx"].iloc[-1]
                
                if current_adx > 25:
                    result["trend_strength"] = "strong"
                elif 20 <= current_adx <= 25:
                    result["trend_strength"] = "moderate"
                else:
                    result["trend_strength"] = "weak"
            
            return result
        
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return {"error": str(e)}
    
    def _identify_support_resistance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify support and resistance levels."""
        result = {}
        
        try:
            # Simple method to identify potential support/resistance levels
            # Looking at recent price extremes
            close_prices = data["close"].iloc[-90:]  # Last 90 days
            
            # Find local minima and maxima
            window_size = 5
            minima = []
            maxima = []
            
            for i in range(window_size, len(close_prices) - window_size):
                window = close_prices.iloc[i - window_size:i + window_size + 1]
                
                if close_prices.iloc[i] == window.min():
                    minima.append((close_prices.index[i], close_prices.iloc[i]))
                
                if close_prices.iloc[i] == window.max():
                    maxima.append((close_prices.index[i], close_prices.iloc[i]))
            
            # Group nearby levels (within 2% of each other)
            support_levels = self._group_price_levels([price for _, price in minima])
            resistance_levels = self._group_price_levels([price for _, price in maxima])
            
            # Convert to simple list of prices
            result["support_levels"] = [float(level) for level in support_levels]
            result["resistance_levels"] = [float(level) for level in resistance_levels]
            
            # Identify key levels (most recent or closest to current price)
            current_price = data["close"].iloc[-1]
            
            if support_levels:
                # Find closest support below current price
                supports_below = [s for s in support_levels if s < current_price]
                if supports_below:
                    result["key_support"] = float(max(supports_below))
                else:
                    result["key_support"] = float(min(support_levels))
            
            if resistance_levels:
                # Find closest resistance above current price
                resistances_above = [r for r in resistance_levels if r > current_price]
                if resistances_above:
                    result["key_resistance"] = float(min(resistances_above))
                else:
                    result["key_resistance"] = float(max(resistance_levels))
            
            return result
        
        except Exception as e:
            logger.error(f"Error identifying support/resistance: {e}")
            return {"error": str(e)}
    
    def _group_price_levels(self, prices: List[float], threshold_pct: float = 0.02) -> List[float]:
        """Group nearby price levels."""
        if not prices:
            return []
        
        # Sort prices
        sorted_prices = sorted(prices)
        
        # Group nearby levels
        groups = []
        current_group = [sorted_prices[0]]
        
        for price in sorted_prices[1:]:
            # Check if price is within threshold of the group's average
            group_avg = sum(current_group) / len(current_group)
            if abs(price - group_avg) / group_avg <= threshold_pct:
                current_group.append(price)
            else:
                # Start a new group
                groups.append(sum(current_group) / len(current_group))
                current_group = [price]
        
        # Add the last group
        if current_group:
            groups.append(sum(current_group) / len(current_group))
        
        return groups
    
    def _analyze_moving_averages(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze moving averages."""
        result = {}
        
        try:
            current_price = data["close"].iloc[-1]
            
            # Check relationship with key moving averages
            for ma_period in [20, 50, 200]:
                ma_col = f"ma_{ma_period}"
                if ma_col in data.columns:
                    current_ma = data[ma_col].iloc[-1]
                    
                    if not np.isnan(current_ma):
                        # Price relation to MA
                        relation = "above" if current_price > current_ma else "below"
                        result[f"price_vs_{ma_col}"] = relation
                        
                        # Distance from MA (percentage)
                        distance = (current_price - current_ma) / current_ma * 100
                        result[f"distance_from_{ma_col}_pct"] = float(distance)
                        
                        # MA slope (trend direction)
                        ma_slope = data[ma_col].diff(5).iloc[-1]  # 5-day change
                        slope_direction = "rising" if ma_slope > 0 else "falling"
                        result[f"{ma_col}_slope"] = slope_direction
            
            # Check for crossovers
            if "ma_50" in data.columns and "ma_200" in data.columns:
                # Golden cross / death cross status
                ma_50 = data["ma_50"].iloc[-1]
                ma_200 = data["ma_200"].iloc[-1]
                
                if ma_50 > ma_200:
                    result["ma_50_200_status"] = "bullish"
                else:
                    result["ma_50_200_status"] = "bearish"
                
                # Check for recent crosses (last 30 days)
                last_30 = data.iloc[-30:]
                
                golden_cross_days = last_30[last_30["ma_50"] > last_30["ma_200"]].index
                death_cross_days = last_30[last_30["ma_50"] < last_30["ma_200"]].index
                
                if len(golden_cross_days) > 0 and len(death_cross_days) > 0:
                    # Both happened - which was more recent?
                    if golden_cross_days[-1] > death_cross_days[-1]:
                        result["recent_cross"] = {
                            "type": "golden_cross",
                            "date": golden_cross_days[-1].strftime("%Y-%m-%d")
                        }
                    else:
                        result["recent_cross"] = {
                            "type": "death_cross",
                            "date": death_cross_days[-1].strftime("%Y-%m-%d")
                        }
                elif len(golden_cross_days) > 0:
                    result["recent_cross"] = {
                        "type": "golden_cross",
                        "date": golden_cross_days[-1].strftime("%Y-%m-%d")
                    }
                elif len(death_cross_days) > 0:
                    result["recent_cross"] = {
                        "type": "death_cross",
                        "date": death_cross_days[-1].strftime("%Y-%m-%d")
                    }
            
            return result
        
        except Exception as e:
            logger.error(f"Error analyzing moving averages: {e}")
            return {"error": str(e)}
    
    def _analyze_rsi(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze RSI indicator."""
        result = {}
        
        try:
            current_rsi = data["rsi_14"].iloc[-1]
            result["current_value"] = float(current_rsi)
            
            # Interpret RSI value
            if current_rsi < 30:
                result["condition"] = "oversold"
            elif current_rsi > 70:
                result["condition"] = "overbought"
            else:
                result["condition"] = "neutral"
            
            # Check for divergence
            price_trend = data["close"].iloc[-5:].is_monotonic_increasing
            rsi_trend = data["rsi_14"].iloc[-5:].is_monotonic_increasing
            
            if price_trend and not rsi_trend:
                result["divergence"] = "bearish"
            elif not price_trend and rsi_trend:
                result["divergence"] = "bullish"
            else:
                result["divergence"] = "none"
            
            # Check RSI trend
            rsi_values = data["rsi_14"].iloc[-14:]
            result["trend"] = "rising" if rsi_values.iloc[-1] > rsi_values.iloc[0] else "falling"
            
            return result
        
        except Exception as e:
            logger.error(f"Error analyzing RSI: {e}")
            return {"error": str(e)}
    
    def _analyze_macd(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze MACD indicator."""
        result = {}
        
        try:
            current_macd = data["macd"].iloc[-1]
            current_signal = data["macd_signal"].iloc[-1]
            current_histogram = data["macd_histogram"].iloc[-1]
            
            result["current_values"] = {
                "macd": float(current_macd),
                "signal": float(current_signal),
                "histogram": float(current_histogram)
            }
            
            # MACD position
            if current_macd > current_signal:
                result["position"] = "bullish"
            else:
                result["position"] = "bearish"
            
            # Check for signal line crossover
            macd_values = data["macd"].iloc[-10:]
            signal_values = data["macd_signal"].iloc[-10:]
            
            crossover_days = []
            for i in range(1, len(macd_values)):
                prev_diff = macd_values.iloc[i-1] - signal_values.iloc[i-1]
                curr_diff = macd_values.iloc[i] - signal_values.iloc[i]
                
                if prev_diff < 0 and curr_diff > 0:
                    # Bullish crossover
                    crossover_days.append({
                        "date": macd_values.index[i].strftime("%Y-%m-%d"),
                        "type": "bullish"
                    })
                elif prev_diff > 0 and curr_diff < 0:
                    # Bearish crossover
                    crossover_days.append({
                        "date": macd_values.index[i].strftime("%Y-%m-%d"),
                        "type": "bearish"
                    })
            
            result["recent_crossovers"] = crossover_days
            
            # MACD trend
            result["trend"] = "rising" if current_macd > data["macd"].iloc[-2] else "falling"
            
            # Check for zero line crossover
            if data["macd"].iloc[-2] < 0 and current_macd > 0:
                result["zero_line_crossover"] = "bullish"
            elif data["macd"].iloc[-2] > 0 and current_macd < 0:
                result["zero_line_crossover"] = "bearish"
            else:
                result["zero_line_crossover"] = "none"
            
            return result
        
        except Exception as e:
            logger.error(f"Error analyzing MACD: {e}")
            return {"error": str(e)}
    
    def _analyze_bollinger_bands(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Bollinger Bands."""
        result = {}
        
        try:
            current_price = data["close"].iloc[-1]
            current_upper = data["bb_upper"].iloc[-1]
            current_lower = data["bb_lower"].iloc[-1]
            current_middle = data["bb_middle"].iloc[-1]
            
            result["current_values"] = {
                "upper": float(current_upper),
                "middle": float(current_middle),
                "lower": float(current_lower)
            }
            
            # Price position relative to bands
            if current_price > current_upper:
                result["price_position"] = "above_upper"
            elif current_price < current_lower:
                result["price_position"] = "below_lower"
            else:
                result["price_position"] = "inside_bands"
            
            # Calculate bandwidth
            if "bb_width" in data.columns:
                current_width = data["bb_width"].iloc[-1]
                avg_width = data["bb_width"].iloc[-20:].mean()
                
                result["bandwidth"] = {
                    "current": float(current_width),
                    "average_20d": float(avg_width),
                    "is_contracting": current_width < avg_width
                }
            
            # Check for recent band touches
            last_20_days = data.iloc[-20:]
            upper_touches = (last_20_days["high"] >= last_20_days["bb_upper"]).sum()
            lower_touches = (last_20_days["low"] <= last_20_days["bb_lower"]).sum()
            
            result["band_touches_last_20d"] = {
                "upper": int(upper_touches),
                "lower": int(lower_touches)
            }
            
            # Check for squeeze (narrow bands)
            if "bb_width" in data.columns:
                width_percentile = stats.percentileofscore(
                    data["bb_width"].iloc[-60:], current_width
                )
                
                result["squeeze_percentile"] = float(width_percentile)
                result["is_squeezed"] = width_percentile < 20  # Bottom 20% of width
            
            return result
        
        except Exception as e:
            logger.error(f"Error analyzing Bollinger Bands: {e}")
            return {"error": str(e)}
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns."""
        result = {}
        
        try:
            # Get recent volume data
            recent_volume = data["volume"].iloc[-30:]
            current_volume = data["volume"].iloc[-1]
            avg_volume_20d = data["volume"].iloc[-20:].mean()
            
            result["current_volume"] = int(current_volume)
            result["avg_volume_20d"] = float(avg_volume_20d)
            result["volume_ratio_to_avg"] = float(current_volume / avg_volume_20d)
            
            # Volume trend
            volume_5d = data["volume"].iloc[-5:].mean()
            volume_20d = data["volume"].iloc[-20:].mean()
            
            result["volume_trend"] = "increasing" if volume_5d > volume_20d else "decreasing"
            
            # Volume spikes
            volume_std = data["volume"].iloc[-20:].std()
            spikes = (data["volume"] > (avg_volume_20d + 2 * volume_std)).iloc[-5:].sum()
            
            result["recent_volume_spikes"] = int(spikes)
            
            # Volume on up vs down days
            up_days_volume = data.loc[data["return"] > 0, "volume"].iloc[-20:].mean()
            down_days_volume = data.loc[data["return"] < 0, "volume"].iloc[-20:].mean()
            
            if not np.isnan(up_days_volume) and not np.isnan(down_days_volume) and down_days_volume > 0:
                up_down_ratio = up_days_volume / down_days_volume
                result["up_down_volume_ratio"] = float(up_down_ratio)
                
                if up_down_ratio > 1.2:
                    result["volume_pattern"] = "stronger_on_up_days"
                elif up_down_ratio < 0.8:
                    result["volume_pattern"] = "stronger_on_down_days"
                else:
                    result["volume_pattern"] = "balanced"
            
            # On-Balance Volume trend
            if "obv" in data.columns:
                obv_trend = data["obv"].iloc[-5:].is_monotonic_increasing
                price_trend = data["close"].iloc[-5:].is_monotonic_increasing
                
                result["obv_trend"] = "rising" if obv_trend else "falling"
                
                # Check for divergence
                if obv_trend and not price_trend:
                    result["obv_divergence"] = "bullish"
                elif not obv_trend and price_trend:
                    result["obv_divergence"] = "bearish"
                else:
                    result["obv_divergence"] = "none"
            
            return result
        
        except Exception as e:
            logger.error(f"Error analyzing volume: {e}")
            return {"error": str(e)}
    
    def _identify_chart_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify common chart patterns."""
        result = {}
        
        try:
            # This is a simplified implementation
            # Real pattern detection would be more complex
            
            # Get recent price data
            close_prices = data["close"].iloc[-60:]
            high_prices = data["high"].iloc[-60:]
            low_prices = data["low"].iloc[-60:]
            
            # Head and shoulders detection (simplified)
            # Look for 3 peaks with middle peak higher
            pattern_found = False
            
            if len(close_prices) >= 30:
                # Use a rolling window to find peaks
                for i in range(5, len(close_prices) - 5):
                    left_window = close_prices.iloc[i-5:i]
                    right_window = close_prices.iloc[i:i+5]
                    
                    # Check if this point is a peak
                    if close_prices.iloc[i] == max(close_prices.iloc[i-5:i+6]):
                        # Found a potential peak
                        # Look for left and right shoulders
                        left_peak = None
                        right_peak = None
                        
                        # Look for left shoulder
                        for j in range(i - 10, i - 5):
                            if j >= 0 and close_prices.iloc[j] == max(close_prices.iloc[max(0, j-5):j+6]):
                                left_peak = j
                                break
                        
                        # Look for right shoulder
                        for j in range(i + 5, i + 10):
                            if j < len(close_prices) and close_prices.iloc[j] == max(close_prices.iloc[j-5:min(len(close_prices), j+6)]):
                                right_peak = j
                                break
                        
                        if left_peak is not None and right_peak is not None:
                            # Check if middle peak is higher
                            if close_prices.iloc[i] > close_prices.iloc[left_peak] and \
                               close_prices.iloc[i] > close_prices.iloc[right_peak] and \
                               abs(close_prices.iloc[left_peak] - close_prices.iloc[right_peak]) / close_prices.iloc[i] < 0.1:
                                pattern_found = True
                                result["head_and_shoulders"] = {
                                    "detected": True,
                                    "head_date": close_prices.index[i].strftime("%Y-%m-%d"),
                                    "left_shoulder": close_prices.index[left_peak].strftime("%Y-%m-%d"),
                                    "right_shoulder": close_prices.index[right_peak].strftime("%Y-%m-%d")
                                }
                                break
            
            if not pattern_found:
                result["head_and_shoulders"] = {"detected": False}
            
            # Double top / double bottom detection (simplified)
            double_top_found = False
            double_bottom_found = False
            
            if len(close_prices) >= 20:
                # Find local maxima and minima
                maxima = []
                minima = []
                window = 5
                
                for i in range(window, len(close_prices) - window):
                    if close_prices.iloc[i] == max(close_prices.iloc[i-window:i+window+1]):
                        maxima.append((close_prices.index[i], close_prices.iloc[i]))
                    
                    if close_prices.iloc[i] == min(close_prices.iloc[i-window:i+window+1]):
                        minima.append((close_prices.index[i], close_prices.iloc[i]))
                
                # Check for double top
                if len(maxima) >= 2:
                    # Check if the two highest peaks are within 5% of each other
                    maxima.sort(key=lambda x: x[1], reverse=True)
                    top1, top2 = maxima[0], maxima[1]
                    
                    if abs(top1[1] - top2[1]) / top1[1] < 0.05:
                        # Check if they're separated by at least 10 days
                        days_between = abs((top1[0] - top2[0]).days)
                        if days_between >= 10:
                            double_top_found = True
                            result["double_top"] = {
                                "detected": True,
                                "peak1_date": top1[0].strftime("%Y-%m-%d"),
                                "peak2_date": top2[0].strftime("%Y-%m-%d"),
                                "price_level": float(top1[1])
                            }
                
                # Check for double bottom
                if len(minima) >= 2:
                    # Check if the two lowest troughs are within 5% of each other
                    minima.sort(key=lambda x: x[1])
                    bottom1, bottom2 = minima[0], minima[1]
                    
                    if abs(bottom1[1] - bottom2[1]) / bottom1[1] < 0.05:
                        # Check if they're separated by at least 10 days
                        days_between = abs((bottom1[0] - bottom2[0]).days)
                        if days_between >= 10:
                            double_bottom_found = True
                            result["double_bottom"] = {
                                "detected": True,
                                "trough1_date": bottom1[0].strftime("%Y-%m-%d"),
                                "trough2_date": bottom2[0].strftime("%Y-%m-%d"),
                                "price_level": float(bottom1[1])
                            }
            
            if not double_top_found:
                result["double_top"] = {"detected": False}
            
            if not double_bottom_found:
                result["double_bottom"] = {"detected": False}
            
            # Detect flags/pennants
            # (Simplified - looking for a strong move followed by consolidation)
            flag_found = False
            pennant_found = False
            
            if len(close_prices) >= 20:
                # Check for a strong move (> 10% in 5 days)
                for i in range(5, len(close_prices) - 10):
                    change_pct = (close_prices.iloc[i] - close_prices.iloc[i-5]) / close_prices.iloc[i-5]
                    
                    if abs(change_pct) > 0.1:
                        # Strong move detected, now check for consolidation
                        consolidation = close_prices.iloc[i:i+10]
                        consolidation_range = (consolidation.max() - consolidation.min()) / consolidation.mean()
                        
                        if consolidation_range < 0.05:  # Tight consolidation
                            if change_pct > 0:  # Bullish flag
                                flag_found = True
                                result["flag"] = {
                                    "detected": True,
                                    "type": "bullish",
                                    "pole_start": close_prices.index[i-5].strftime("%Y-%m-%d"),
                                    "pole_end": close_prices.index[i].strftime("%Y-%m-%d"),
                                    "flag_end": close_prices.index[i+10].strftime("%Y-%m-%d")
                                }
                                break
                            else:  # Bearish flag
                                flag_found = True
                                result["flag"] = {
                                    "detected": True,
                                    "type": "bearish",
                                    "pole_start": close_prices.index[i-5].strftime("%Y-%m-%d"),
                                    "pole_end": close_prices.index[i].strftime("%Y-%m-%d"),
                                    "flag_end": close_prices.index[i+10].strftime("%Y-%m-%d")
                                }
                                break
            
            if not flag_found:
                result["flag"] = {"detected": False}
            
            if not pennant_found:
                result["pennant"] = {"detected": False}
            
            return result
        
        except Exception as e:
            logger.error(f"Error identifying chart patterns: {e}")
            return {"error": str(e)}
    
    def _identify_trading_range(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify if price is in a trading range."""
        result = {}
        
        try:
            # Use recent price data (last 30 days)
            close_prices = data["close"].iloc[-30:]
            
            # Calculate the range
            price_range = (close_prices.max() - close_prices.min()) / close_prices.mean()
            
            # Check if price is in a tight range
            if price_range < 0.05:  # Less than 5% range
                result["in_trading_range"] = True
                result["range_width_pct"] = float(price_range * 100)
                result["range_high"] = float(close_prices.max())
                result["range_low"] = float(close_prices.min())
                result["days_in_range"] = int(len(close_prices))
            else:
                result["in_trading_range"] = False
            
            return result
        
        except Exception as e:
            logger.error(f"Error identifying trading range: {e}")
            return {"error": str(e)}
    
    def _identify_price_gaps(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify significant price gaps."""
        result = {"recent_gaps": []}
        
        try:
            # Look for gaps in the last 30 days
            for i in range(1, min(30, len(data))):
                today = data.iloc[-i]
                yesterday = data.iloc[-(i+1)]
                
                # Check for gap up
                if today["low"] > yesterday["high"]:
                    gap_pct = (today["low"] - yesterday["high"]) / yesterday["high"] * 100
                    if gap_pct > 1.0:  # Gap greater than 1%
                        result["recent_gaps"].append({
                            "date": data.index[-i].strftime("%Y-%m-%d"),
                            "type": "gap_up",
                            "gap_pct": float(gap_pct),
                            "filled": bool(
                                data.iloc[-i:]["low"].min() <= yesterday["high"]
                            )
                        })
                
                # Check for gap down
                elif today["high"] < yesterday["low"]:
                    gap_pct = (yesterday["low"] - today["high"]) / yesterday["low"] * 100
                    if gap_pct > 1.0:  # Gap greater than 1%
                        result["recent_gaps"].append({
                            "date": data.index[-i].strftime("%Y-%m-%d"),
                            "type": "gap_down",
                            "gap_pct": float(gap_pct),
                            "filled": bool(
                                data.iloc[-i:]["high"].max() >= yesterday["low"]
                            )
                        })
            
            result["gap_count"] = len(result["recent_gaps"])
            
            return result
        
        except Exception as e:
            logger.error(f"Error identifying price gaps: {e}")
            return {"error": str(e)}
    
    def _detect_price_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect price anomalies."""
        result = {"anomalies": []}
        
        try:
            # Check for abnormal price movements
            close_prices = data["close"]
            returns = data["return"] if "return" in data.columns else close_prices.pct_change()
            
            # Calculate rolling mean and std of returns
            rolling_mean = returns.rolling(window=20).mean()
            rolling_std = returns.rolling(window=20).std()
            
            # Z-scores
            z_scores = (returns - rolling_mean) / rolling_std
            
            # Find days with extreme z-scores in the last 30 days
            for i in range(min(30, len(data))):
                day = data.index[-i-1]  # Skip the most recent day
                z_score = z_scores.loc[day]
                
                if abs(z_score) > 3:  # More than 3 standard deviations
                    result["anomalies"].append({
                        "date": day.strftime("%Y-%m-%d"),
                        "type": "price_movement",
                        "z_score": float(z_score),
                        "return_pct": float(returns.loc[day] * 100)
                    })
            
            # Check for abnormal relationship with moving averages
            if "ma_20" in data.columns:
                # Calculate distance from 20-day MA
                distance = (data["close"] - data["ma_20"]) / data["ma_20"]
                
                # Calculate rolling average and std of the distance
                rolling_dist_mean = distance.rolling(window=60).mean()
                rolling_dist_std = distance.rolling(window=60).std()
                
                # Z-scores
                dist_z_scores = (distance - rolling_dist_mean) / rolling_dist_std
                
                # Find days with extreme distance from MA
                for i in range(min(30, len(data))):
                    day = data.index[-i-1]  # Skip the most recent day
                    z_score = dist_z_scores.loc[day]
                    
                    if abs(z_score) > 3:  # More than 3 standard deviations
                        result["anomalies"].append({
                            "date": day.strftime("%Y-%m-%d"),
                            "type": "ma_deviation",
                            "z_score": float(z_score),
                            "deviation_pct": float(distance.loc[day] * 100)
                        })
            
            return result
        
        except Exception as e:
            logger.error(f"Error detecting price anomalies: {e}")
            return {"error": str(e)}
    
    def _detect_volume_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect volume anomalies."""
        result = {"anomalies": []}
        
        try:
            # Check for abnormal volume
            volume = data["volume"]
            
            # Calculate rolling mean and std of volume
            rolling_mean = volume.rolling(window=20).mean()
            rolling_std = volume.rolling(window=20).std()
            
            # Z-scores
            z_scores = (volume - rolling_mean) / rolling_std
            
            # Find days with extreme z-scores in the last 30 days
            for i in range(min(30, len(data))):
                day = data.index[-i-1]  # Skip the most recent day
                z_score = z_scores.loc[day]
                
                if z_score > 3:  # More than 3 standard deviations
                    result["anomalies"].append({
                        "date": day.strftime("%Y-%m-%d"),
                        "type": "high_volume",
                        "z_score": float(z_score),
                        "volume_ratio": float(volume.loc[day] / rolling_mean.loc[day])
                    })
                elif z_score < -2:  # Very low volume (2 std below mean)
                    result["anomalies"].append({
                        "date": day.strftime("%Y-%m-%d"),
                        "type": "low_volume",
                        "z_score": float(z_score),
                        "volume_ratio": float(volume.loc[day] / rolling_mean.loc[day])
                    })
            
            return result
        
        except Exception as e:
            logger.error(f"Error detecting volume anomalies: {e}")
            return {"error": str(e)}
    
    def _detect_volatility_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect volatility anomalies."""
        result = {"anomalies": []}
        
        try:
            # Check for volatility spikes or drops
            if "volatility_20d" in data.columns:
                volatility = data["volatility_20d"]
                
                # Calculate rolling mean and std of volatility
                rolling_mean = volatility.rolling(window=30).mean()
                rolling_std = volatility.rolling(window=30).std()
                
                # Z-scores
                z_scores = (volatility - rolling_mean) / rolling_std
                
                # Find days with extreme z-scores in the last 30 days
                for i in range(min(30, len(data))):
                    day = data.index[-i-1]  # Skip the most recent day
                    z_score = z_scores.loc[day]
                    
                    if abs(z_score) > 2.5:  # More than 2.5 standard deviations
                        result["anomalies"].append({
                            "date": day.strftime("%Y-%m-%d"),
                            "type": "volatility_spike" if z_score > 0 else "volatility_drop",
                            "z_score": float(z_score),
                            "volatility_ratio": float(volatility.loc[day] / rolling_mean.loc[day])
                        })
            
            return result
        
        except Exception as e:
            logger.error(f"Error detecting volatility anomalies: {e}")
            return {"error": str(e)}
    
    def _detect_correlation_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect correlation anomalies between price and volume."""
        result = {"anomalies": []}
        
        try:
            # Check for price-volume correlation anomalies
            if "return" in data.columns:
                # Calculate rolling correlation between returns and volume changes
                returns = data["return"]
                volume_changes = data["volume"].pct_change()
                
                # Calculate 20-day rolling correlation
                window = 20
                correlations = pd.Series(index=data.index[window:])
                
                for i in range(window, len(data)):
                    window_returns = returns.iloc[i-window:i]
                    window_volume = volume_changes.iloc[i-window:i]
                    correlation = window_returns.corr(window_volume)
                    correlations.iloc[i-window] = correlation
                
                # Calculate rolling mean and std of correlations
                rolling_mean = correlations.rolling(window=30).mean()
                rolling_std = correlations.rolling(window=30).std()
                
                # Z-scores
                z_scores = (correlations - rolling_mean) / rolling_std
                
                # Find days with extreme z-scores in the last 30 days
                for i in range(min(30, len(z_scores))):
                    if i >= len(z_scores) or (-i-1) >= len(z_scores):
                        continue
                    
                    day = z_scores.index[-i-1]  # Skip the most recent day
                    z_score = z_scores.loc[day]
                    
                    if abs(z_score) > 2.5:  # More than 2.5 standard deviations
                        result["anomalies"].append({
                            "date": day.strftime("%Y-%m-%d"),
                            "type": "correlation_anomaly",
                            "z_score": float(z_score),
                            "correlation": float(correlations.loc[day])
                        })
            
            return result
        
        except Exception as e:
            logger.error(f"Error detecting correlation anomalies: {e}")
            return {"error": str(e)}
    
    def _analyze_financial_statements(self, financial_statements: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze financial statements."""
        result = {}
        
        try:
            # Extract statements
            income_statement = financial_statements.get("income_statement", pd.DataFrame())
            balance_sheet = financial_statements.get("balance_sheet", pd.DataFrame())
            cash_flow = financial_statements.get("cash_flow", pd.DataFrame())
            
            if income_statement.empty or balance_sheet.empty:
                return {"error": "Insufficient financial statement data"}
            
            # Analyze income statement
            result["income_statement"] = self._analyze_income_statement(income_statement)
            
            # Analyze balance sheet
            result["balance_sheet"] = self._analyze_balance_sheet(balance_sheet)
            
            # Analyze cash flow
            if not cash_flow.empty:
                result["cash_flow"] = self._analyze_cash_flow(cash_flow)
            
            # Calculate key financial ratios
            result["key_ratios"] = self._calculate_financial_ratios(income_statement, balance_sheet, cash_flow)
            
            return result
        
        except Exception as e:
            logger.error(f"Error analyzing financial statements: {e}")
            return {"error": str(e)}
    
    def _analyze_income_statement(self, income_statement: pd.DataFrame) -> Dict[str, Any]:
        """Analyze income statement."""
        result = {}
        
        try:
            # Extract latest period data
            latest = income_statement.iloc[:, 0]
            prev_year = income_statement.iloc[:, 1] if income_statement.shape[1] > 1 else None
            
            # Extract key metrics
            metrics = {
                "revenue": latest.get("totalRevenue", 0),
                "gross_profit": latest.get("grossProfit", 0),
                "operating_income": latest.get("operatingIncome", 0),
                "net_income": latest.get("netIncome", 0),
                "eps": latest.get("dilutedEPS", 0)
            }
            
            result["latest"] = {k: float(v) for k, v in metrics.items()}
            
            # Calculate growth rates if previous year data is available
            if prev_year is not None:
                growth_rates = {}
                
                for metric in metrics:
                    if metric in prev_year and prev_year[metric] != 0:
                        growth = (metrics[metric] - prev_year[metric]) / prev_year[metric]
                        growth_rates[f"{metric}_growth"] = float(growth)
                
                result["growth_rates"] = growth_rates
            
            # Calculate margins
            if metrics["revenue"] != 0:
                result["margins"] = {
                    "gross_margin": float(metrics["gross_profit"] / metrics["revenue"]),
                    "operating_margin": float(metrics["operating_income"] / metrics["revenue"]),
                    "net_margin": float(metrics["net_income"] / metrics["revenue"])
                }
            
            return result
        
        except Exception as e:
            logger.error(f"Error analyzing income statement: {e}")
            return {"error": str(e)}
    
    def _analyze_balance_sheet(self, balance_sheet: pd.DataFrame) -> Dict[str, Any]:
        """Analyze balance sheet."""
        result = {}
        
        try:
            # Extract latest period data
            latest = balance_sheet.iloc[:, 0]
            prev_year = balance_sheet.iloc[:, 1] if balance_sheet.shape[1] > 1 else None
            
            # Extract key metrics
            metrics = {
                "total_assets": latest.get("totalAssets", 0),
                "total_liabilities": latest.get("totalLiabilities", 0),
                "total_equity": latest.get("totalShareholderEquity", 0),
                "current_assets": latest.get("totalCurrentAssets", 0),
                "current_liabilities": latest.get("totalCurrentLiabilities", 0),
                "cash": latest.get("cashAndShortTermInvestments", 0)
            }
            
            result["latest"] = {k: float(v) for k, v in metrics.items()}
            
            # Calculate ratios
            if metrics["total_liabilities"] != 0 and metrics["total_equity"] != 0:
                result["solvency_ratios"] = {
                    "debt_to_equity": float(metrics["total_liabilities"] / metrics["total_equity"]),
                    "debt_to_assets": float(metrics["total_liabilities"] / metrics["total_assets"])
                }
            
            if metrics["current_liabilities"] != 0:
                result["liquidity_ratios"] = {
                    "current_ratio": float(metrics["current_assets"] / metrics["current_liabilities"]),
                    "cash_ratio": float(metrics["cash"] / metrics["current_liabilities"])
                }
            
            # Calculate growth rates if previous year data is available
            if prev_year is not None:
                growth_rates = {}
                
                for metric in metrics:
                    if metric in prev_year and prev_year[metric] != 0:
                        growth = (metrics[metric] - prev_year[metric]) / prev_year[metric]
                        growth_rates[f"{metric}_growth"] = float(growth)
                
                result["growth_rates"] = growth_rates
            
            return result
        
        except Exception as e:
            logger.error(f"Error analyzing balance sheet: {e}")
            return {"error": str(e)}
    
    def _analyze_cash_flow(self, cash_flow: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cash flow statement."""
        result = {}
        
        try:
            # Extract latest period data
            latest = cash_flow.iloc[:, 0]
            prev_year = cash_flow.iloc[:, 1] if cash_flow.shape[1] > 1 else None
            
            # Extract key metrics
            metrics = {
                "operating_cash_flow": latest.get("operatingCashflow", 0),
                "capital_expenditure": latest.get("capitalExpenditures", 0),
                "free_cash_flow": latest.get("freeCashFlow", 0),
                "dividend_paid": latest.get("dividendPaid", 0)
            }
            
            result["latest"] = {k: float(v) for k, v in metrics.items()}
            
            # Calculate growth rates if previous year data is available
            if prev_year is not None:
                growth_rates = {}
                
                for metric in metrics:
                    if metric in prev_year and prev_year[metric] != 0:
                        growth = (metrics[metric] - prev_year[metric]) / prev_year[metric]
                        growth_rates[f"{metric}_growth"] = float(growth)
                
                result["growth_rates"] = growth_rates
            
            return result
        
        except Exception as e:
            logger.error(f"Error analyzing cash flow: {e}")
            return {"error": str(e)}
    
    def _calculate_financial_ratios(
        self, income_statement: pd.DataFrame, balance_sheet: pd.DataFrame, cash_flow: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate key financial ratios."""
        result = {}
        
        try:
            # Extract latest period data
            latest_income = income_statement.iloc[:, 0]
            latest_balance = balance_sheet.iloc[:, 0]
            latest_cash_flow = cash_flow.iloc[:, 0] if not cash_flow.empty else None
            
            # Profitability ratios
            if "totalShareholderEquity" in latest_balance and "netIncome" in latest_income:
                equity = latest_balance["totalShareholderEquity"]
                net_income = latest_income["netIncome"]
                
                if equity != 0:
                    result["roe"] = float(net_income / equity)  # Return on Equity
            
            if "totalAssets" in latest_balance and "netIncome" in latest_income:
                assets = latest_balance["totalAssets"]
                net_income = latest_income["netIncome"]
                
                if assets != 0:
                    result["roa"] = float(net_income / assets)  # Return on Assets
            
            # Efficiency ratios
            if "totalRevenue" in latest_income and "totalAssets" in latest_balance:
                revenue = latest_income["totalRevenue"]
                assets = latest_balance["totalAssets"]
                
                if assets != 0:
                    result["asset_turnover"] = float(revenue / assets)
            
            # Valuation ratios
            # These would typically include market data like current stock price
            # which we don't have in this implementation
            
            # Cash flow ratios
            if latest_cash_flow is not None and "operatingCashflow" in latest_cash_flow and "netIncome" in latest_income:
                op_cash_flow = latest_cash_flow["operatingCashflow"]
                net_income = latest_income["netIncome"]
                
                if net_income != 0:
                    result["cash_flow_to_income"] = float(op_cash_flow / net_income)
            
            return result
        
        except Exception as e:
            logger.error(f"Error calculating financial ratios: {e}")
            return {"error": str(e)}
    
    def _longest_streak(self, bool_series: pd.Series) -> int:
        """Calculate longest streak of True values."""
        longest = current = 0
        
        for val in bool_series:
            if val:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        
        return longest
