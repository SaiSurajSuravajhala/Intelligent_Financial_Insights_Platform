"""
Dashboard components for the Streamlit application.
"""
import logging
import pandas as pd
import streamlit as st
import json
from typing import Dict, List, Any, Optional

from visualization.charts import (
    create_candlestick_chart, add_technical_indicators,
    create_indicator_chart, create_comparison_chart,
    create_performance_chart, create_volume_profile_chart,
    create_correlation_heatmap, create_risk_return_scatter,
    create_sentiment_timeline, create_financial_metrics_radar,
    create_strategy_visualization
)

# Set up logging
logger = logging.getLogger(__name__)

def display_dashboard(
    results: Dict[str, Any], comparison_data: Dict[str, pd.DataFrame] = None
) -> None:
    """
    Display the main dashboard with analysis results.
    
    Args:
        results: Comprehensive analysis results
        comparison_data: Optional comparison data for multiple symbols
    """
    try:
        # Extract key components from results
        symbol = results.get("symbol", "")
        company_info = results.get("company_info", {})
        current_data = results.get("current_data", {})
        historical_data = results.get("historical_data", pd.DataFrame())
        analysis = results.get("analysis", {})
        news = results.get("news", [])
        insights = results.get("insights", {})
        strategy = results.get("strategy", {})
        
        # Create main dashboard layout
        st.title(f"Financial Analysis Dashboard: {symbol}")
        
        # Company info and current price
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                display_company_info(company_info, current_data)
            
            with col2:
                display_current_price(current_data)
            
            with col3:
                display_quick_stats(analysis.get("basic_stats", {}))
        
        # Tabs for different analysis sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Price Analysis", "Technical Indicators", "News & Sentiment", 
            "Fundamental Analysis", "Strategy"
        ])
        
        # Tab 1: Price Analysis
        with tab1:
            display_price_analysis(historical_data, analysis, comparison_data)
        
        # Tab 2: Technical Indicators
        with tab2:
            display_technical_indicators(historical_data, analysis)
        
        # Tab 3: News & Sentiment
        with tab3:
            display_news_sentiment(news, insights)
        
        # Tab 4: Fundamental Analysis
        with tab4:
            display_fundamental_analysis(analysis.get("fundamental", {}), insights)
        
        # Tab 5: Strategy
        with tab5:
            display_strategy(strategy, historical_data, insights)
        
        # Bottom section - Full insights
        with st.expander("Complete Analysis Insights", expanded=False):
            display_full_insights(insights)
    
    except Exception as e:
        logger.error(f"Error displaying dashboard: {e}")
        st.error(f"Error displaying dashboard: {e}")

def display_company_info(company_info: Dict[str, Any], current_data: Dict[str, Any]) -> None:
    """
    Display company information.
    
    Args:
        company_info: Company information
        current_data: Current price data
    """
    try:
        # Extract info
        name = company_info.get("name", "")
        sector = company_info.get("sector", "")
        industry = company_info.get("industry", "")
        country = company_info.get("country", "")
        
        # Display info
        st.subheader(name)
        
        info_cols = st.columns(3)
        with info_cols[0]:
            st.write(f"**Sector:** {sector}")
        with info_cols[1]:
            st.write(f"**Industry:** {industry}")
        with info_cols[2]:
            st.write(f"**Country:** {country}")
        
        # Additional data if available
        if "market_cap" in company_info:
            market_cap = company_info.get("market_cap", 0)
            market_cap_str = f"${market_cap:,.0f}" if market_cap >= 1e9 else f"${market_cap/1e6:,.1f}M"
            st.write(f"**Market Cap:** {market_cap_str}")
    
    except Exception as e:
        logger.error(f"Error displaying company info: {e}")
        st.error(f"Error displaying company info: {e}")

def display_current_price(current_data: Dict[str, Any]) -> None:
    """
    Display current price information.
    
    Args:
        current_data: Current price data
    """
    try:
        # Extract data
        price = current_data.get("close", 0)
        change_pct = current_data.get("change_pct", 0)
        date = current_data.get("date", "")
        
        # Display current price
        st.metric(
            label=f"Current Price ({date})",
            value=f"${price:,.2f}",
            delta=f"{change_pct:.2f}%"
        )
        
        # Display OHLC
        with st.container():
            ohlc_cols = st.columns(4)
            ohlc_cols[0].metric("Open", f"${current_data.get('open', 0):,.2f}")
            ohlc_cols[1].metric("High", f"${current_data.get('high', 0):,.2f}")
            ohlc_cols[2].metric("Low", f"${current_data.get('low', 0):,.2f}")
            ohlc_cols[3].metric("Volume", f"{current_data.get('volume', 0):,.0f}")
    
    except Exception as e:
        logger.error(f"Error displaying current price: {e}")
        st.error(f"Error displaying current price: {e}")

def display_quick_stats(basic_stats: Dict[str, Any]) -> None:
    """
    Display quick statistics.
    
    Args:
        basic_stats: Basic statistics
    """
    try:
        # Get 30-day stats
        stats_30d = basic_stats.get("30d", {})
        
        if stats_30d:
            # Min/Max
            st.metric("30-Day Range", 
                      f"${stats_30d.get('price_min', 0):,.2f} - ${stats_30d.get('price_max', 0):,.2f}")
            
            # Avg Return
            avg_return = stats_30d.get("return_mean", 0) * 100
            st.metric("Avg. Daily Return (30d)", f"{avg_return:.2f}%")
            
            # Up/Down days
            pos_days = stats_30d.get("positive_days", 0)
            neg_days = stats_30d.get("negative_days", 0)
            total_days = pos_days + neg_days
            up_pct = (pos_days / total_days * 100) if total_days > 0 else 0
            
            st.metric("Up Days (30d)", f"{pos_days}/{total_days} ({up_pct:.0f}%)")
        else:
            st.write("No 30-day statistics available")
    
    except Exception as e:
        logger.error(f"Error displaying quick stats: {e}")
        st.error(f"Error displaying quick stats: {e}")

def display_price_analysis(
    historical_data: pd.DataFrame, 
    analysis: Dict[str, Any],
    comparison_data: Dict[str, pd.DataFrame] = None
) -> None:
    """
    Display price analysis tab.
    
    Args:
        historical_data: Historical price data
        analysis: Analysis results
        comparison_data: Optional comparison data for multiple symbols
    """
    try:
        # Extract performance data
        performance = analysis.get("performance", {})
        
        # Section 1: Price Chart
        st.subheader("Price Chart")
        
        # Technical indicators selection
        indicator_options = ["ma_20", "ma_50", "ma_200", "bb"]
        selected_indicators = st.multiselect(
            "Select Technical Indicators", 
            options=indicator_options,
            default=["ma_50", "ma_200"]
        )
        
        # Create candlestick chart
        fig = create_candlestick_chart(historical_data, "Price History")
        fig = add_technical_indicators(fig, historical_data, selected_indicators)
        st.plotly_chart(fig, use_container_width=True)
        
        # Section 2: Performance Metrics
        st.subheader("Performance Analysis")
        
        perf_cols = st.columns([2, 1])
        
        with perf_cols[0]:
            # Performance chart
            perf_chart = create_performance_chart(performance, "Performance by Time Period")
            st.plotly_chart(perf_chart, use_container_width=True)
        
        with perf_cols[1]:
            # Performance metrics table
            st.write("Key Performance Metrics")
            
            # Extract key metrics
            metrics_to_display = {
                "1D Return": f"{performance.get('1d_return_pct', 0):.2f}%",
                "1W Return": f"{performance.get('1w_return_pct', 0):.2f}%",
                "1M Return": f"{performance.get('1m_return_pct', 0):.2f}%",
                "3M Return": f"{performance.get('3m_return_pct', 0):.2f}%",
                "YTD Return": f"{performance.get('ytd_return_pct', 0):.2f}%",
                "1Y Return": f"{performance.get('1y_return_pct', 0):.2f}%",
                "Annual. Return": f"{performance.get('annualized_return_pct', 0):.2f}%",
                "Sharpe Ratio": f"{performance.get('sharpe_ratio', 0):.2f}",
                "Max Drawdown": f"{performance.get('max_drawdown_pct', 0):.2f}%"
            }
            
            # Display metrics
            for label, value in metrics_to_display.items():
                st.write(f"**{label}:** {value}")
        
        # Section 3: Comparison Chart (if comparison data available)
        if comparison_data and len(comparison_data) > 1:
            st.subheader("Comparative Analysis")
            
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
            
            # Risk-return scatter plot
            if len(comparison_data) >= 3:  # Only show if we have enough symbols
                risk_return = create_risk_return_scatter(
                    comparison_data,
                    title="Risk vs. Return Analysis"
                )
                st.plotly_chart(risk_return, use_container_width=True)
        
        # Section 4: Volume Profile
        st.subheader("Volume Analysis")
        
        vol_cols = st.columns(2)
        
        with vol_cols[0]:
            # Volume profile chart
            vol_profile = create_volume_profile_chart(
                historical_data,
                title="Volume Profile"
            )
            st.plotly_chart(vol_profile, use_container_width=True)
        
        with vol_cols[1]:
            # Volume analysis text (if available)
            vol_analysis = analysis.get("technical", {}).get("volume", {})
            
            if vol_analysis:
                st.write("### Volume Analysis")
                
                st.write(f"**Current Volume:** {vol_analysis.get('current_volume', 0):,.0f}")
                st.write(f"**20-Day Avg Volume:** {vol_analysis.get('avg_volume_20d', 0):,.0f}")
                st.write(f"**Volume Ratio to Avg:** {vol_analysis.get('volume_ratio_to_avg', 0):.2f}")
                st.write(f"**Volume Trend:** {vol_analysis.get('volume_trend', '')}")
                
                if "up_down_volume_ratio" in vol_analysis:
                    st.write(f"**Up/Down Volume Ratio:** {vol_analysis.get('up_down_volume_ratio', 0):.2f}")
                
                if "volume_pattern" in vol_analysis:
                    st.write(f"**Volume Pattern:** {vol_analysis.get('volume_pattern', '')}")
    
    except Exception as e:
        logger.error(f"Error displaying price analysis: {e}")
        st.error(f"Error displaying price analysis: {e}")

def display_technical_indicators(
    historical_data: pd.DataFrame, analysis: Dict[str, Any]
) -> None:
    """
    Display technical indicators tab.
    
    Args:
        historical_data: Historical price data
        analysis: Analysis results
    """
    try:
        # Extract technical analysis data
        technical = analysis.get("technical", {})
        patterns = analysis.get("patterns", {})
        
        # Section 1: Technical Indicators Chart
        st.subheader("Technical Indicators")
        
        # Indicator selection
        available_indicators = [col for col in historical_data.columns 
                               if col not in ['open', 'high', 'low', 'close', 'volume', 'symbol']]
        
        default_indicators = ['rsi_14', 'macd', 'macd_signal', 'macd_histogram'] 
        default_indicators = [i for i in default_indicators if i in available_indicators]
        
        selected_indicators = st.multiselect(
            "Select Technical Indicators to Display",
            options=available_indicators,
            default=default_indicators
        )
        
        if selected_indicators:
            # Create technical indicators chart
            indicators_chart = create_indicator_chart(
                historical_data,
                selected_indicators,
                "Technical Indicators"
            )
            st.plotly_chart(indicators_chart, use_container_width=True)
        else:
            st.info("Please select at least one indicator to display")
        
        # Section 2: Technical Analysis Results
        st.subheader("Technical Analysis")
        
        tech_cols = st.columns(2)
        
        with tech_cols[0]:
            # Trend Analysis
            trend = technical.get("trend", {})
            
            if trend:
                st.write("### Trend Analysis")
                
                st.write(f"**Overall Trend:** {trend.get('overall_trend', 'N/A')}")
                st.write(f"**Short-term Trend:** {trend.get('short_term_trend', 'N/A')}")
                
                if "trend_strength" in trend:
                    st.write(f"**Trend Strength:** {trend.get('trend_strength', 'N/A')}")
                
                if "recent_signal" in trend:
                    st.write(f"**Recent Signal:** {trend.get('recent_signal', 'N/A')}")
            
            # Support/Resistance
            support_resistance = technical.get("support_resistance", {})
            
            if support_resistance:
                st.write("### Support & Resistance")
                
                if "key_support" in support_resistance:
                    st.write(f"**Key Support:** ${support_resistance.get('key_support', 0):.2f}")
                
                if "key_resistance" in support_resistance:
                    st.write(f"**Key Resistance:** ${support_resistance.get('key_resistance', 0):.2f}")
                
                # Additional levels if available
                support_levels = support_resistance.get("support_levels", [])
                resistance_levels = support_resistance.get("resistance_levels", [])
                
                if support_levels:
                    levels_str = ", ".join([f"${level:.2f}" for level in support_levels[:3]])
                    st.write(f"**Support Levels:** {levels_str}")
                
                if resistance_levels:
                    levels_str = ", ".join([f"${level:.2f}" for level in resistance_levels[:3]])
                    st.write(f"**Resistance Levels:** {levels_str}")
        
        with tech_cols[1]:
            # RSI Analysis
            rsi = technical.get("rsi", {})
            
            if rsi:
                st.write("### RSI Analysis")
                
                st.write(f"**Current RSI:** {rsi.get('current_value', 0):.1f}")
                st.write(f"**Condition:** {rsi.get('condition', 'N/A')}")
                
                if "divergence" in rsi and rsi["divergence"] != "none":
                    st.write(f"**Divergence:** {rsi.get('divergence', 'N/A')}")
                
                st.write(f"**Trend:** {rsi.get('trend', 'N/A')}")
            
            # MACD Analysis
            macd = technical.get("macd", {})
            
            if macd:
                st.write("### MACD Analysis")
                
                current_values = macd.get("current_values", {})
                
                st.write(f"**Current MACD:** {current_values.get('macd', 0):.3f}")
                st.write(f"**Signal Line:** {current_values.get('signal', 0):.3f}")
                st.write(f"**Histogram:** {current_values.get('histogram', 0):.3f}")
                st.write(f"**Position:** {macd.get('position', 'N/A')}")
                
                if "zero_line_crossover" in macd and macd["zero_line_crossover"] != "none":
                    st.write(f"**Zero Line Crossover:** {macd.get('zero_line_crossover', 'N/A')}")
        
        # Section 3: Chart Patterns
        st.subheader("Chart Patterns")
        
        patterns_detected = False
        
        # Candlestick patterns
        candlestick_patterns = patterns.get("candlestick_patterns", {})
        if candlestick_patterns:
            patterns_found = []
            
            for pattern_name, pattern_data in candlestick_patterns.items():
                if isinstance(pattern_data, list) and pattern_data:
                    patterns_found.append(f"**{pattern_name.replace('_', ' ').title()}** on {', '.join(pattern_data)}")
            
            if patterns_found:
                st.write("### Candlestick Patterns")
                for pattern in patterns_found:
                    st.write(pattern)
                patterns_detected = True
        
        # Chart patterns
        chart_patterns = patterns.get("chart_patterns", {})
        pattern_types = ["head_and_shoulders", "double_top", "double_bottom", "flag", "pennant"]
        
        detected_chart_patterns = []
        for pattern in pattern_types:
            if pattern in chart_patterns and chart_patterns[pattern].get("detected", False):
                pattern_info = chart_patterns[pattern]
                pattern_name = pattern.replace('_', ' ').title()
                detected_chart_patterns.append(f"**{pattern_name}** detected")
                
                # Additional pattern details if available
                if pattern == "head_and_shoulders" and "head_date" in pattern_info:
                    st.write(f"Head and Shoulders pattern with head on {pattern_info['head_date']}")
                
                elif pattern == "double_top" and "price_level" in pattern_info:
                    st.write(f"Double Top at price level ${pattern_info['price_level']:.2f}")
                
                elif pattern == "double_bottom" and "price_level" in pattern_info:
                    st.write(f"Double Bottom at price level ${pattern_info['price_level']:.2f}")
                
                elif pattern == "flag" and "type" in pattern_info:
                    st.write(f"{pattern_info['type'].title()} Flag pattern")
        
        if detected_chart_patterns:
            st.write("### Chart Patterns")
            for pattern in detected_chart_patterns:
                st.write(pattern)
            patterns_detected = True
        
        # Trading range
        trading_range = patterns.get("trading_range", {})
        if trading_range.get("in_trading_range", False):
            st.write("### Trading Range")
            st.write(f"Stock is in a trading range between ${trading_range.get('range_low', 0):.2f} and ${trading_range.get('range_high', 0):.2f}")
            st.write(f"Range width: {trading_range.get('range_width_pct', 0):.1f}% of average price")
            st.write(f"Days in range: {trading_range.get('days_in_range', 0)}")
            patterns_detected = True
        
        # Gaps
        gaps = patterns.get("gaps", {})
        recent_gaps = gaps.get("recent_gaps", [])
        if recent_gaps:
            st.write("### Price Gaps")
            for gap in recent_gaps[:3]:  # Show only the 3 most recent gaps
                gap_type = gap.get("type", "").replace("_", " ").title()
                gap_date = gap.get("date", "")
                gap_pct = gap.get("gap_pct", 0)
                filled = "Filled" if gap.get("filled", False) else "Unfilled"
                
                st.write(f"**{gap_type}** of {gap_pct:.1f}% on {gap_date} ({filled})")
            patterns_detected = True
        
        if not patterns_detected:
            st.info("No significant chart patterns detected")
        
        # Section 4: Anomalies
        st.subheader("Market Anomalies")
        
        anomalies = analysis.get("anomalies", {})
        
        anomaly_types = {
            "price_anomalies": "Price Anomalies",
            "volume_anomalies": "Volume Anomalies",
            "volatility_anomalies": "Volatility Anomalies",
            "correlation_anomalies": "Correlation Anomalies"
        }
        
        anomalies_detected = False
        
        for anomaly_key, anomaly_title in anomaly_types.items():
            anomaly_list = anomalies.get(anomaly_key, {}).get("anomalies", [])
            
            if anomaly_list:
                st.write(f"### {anomaly_title}")
                
                for anomaly in anomaly_list[:3]:  # Show only the 3 most recent anomalies
                    anomaly_date = anomaly.get("date", "")
                    anomaly_type = anomaly.get("type", "").replace("_", " ").title()
                    z_score = anomaly.get("z_score", 0)
                    
                    st.write(f"**{anomaly_type}** on {anomaly_date} (z-score: {z_score:.1f})")
                anomalies_detected = True
        
        if not anomalies_detected:
            st.info("No significant market anomalies detected")
    
    except Exception as e:
        logger.error(f"Error displaying technical indicators: {e}")
        st.error(f"Error displaying technical indicators: {e}")

def display_news_sentiment(
    news: List[Dict[str, Any]], insights: Dict[str, Any]
) -> None:
    """
    Display news and sentiment tab.
    
    Args:
        news: News data
        insights: Generated insights
    """
    try:
        # Section 1: News Sentiment Timeline
        st.subheader("News Sentiment Analysis")
        
        # News insights
        news_insights = insights.get("news", "")
        if news_insights:
            st.write(news_insights)
        
        # News sentiment timeline
        if news:
            sentiment_chart = create_sentiment_timeline(news, "News Sentiment Timeline")
            st.plotly_chart(sentiment_chart, use_container_width=True)
        
        # Section 2: Recent News Articles
        st.subheader("Recent News Articles")
        
        if news:
            # Display news articles
            for i, article in enumerate(news[:10]):  # Show only the 10 most recent articles
                with st.expander(
                    f"{article.get('title', 'No Title')} - {article.get('source', 'Unknown')}"
                ):
                    # Format sentiment score with color
                    sentiment_score = article.get("sentiment_score", 0)
                    sentiment_color = "green" if sentiment_score > 0.2 else "red" if sentiment_score < -0.2 else "gray"
                    sentiment_label = "Positive" if sentiment_score > 0.2 else "Negative" if sentiment_score < -0.2 else "Neutral"
                    
                    st.write(f"**Published:** {article.get('published_at', '')}")
                    st.write(f"**Source:** {article.get('source', 'Unknown')}")
                    st.write(f"**Sentiment:** <span style='color:{sentiment_color}'>{sentiment_label} ({sentiment_score:.2f})</span>", unsafe_allow_html=True)
                    st.write("**Summary:**")
                    st.write(article.get("summary", "No summary available."))
                    
                    if "url" in article:
                        st.write(f"[Read full article]({article['url']})")
        else:
            st.info("No recent news articles available")
        
        # Section 3: Social Media Sentiment (if available)
        social_sentiment = {}  # This would come from the alternative data collector
        
        if social_sentiment:
            st.subheader("Social Media Sentiment")
            
            # Display social sentiment data
            sentiment_cols = st.columns(3)
            
            avg_sentiment = social_sentiment.get("average_sentiment", 0)
            sentiment_color = "green" if avg_sentiment > 0.2 else "red" if avg_sentiment < -0.2 else "gray"
            
            with sentiment_cols[0]:
                st.metric(
                    label="Average Sentiment",
                    value=f"{avg_sentiment:.2f}",
                    delta=None,
                    delta_color=sentiment_color
                )
            
            sentiment_count = social_sentiment.get("sentiment_count", {})
            
            with sentiment_cols[1]:
                st.metric(
                    label="Positive Mentions",
                    value=sentiment_count.get("positive", 0)
                )
            
            with sentiment_cols[2]:
                st.metric(
                    label="Negative Mentions",
                    value=sentiment_count.get("negative", 0)
                )
    
    except Exception as e:
        logger.error(f"Error displaying news and sentiment: {e}")
        st.error(f"Error displaying news and sentiment: {e}")

def display_fundamental_analysis(
    fundamental: Dict[str, Any], insights: Dict[str, Any]
) -> None:
    """
    Display fundamental analysis tab.
    
    Args:
        fundamental: Fundamental analysis data
        insights: Generated insights
    """
    try:
        # Check if we have fundamental data
        if not fundamental:
            st.info("No fundamental analysis data available")
            return
        
        # Section 1: Fundamental Insights
        st.subheader("Fundamental Analysis")
        
        # Display fundamental insights
        fundamental_insights = insights.get("fundamental", "")
        if fundamental_insights:
            st.write(fundamental_insights)
        
        # Extract key components
        company_data = fundamental.get("company", {})
        key_metrics = fundamental.get("key_metrics", {})
        financial_analysis = fundamental.get("financial_analysis", {})
        
        # Section 2: Key Metrics
        st.subheader("Key Financial Metrics")
        
        metrics_cols = st.columns(4)
        
        with metrics_cols[0]:
            # Valuation metrics
            st.write("### Valuation")
            st.write(f"**P/E Ratio:** {key_metrics.get('pe_ratio', 0):.2f}")
            st.write(f"**Price to Book:** {key_metrics.get('price_to_book', 0):.2f}")
            st.write(f"**Dividend Yield:** {key_metrics.get('dividend_yield', 0):.2%}")
        
        with metrics_cols[1]:
            # Price metrics
            st.write("### Price")
            st.write(f"**52-Week High:** ${key_metrics.get('52_week_high', 0):.2f}")
            st.write(f"**52-Week Low:** ${key_metrics.get('52_week_low', 0):.2f}")
            st.write(f"**Beta:** {key_metrics.get('beta', 0):.2f}")
        
        with metrics_cols[2]:
            # Size metrics
            st.write("### Size")
            market_cap = key_metrics.get('market_cap', 0)
            market_cap_str = f"${market_cap:,.0f}" if market_cap >= 1e9 else f"${market_cap/1e6:,.1f}M"
            st.write(f"**Market Cap:** {market_cap_str}")
        
        # Section 3: Financial Statements Analysis
        if financial_analysis:
            st.subheader("Financial Statement Analysis")
            
            # Income Statement
            income_data = financial_analysis.get("income_statement", {})
            if income_data:
                st.write("### Income Statement")
                
                income_cols = st.columns(3)
                
                with income_cols[0]:
                    # Key figures
                    latest = income_data.get("latest", {})
                    st.write(f"**Revenue:** ${latest.get('revenue', 0)/1e6:.1f}M")
                    st.write(f"**Gross Profit:** ${latest.get('gross_profit', 0)/1e6:.1f}M")
                    st.write(f"**Operating Income:** ${latest.get('operating_income', 0)/1e6:.1f}M")
                    st.write(f"**Net Income:** ${latest.get('net_income', 0)/1e6:.1f}M")
                
                with income_cols[1]:
                    # Margins
                    margins = income_data.get("margins", {})
                    st.write(f"**Gross Margin:** {margins.get('gross_margin', 0):.1%}")
                    st.write(f"**Operating Margin:** {margins.get('operating_margin', 0):.1%}")
                    st.write(f"**Net Margin:** {margins.get('net_margin', 0):.1%}")
                
                with income_cols[2]:
                    # Growth rates
                    growth = income_data.get("growth_rates", {})
                    st.write(f"**Revenue Growth:** {growth.get('revenue_growth', 0):.1%}")
                    st.write(f"**Net Income Growth:** {growth.get('net_income_growth', 0):.1%}")
            
            # Balance Sheet
            balance_data = financial_analysis.get("balance_sheet", {})
            if balance_data:
                st.write("### Balance Sheet")
                
                balance_cols = st.columns(3)
                
                with balance_cols[0]:
                    # Key figures
                    latest = balance_data.get("latest", {})
                    st.write(f"**Total Assets:** ${latest.get('total_assets', 0)/1e6:.1f}M")
                    st.write(f"**Total Liabilities:** ${latest.get('total_liabilities', 0)/1e6:.1f}M")
                    st.write(f"**Total Equity:** ${latest.get('total_equity', 0)/1e6:.1f}M")
                
                with balance_cols[1]:
                    # Ratios
                    solvency = balance_data.get("solvency_ratios", {})
                    st.write(f"**Debt/Equity:** {solvency.get('debt_to_equity', 0):.2f}")
                    st.write(f"**Debt/Assets:** {solvency.get('debt_to_assets', 0):.2f}")
                
                with balance_cols[2]:
                    # Liquidity
                    liquidity = balance_data.get("liquidity_ratios", {})
                    st.write(f"**Current Ratio:** {liquidity.get('current_ratio', 0):.2f}")
                    st.write(f"**Cash Ratio:** {liquidity.get('cash_ratio', 0):.2f}")
            
            # Financial Ratios visualization
            key_ratios = financial_analysis.get("key_ratios", {})
            if key_ratios:
                st.subheader("Financial Ratios")
                
                # Prepare data for radar chart
                radar_data = {}
                
                # Map ratio names to display names
                ratio_mapping = {
                    "roe": "Return on Equity",
                    "roa": "Return on Assets",
                    "asset_turnover": "Asset Turnover",
                    "cash_flow_to_income": "CF/Income"
                }
                
                for key, value in key_ratios.items():
                    display_name = ratio_mapping.get(key, key)
                    radar_data[display_name] = value
                
                if radar_data:
                    # Create radar chart
                    radar_chart = create_financial_metrics_radar(
                        radar_data,
                        title="Financial Ratios Analysis"
                    )
                    st.plotly_chart(radar_chart, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error displaying fundamental analysis: {e}")
        st.error(f"Error displaying fundamental analysis: {e}")

def display_strategy(
    strategy: Dict[str, Any], 
    historical_data: pd.DataFrame,
    insights: Dict[str, Any]
) -> None:
    """
    Display strategy recommendations tab.
    
    Args:
        strategy: Strategy recommendations
        historical_data: Historical price data
        insights: Generated insights
    """
    try:
        # Section 1: Overall Strategy
        st.subheader("Investment Strategy")
        
        overall = strategy.get("overall", {})
        
        if overall:
            # Strategy chart
            fig, summary = create_strategy_visualization(
                strategy, historical_data, "Investment Strategy"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Strategy summary
            st.markdown(summary, unsafe_allow_html=True)
        else:
            st.info("No strategy recommendations available")
        
        # Section 2: Time Horizon Tabs
        st.subheader("Strategy by Time Horizon")
        
        # Check if we have strategy data for different time horizons
        has_short = "short_term" in strategy
        has_medium = "medium_term" in strategy
        has_long = "long_term" in strategy
        
        if has_short or has_medium or has_long:
            horizon_tabs = st.tabs(["Short-term", "Medium-term", "Long-term"])
            
            # Short-term tab
            with horizon_tabs[0]:
                if has_short:
                    display_time_horizon_strategy(strategy["short_term"], "short")
                else:
                    st.info("No short-term strategy available")
            
            # Medium-term tab
            with horizon_tabs[1]:
                if has_medium:
                    display_time_horizon_strategy(strategy["medium_term"], "medium")
                else:
                    st.info("No medium-term strategy available")
            
            # Long-term tab
            with horizon_tabs[2]:
                if has_long:
                    display_time_horizon_strategy(strategy["long_term"], "long")
                else:
                    st.info("No long-term strategy available")
        
        # Section 3: Risk Management
        st.subheader("Risk Management")
        
        risk_mgmt = strategy.get("risk_management", {})
        
        if risk_mgmt:
            risk_cols = st.columns(2)
            
            with risk_cols[0]:
                # Risk assessment
                risk_level = risk_mgmt.get("risk_level", "medium")
                risk_score = risk_mgmt.get("risk_score", 5.0)
                
                # Color based on risk level
                risk_color = "red" if risk_level == "high" else "orange" if risk_level == "medium" else "green"
                
                st.metric(
                    label="Risk Assessment",
                    value=f"{risk_level.upper()}",
                    delta=f"Score: {risk_score}/10",
                    delta_color="off"
                )
                
                # Position sizing
                st.write(f"**Recommended Position Size:** {risk_mgmt.get('position_sizing', 'N/A')}")
                
                # Risk mitigation tactics
                tactics = risk_mgmt.get("risk_mitigation_tactics", [])
                if tactics:
                    st.write("### Risk Mitigation Tactics")
                    for tactic in tactics:
                        st.write(f"- {tactic}")
            
            with risk_cols[1]:
                # Diversification
                st.write("### Diversification Strategy")
                st.write(risk_mgmt.get("diversification", "No diversification strategy available."))
                
                # Hedging
                hedging = risk_mgmt.get("hedging_strategies", [])
                if hedging:
                    st.write("### Hedging Strategies")
                    for strategy in hedging:
                        st.write(f"- {strategy}")
    
    except Exception as e:
        logger.error(f"Error displaying strategy: {e}")
        st.error(f"Error displaying strategy: {e}")

def display_time_horizon_strategy(strategy_data: Dict[str, Any], horizon: str) -> None:
    """
    Display strategy for a specific time horizon.
    
    Args:
        strategy_data: Strategy data for the time horizon
        horizon: Time horizon type (short, medium, long)
    """
    try:
        # Extract key info
        recommendation = strategy_data.get("recommendation", "HOLD")
        confidence = strategy_data.get("confidence", 0.5)
        reasoning = strategy_data.get("reasoning", "No detailed reasoning available.")
        
        # Color based on recommendation
        rec_color = "green" if recommendation == "BUY" else "red" if recommendation == "SELL" else "blue"
        
        # Display recommendation
        st.markdown(
            f"<h2 style='color: {rec_color};'>{recommendation} - {confidence:.0%} Confidence</h2>",
            unsafe_allow_html=True
        )
        
        # Display reasoning
        st.write("### Reasoning")
        st.write(reasoning)
        
        # Display additional info based on time horizon
        if horizon == "short":
            # Key levels
            key_levels = strategy_data.get("key_levels", {})
            if key_levels:
                st.write("### Key Price Levels")
                
                support = key_levels.get("support", [])
                resistance = key_levels.get("resistance", [])
                
                if support:
                    support_str = ", ".join([f"${level:.2f}" for level in support[:3]])
                    st.write(f"**Support Levels:** {support_str}")
                
                if resistance:
                    resistance_str = ", ".join([f"${level:.2f}" for level in resistance[:3]])
                    st.write(f"**Resistance Levels:** {resistance_str}")
            
            # Catalysts and risks
            catalysts = strategy_data.get("catalysts", [])
            risks = strategy_data.get("risk_factors", [])
            
            if catalysts:
                st.write("### Potential Catalysts")
                for catalyst in catalysts:
                    st.write(f"- {catalyst}")
            
            if risks:
                st.write("### Risk Factors")
                for risk in risks:
                    st.write(f"- {risk}")
        
        elif horizon == "medium":
            # Price targets
            price_targets = strategy_data.get("price_targets", {})
            if price_targets:
                st.write("### Price Targets")
                st.write(f"**Upper Target:** ${price_targets.get('upper', 0):.2f}")
                st.write(f"**Lower Target:** ${price_targets.get('lower', 0):.2f}")
            
            # Catalysts and sector trends
            catalysts = strategy_data.get("catalysts", [])
            sector_trends = strategy_data.get("sector_trends", [])
            risks = strategy_data.get("risk_factors", [])
            
            if catalysts:
                st.write("### Potential Catalysts")
                for catalyst in catalysts:
                    st.write(f"- {catalyst}")
            
            if sector_trends:
                st.write("### Sector Trends")
                for trend in sector_trends:
                    st.write(f"- {trend}")
            
            if risks:
                st.write("### Risk Factors")
                for risk in risks:
                    st.write(f"- {risk}")
        
        elif horizon == "long":
            # Growth and competitive position
            growth = strategy_data.get("growth_potential", "Medium")
            competitive = strategy_data.get("competitive_position", "Moderate")
            valuation = strategy_data.get("valuation_assessment", "Fairly Valued")
            
            st.write("### Long-term Assessment")
            st.write(f"**Growth Potential:** {growth}")
            st.write(f"**Competitive Position:** {competitive}")
            st.write(f"**Valuation Assessment:** {valuation}")
            
            # Long-term catalysts and headwinds
            catalysts = strategy_data.get("long_term_catalysts", [])
            headwinds = strategy_data.get("long_term_headwinds", [])
            
            if catalysts:
                st.write("### Long-term Catalysts")
                for catalyst in catalysts:
                    st.write(f"- {catalyst}")
            
            if headwinds:
                st.write("### Long-term Headwinds")
                for headwind in headwinds:
                    st.write(f"- {headwind}")
    
    except Exception as e:
        logger.error(f"Error displaying time horizon strategy: {e}")
        st.error(f"Error displaying time horizon strategy: {e}")

def display_full_insights(insights: Dict[str, Any]) -> None:
    """
    Display full insights in an expandable section.
    
    Args:
        insights: All generated insights
    """
    try:
        # Create tabs for different insight types
        tabs = st.tabs([
            "Summary", "Technical", "Fundamental", "News", "Risk", "Market Context"
        ])
        
        # Summary tab
        with tabs[0]:
            st.write(insights.get("summary", "No summary insights available."))
        
        # Technical tab
        with tabs[1]:
            st.write(insights.get("technical", "No technical insights available."))
        
        # Fundamental tab
        with tabs[2]:
            st.write(insights.get("fundamental", "No fundamental insights available."))
        
        # News tab
        with tabs[3]:
            st.write(insights.get("news", "No news insights available."))
        
        # Risk tab
        with tabs[4]:
            st.write(insights.get("risk", "No risk insights available."))
        
        # Market Context tab
        with tabs[5]:
            st.write(insights.get("market_context", "No market context available."))
    
    except Exception as e:
        logger.error(f"Error displaying full insights: {e}")
        st.error(f"Error displaying full insights: {e}")
