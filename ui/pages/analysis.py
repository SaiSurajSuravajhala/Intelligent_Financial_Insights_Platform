"""
Analysis page component for the Streamlit application.
"""
import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional

from visualization.charts import (
    create_candlestick_chart, 
    add_technical_indicators,
    create_indicator_chart, 
    create_volume_profile_chart,
    create_financial_metrics_radar
)
from ui.components.widgets import tabbed_sections, expandable_section

def render_analysis_page(
    symbol: str,
    historical_data: pd.DataFrame,
    analysis_results: Dict[str, Any],
    insights: Dict[str, Any],
    company_info: Dict[str, Any]
) -> None:
    """
    Render the analysis page content.
    
    Args:
        symbol: Stock symbol
        historical_data: Historical price data
        analysis_results: Analysis results
        insights: Generated insights
        company_info: Company information
    """
    st.markdown(f"# Technical & Fundamental Analysis: {symbol}")
    
    # Extract key components from results
    technical = analysis_results.get("technical", {})
    fundamental = analysis_results.get("fundamental", {})
    patterns = analysis_results.get("patterns", {})
    performance = analysis_results.get("performance", {})
    volatility = analysis_results.get("volatility", {})
    
    # Create tabs for different analysis types
    tabs = ["Technical Analysis", "Fundamental Analysis", "Performance Metrics", "Volatility Analysis"]
    
    # Define content rendering functions for each tab
    def render_technical():
        render_technical_analysis(symbol, historical_data, technical, patterns, insights)
    
    def render_fundamental():
        render_fundamental_analysis(symbol, fundamental, company_info, insights)
    
    def render_performance():
        render_performance_analysis(symbol, performance, historical_data)
    
    def render_volatility():
        render_volatility_analysis(symbol, volatility, historical_data)
    
    # Render tabbed sections
    tabbed_sections(tabs, [render_technical, render_fundamental, render_performance, render_volatility])

def render_technical_analysis(
    symbol: str,
    historical_data: pd.DataFrame,
    technical: Dict[str, Any],
    patterns: Dict[str, Any],
    insights: Dict[str, Any]
) -> None:
    """
    Render technical analysis section.
    
    Args:
        symbol: Stock symbol
        historical_data: Historical price data
        technical: Technical analysis results
        patterns: Pattern detection results
        insights: Generated insights
    """
    # Technical insights
    technical_insights = insights.get("technical", "No technical insights available.")
    st.markdown("### Technical Analysis Insights")
    st.markdown(technical_insights)
    
    # Price chart with technical indicators
    st.markdown("### Price Chart with Technical Indicators")
    
    # Technical indicators selection
    indicator_options = ["ma_20", "ma_50", "ma_200", "bb"]
    selected_indicators = st.multiselect(
        "Select Technical Indicators", 
        options=indicator_options,
        default=["ma_50", "ma_200"]
    )
    
    # Create candlestick chart
    fig = create_candlestick_chart(historical_data, "Price Chart with Technical Indicators")
    fig = add_technical_indicators(fig, historical_data, selected_indicators)
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical indicators charts
    st.markdown("### Technical Indicators")
    
    # Indicator selection
    available_indicators = [col for col in historical_data.columns 
                           if col not in ['open', 'high', 'low', 'close', 'volume', 'symbol']]
    
    default_indicators = ['rsi_14', 'macd', 'macd_signal', 'macd_histogram'] 
    default_indicators = [i for i in default_indicators if i in available_indicators]
    
    selected_indic = st.multiselect(
        "Select Technical Indicators to Display",
        options=available_indicators,
        default=default_indicators,
        key="tech_indicator_select"
    )
    
    if selected_indic:
        # Create technical indicators chart
        indicators_chart = create_indicator_chart(
            historical_data,
            selected_indic,
            "Technical Indicators"
        )
        st.plotly_chart(indicators_chart, use_container_width=True)
    else:
        st.info("Please select at least one indicator to display")
    
    # Chart patterns section
    st.markdown("### Chart Patterns")
    
    # Display detected patterns
    patterns_detected = False
    
    # Candlestick patterns
    candlestick_patterns = patterns.get("candlestick_patterns", {})
    if candlestick_patterns:
        patterns_found = []
        
        for pattern_name, pattern_data in candlestick_patterns.items():
            if isinstance(pattern_data, list) and pattern_data:
                patterns_found.append(f"**{pattern_name.replace('_', ' ').title()}** on {', '.join(pattern_data)}")
        
        if patterns_found:
            st.markdown("#### Candlestick Patterns")
            for pattern in patterns_found:
                st.markdown(pattern)
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
    
    if detected_chart_patterns:
        st.markdown("#### Technical Chart Patterns")
        for pattern in detected_chart_patterns:
            st.markdown(pattern)
        patterns_detected = True
    
    if not patterns_detected:
        st.info("No significant chart patterns detected")
    
    # Support and resistance levels
    st.markdown("### Support and Resistance")
    
    support_resistance = technical.get("support_resistance", {})
    
    if support_resistance:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Support Levels")
            support_levels = support_resistance.get("support_levels", [])
            if support_levels:
                for level in sorted(support_levels):
                    st.markdown(f"- ${level:.2f}")
            else:
                st.markdown("No support levels identified")
        
        with col2:
            st.markdown("#### Resistance Levels")
            resistance_levels = support_resistance.get("resistance_levels", [])
            if resistance_levels:
                for level in sorted(resistance_levels):
                    st.markdown(f"- ${level:.2f}")
            else:
                st.markdown("No resistance levels identified")
    else:
        st.info("No support and resistance data available")
    
    # Volume analysis
    st.markdown("### Volume Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        volume_profile = create_volume_profile_chart(historical_data, "Volume Profile")
        st.plotly_chart(volume_profile, use_container_width=True)
    
    with col2:
        volume_analysis = technical.get("volume", {})
        if volume_analysis:
            st.markdown("#### Volume Indicators")
            st.markdown(f"**Current Volume:** {volume_analysis.get('current_volume', 0):,.0f}")
            st.markdown(f"**20-Day Avg Volume:** {volume_analysis.get('avg_volume_20d', 0):,.0f}")
            st.markdown(f"**Volume Ratio:** {volume_analysis.get('volume_ratio_to_avg', 0):.2f}")
            st.markdown(f"**Volume Trend:** {volume_analysis.get('volume_trend', 'N/A')}")
            
            if "up_down_volume_ratio" in volume_analysis:
                st.markdown(f"**Up/Down Volume Ratio:** {volume_analysis.get('up_down_volume_ratio', 0):.2f}")
            
            if "volume_pattern" in volume_analysis:
                st.markdown(f"**Volume Pattern:** {volume_analysis.get('volume_pattern', 'N/A')}")
        else:
            st.info("No volume analysis data available")

def render_fundamental_analysis(
    symbol: str,
    fundamental: Dict[str, Any],
    company_info: Dict[str, Any],
    insights: Dict[str, Any]
) -> None:
    """
    Render fundamental analysis section.
    
    Args:
        symbol: Stock symbol
        fundamental: Fundamental analysis results
        company_info: Company information
        insights: Generated insights
    """
    # Fundamental insights
    fundamental_insights = insights.get("fundamental", "No fundamental insights available.")
    st.markdown("### Fundamental Analysis Insights")
    st.markdown(fundamental_insights)
    
    # Company overview
    st.markdown("### Company Overview")
    
    # Extract company data
    company_data = fundamental.get("company", {})
    if not company_data and company_info:
        company_data = {
            "name": company_info.get("name", ""),
            "sector": company_info.get("sector", ""),
            "industry": company_info.get("industry", ""),
            "country": company_info.get("country", "")
        }
    
    if company_data:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Name:** {company_data.get('name', 'N/A')}")
            st.markdown(f"**Sector:** {company_data.get('sector', 'N/A')}")
            st.markdown(f"**Industry:** {company_data.get('industry', 'N/A')}")
        
        with col2:
            st.markdown(f"**Country:** {company_data.get('country', 'N/A')}")
            st.markdown(f"**Exchange:** {company_info.get('exchange', 'N/A')}")
            st.markdown(f"**Currency:** {company_info.get('currency', 'USD')}")
    else:
        st.info("No company overview data available")
    
    # Key financial metrics
    st.markdown("### Key Financial Metrics")
    
    key_metrics = fundamental.get("key_metrics", {})
    
    if key_metrics:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Valuation")
            st.markdown(f"**P/E Ratio:** {key_metrics.get('pe_ratio', 0):.2f}")
            st.markdown(f"**Price to Book:** {key_metrics.get('price_to_book', 0):.2f}")
            st.markdown(f"**Dividend Yield:** {key_metrics.get('dividend_yield', 0):.2%}")
        
        with col2:
            st.markdown("#### Performance")
            st.markdown(f"**52-Week High:** ${key_metrics.get('52_week_high', 0):.2f}")
            st.markdown(f"**52-Week Low:** ${key_metrics.get('52_week_low', 0):.2f}")
            st.markdown(f"**Beta:** {key_metrics.get('beta', 0):.2f}")
        
        with col3:
            st.markdown("#### Size")
            market_cap = key_metrics.get('market_cap', 0)
            market_cap_str = f"${market_cap:,.0f}" if market_cap >= 1e9 else f"${market_cap/1e6:,.1f}M"
            st.markdown(f"**Market Cap:** {market_cap_str}")
    else:
        st.info("No key financial metrics available")
    
    # Financial statements analysis
    financial_analysis = fundamental.get("financial_analysis", {})
    
    if financial_analysis:
        st.markdown("### Financial Statement Analysis")
        
        # Income statement
        income_data = financial_analysis.get("income_statement", {})
        if income_data:
            with expandable_section("Income Statement", lambda: render_income_statement(income_data), False):
                pass
        
        # Balance sheet
        balance_data = financial_analysis.get("balance_sheet", {})
        if balance_data:
            with expandable_section("Balance Sheet", lambda: render_balance_sheet(balance_data), False):
                pass
        
        # Financial ratios
        key_ratios = financial_analysis.get("key_ratios", {})
        if key_ratios:
            st.markdown("### Financial Ratios")
            
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
                radar_chart = create_financial_metrics_radar(
                    radar_data,
                    title="Financial Ratios Analysis"
                )
                st.plotly_chart(radar_chart, use_container_width=True)
        
        if not income_data and not balance_data and not key_ratios:
            st.info("No financial statement analysis available")
    else:
        st.info("No financial statement analysis available")

def render_income_statement(income_data: Dict[str, Any]) -> None:
    """
    Render income statement details.
    
    Args:
        income_data: Income statement data
    """
    # Key figures
    latest = income_data.get("latest", {})
    st.markdown("#### Key Figures")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Revenue:** ${latest.get('revenue', 0)/1e6:.1f}M")
        st.markdown(f"**Gross Profit:** ${latest.get('gross_profit', 0)/1e6:.1f}M")
        st.markdown(f"**Operating Income:** ${latest.get('operating_income', 0)/1e6:.1f}M")
    
    with col2:
        st.markdown(f"**Net Income:** ${latest.get('net_income', 0)/1e6:.1f}M")
        st.markdown(f"**EPS:** ${latest.get('eps', 0):.2f}")
    
    # Margins
    margins = income_data.get("margins", {})
    if margins:
        st.markdown("#### Margins")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**Gross Margin:** {margins.get('gross_margin', 0):.1%}")
        
        with col2:
            st.markdown(f"**Operating Margin:** {margins.get('operating_margin', 0):.1%}")
        
        with col3:
            st.markdown(f"**Net Margin:** {margins.get('net_margin', 0):.1%}")
    
    # Growth rates
    growth = income_data.get("growth_rates", {})
    if growth:
        st.markdown("#### Growth Rates")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Revenue Growth:** {growth.get('revenue_growth', 0):.1%}")
        
        with col2:
            st.markdown(f"**Net Income Growth:** {growth.get('net_income_growth', 0):.1%}")

def render_balance_sheet(balance_data: Dict[str, Any]) -> None:
    """
    Render balance sheet details.
    
    Args:
        balance_data: Balance sheet data
    """
    # Key figures
    latest = balance_data.get("latest", {})
    st.markdown("#### Key Figures")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Total Assets:** ${latest.get('total_assets', 0)/1e6:.1f}M")
        st.markdown(f"**Total Liabilities:** ${latest.get('total_liabilities', 0)/1e6:.1f}M")
        st.markdown(f"**Total Equity:** ${latest.get('total_equity', 0)/1e6:.1f}M")
    
    with col2:
        st.markdown(f"**Current Assets:** ${latest.get('current_assets', 0)/1e6:.1f}M")
        st.markdown(f"**Current Liabilities:** ${latest.get('current_liabilities', 0)/1e6:.1f}M")
        st.markdown(f"**Cash:** ${latest.get('cash', 0)/1e6:.1f}M")
    
    # Ratios
    solvency = balance_data.get("solvency_ratios", {})
    if solvency:
        st.markdown("#### Solvency Ratios")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Debt/Equity:** {solvency.get('debt_to_equity', 0):.2f}")
        
        with col2:
            st.markdown(f"**Debt/Assets:** {solvency.get('debt_to_assets', 0):.2f}")
    
    liquidity = balance_data.get("liquidity_ratios", {})
    if liquidity:
        st.markdown("#### Liquidity Ratios")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Current Ratio:** {liquidity.get('current_ratio', 0):.2f}")
        
        with col2:
            st.markdown(f"**Cash Ratio:** {liquidity.get('cash_ratio', 0):.2f}")

def render_performance_analysis(
    symbol: str,
    performance: Dict[str, Any],
    historical_data: pd.DataFrame
) -> None:
    """
    Render performance analysis section.
    
    Args:
        symbol: Stock symbol
        performance: Performance analysis results
        historical_data: Historical price data
    """
    st.markdown("### Performance Metrics")
    
    # Performance table
    if performance:
        # Create a table of performance metrics
        metrics_table = []
        
        # Collect time period returns
        period_mapping = {
            "1d_return_pct": "1 Day",
            "1w_return_pct": "1 Week",
            "1m_return_pct": "1 Month",
            "3m_return_pct": "3 Months",
            "6m_return_pct": "6 Months",
            "ytd_return_pct": "Year-to-Date",
            "1y_return_pct": "1 Year",
            "3y_return_pct": "3 Years",
            "5y_return_pct": "5 Years"
        }
        
        for key, display in period_mapping.items():
            if key in performance:
                metrics_table.append({
                    "Period": display,
                    "Return (%)": f"{performance[key]:.2f}%"
                })
        
        # Add annualized return if available
        if "annualized_return_pct" in performance:
            metrics_table.append({
                "Period": "Annualized",
                "Return (%)": f"{performance['annualized_return_pct']:.2f}%"
            })
        
        # Display table
        if metrics_table:
            st.table(pd.DataFrame(metrics_table))
        
        # Risk-adjusted metrics
        st.markdown("### Risk-Adjusted Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if "sharpe_ratio" in performance:
                st.metric("Sharpe Ratio", f"{performance['sharpe_ratio']:.2f}")
            
        with col2:
            if "sortino_ratio" in performance:
                st.metric("Sortino Ratio", f"{performance['sortino_ratio']:.2f}")
            
        with col3:
            if "max_drawdown_pct" in performance:
                st.metric("Maximum Drawdown", f"{performance['max_drawdown_pct']:.2f}%", 
                         delta=None, delta_color="inverse")
        
        # Additional metrics if available
        if "treynor_ratio" in performance or "information_ratio" in performance:
            col1, col2 = st.columns(2)
            
            with col1:
                if "treynor_ratio" in performance:
                    st.metric("Treynor Ratio", f"{performance['treynor_ratio']:.2f}")
            
            with col2:
                if "information_ratio" in performance:
                    st.metric("Information Ratio", f"{performance['information_ratio']:.2f}")
    else:
        st.info("No performance metrics available")
    
    # Drawdown chart if available
    if "drawdown" in historical_data.columns:
        st.markdown("### Drawdown Analysis")
        
        # Create drawdown chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data["drawdown"] * 100,
            name="Drawdown",
            line=dict(color="red")
        ))
        
        fig.update_layout(
            title="Historical Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            yaxis=dict(tickformat=".1f", ticksuffix="%"),
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_volatility_analysis(
    symbol: str,
    volatility: Dict[str, Any],
    historical_data: pd.DataFrame
) -> None:
    """
    Render volatility analysis section.
    
    Args:
        symbol: Stock symbol
        volatility: Volatility analysis results
        historical_data: Historical price data
    """
    st.markdown("### Volatility Analysis")
    
    if volatility:
        # Display volatility metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 30-day volatility
            vol_30d = volatility.get("volatility_30d", volatility.get("historical_volatility_30d", 0))
            if vol_30d:
                st.metric("30-Day Volatility", f"{vol_30d * 100:.2f}%")
            
        with col2:
            # 90-day volatility
            vol_90d = volatility.get("volatility_90d", volatility.get("historical_volatility_90d", 0))
            if vol_90d:
                st.metric("90-Day Volatility", f"{vol_90d * 100:.2f}%")
            
        with col3:
            # Compare to sector/market if available
            relative_vol = volatility.get("relative_volatility", 1.0)
            if relative_vol:
                st.metric("Relative Volatility", f"{relative_vol:.2f}x Market")
        
        # Volatility analysis details
        if "extreme_moves" in volatility:
            st.markdown("### Extreme Price Movements")
            
            extreme_moves = volatility["extreme_moves"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                moves_1std = extreme_moves.get("moves_gt_1std", 0)
                pct_1std = extreme_moves.get("moves_gt_1std_pct", 0)
                st.metric(">1σ Moves", f"{moves_1std}", f"{pct_1std:.1f}%")
            
            with col2:
                moves_2std = extreme_moves.get("moves_gt_2std", 0)
                pct_2std = extreme_moves.get("moves_gt_2std_pct", 0)
                st.metric(">2σ Moves", f"{moves_2std}", f"{pct_2std:.1f}%")
            
            with col3:
                moves_3std = extreme_moves.get("moves_gt_3std", 0)
                pct_3std = extreme_moves.get("moves_gt_3std_pct", 0)
                st.metric(">3σ Moves", f"{moves_3std}", f"{pct_3std:.1f}%")
        
        # Up/down day streaks
        if "streaks" in volatility:
            st.markdown("### Price Streaks")
            
            streaks = volatility["streaks"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                up_streak = streaks.get("longest_up_streak", 0)
                st.metric("Longest Up Streak", f"{up_streak} days")
            
            with col2:
                down_streak = streaks.get("longest_down_streak", 0)
                st.metric("Longest Down Streak", f"{down_streak} days")
            
            with col3:
                current_streak = streaks.get("current_streak", 0)
                current_type = streaks.get("current_streak_type", "None")
                st.metric(f"Current {current_type.title()} Streak", f"{current_streak} days")
    else:
        st.info("No volatility analysis available")
    
    # Volatility chart if available
    vol_columns = [col for col in historical_data.columns if "volatility" in col]
    if vol_columns:
        st.markdown("### Historical Volatility")
        
        # Select volatility metric to display
        selected_vol = st.selectbox(
            "Select Volatility Metric",
            options=vol_columns,
            index=0
        )
        
        # Create volatility chart
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data[selected_vol] * 100,  # Convert to percentage
            name="Volatility",
            line=dict(color="purple")
        ))
        
        fig.update_layout(
            title=f"Historical {selected_vol.replace('_', ' ').title()}",
            xaxis_title="Date",
            yaxis_title="Annualized Volatility (%)",
            yaxis=dict(tickformat=".1f", ticksuffix="%"),
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
