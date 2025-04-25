"""
Strategy page component for the Streamlit application.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional

from visualization.charts import create_strategy_visualization
from ui.components.widgets import tabbed_sections, expandable_section, gauge_chart, risk_gauge

def render_strategy_page(
    symbol: str,
    historical_data: pd.DataFrame,
    strategy: Dict[str, Any],
    insights: Dict[str, Any]
) -> None:
    """
    Render the strategy page content.
    
    Args:
        symbol: Stock symbol
        historical_data: Historical price data
        strategy: Strategy recommendations
        insights: Generated insights
    """
    st.markdown(f"# Investment Strategy: {symbol}")
    
    # Extract key components
    overall = strategy.get("overall", {})
    short_term = strategy.get("short_term", {})
    medium_term = strategy.get("medium_term", {})
    long_term = strategy.get("long_term", {})
    entry_exit = strategy.get("entry_exit", {})
    risk_management = strategy.get("risk_management", {})
    
    # Strategy visualization
    if overall:
        # Create strategy visualization
        fig, summary = create_strategy_visualization(
            strategy, historical_data, f"Investment Strategy: {symbol}"
        )
        
        # Display strategy chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Display strategy summary
        st.markdown(summary, unsafe_allow_html=True)
    else:
        st.info("No overall strategy recommendation available")
    
    # Strategy tabs by time horizon
    st.markdown("## Strategy by Time Horizon")
    
    # Define content rendering functions for each tab
    def render_short():
        render_time_horizon_strategy(short_term, historical_data, "short")
    
    def render_medium():
        render_time_horizon_strategy(medium_term, historical_data, "medium")
    
    def render_long():
        render_time_horizon_strategy(long_term, historical_data, "long")
    
    # Render tabbed sections
    tabbed_sections(
        ["Short-term Strategy", "Medium-term Strategy", "Long-term Strategy"],
        [render_short, render_medium, render_long]
    )
    
    # Entry and exit points
    st.markdown("## Entry and Exit Points")
    render_entry_exit_points(entry_exit, historical_data)
    
    # Risk management
    st.markdown("## Risk Management")
    render_risk_management(risk_management, insights.get("risk", ""))

def render_time_horizon_strategy(
    strategy_data: Dict[str, Any], historical_data: pd.DataFrame, horizon: str
) -> None:
    """
    Render strategy for a specific time horizon.
    
    Args:
        strategy_data: Strategy data for the time horizon
        historical_data: Historical price data
        horizon: Time horizon type (short, medium, long)
    """
    if not strategy_data:
        st.info(f"No {horizon}-term strategy available")
        return
    
    # Extract key info
    recommendation = strategy_data.get("recommendation", "HOLD")
    confidence = strategy_data.get("confidence", 0.5)
    reasoning = strategy_data.get("reasoning", "No detailed reasoning available.")
    
    # Color based on recommendation
    rec_color = "green" if recommendation == "BUY" else "red" if recommendation == "SELL" else "blue"
    
    # Display recommendation
    st.markdown(
        f"<h3 style='color: {rec_color};'>{recommendation} - {confidence:.0%} Confidence</h3>",
        unsafe_allow_html=True
    )
    
    # Display recommendation details in columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Reasoning
        st.markdown("### Reasoning")
        st.write(reasoning)
    
    with col2:
        # Confidence gauge
        fig = gauge_chart(
            value=confidence,
            title="Confidence Level",
            min_value=0,
            max_value=1,
            threshold_ranges=[
                (0, 0.33, "red"),
                (0.33, 0.66, "yellow"),
                (0.66, 1, "green")
            ]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional details based on time horizon
    if horizon == "short":
        render_short_term_details(strategy_data)
    elif horizon == "medium":
        render_medium_term_details(strategy_data)
    elif horizon == "long":
        render_long_term_details(strategy_data)

def render_short_term_details(strategy_data: Dict[str, Any]) -> None:
    """
    Render short-term strategy details.
    
    Args:
        strategy_data: Short-term strategy data
    """
    # Key levels
    key_levels = strategy_data.get("key_levels", {})
    if key_levels:
        st.markdown("### Key Price Levels")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Support levels
            support = key_levels.get("support", [])
            if support:
                st.markdown("#### Support Levels")
                for level in support:
                    st.markdown(f"- ${level:.2f}")
            else:
                st.markdown("No support levels identified")
        
        with col2:
            # Resistance levels
            resistance = key_levels.get("resistance", [])
            if resistance:
                st.markdown("#### Resistance Levels")
                for level in resistance:
                    st.markdown(f"- ${level:.2f}")
            else:
                st.markdown("No resistance levels identified")
    
    # Catalysts and risks
    col1, col2 = st.columns(2)
    
    with col1:
        # Catalysts
        catalysts = strategy_data.get("catalysts", [])
        if catalysts:
            st.markdown("### Potential Catalysts")
            for catalyst in catalysts:
                st.markdown(f"- {catalyst}")
    
    with col2:
        # Risk factors
        risks = strategy_data.get("risk_factors", [])
        if risks:
            st.markdown("### Risk Factors")
            for risk in risks:
                st.markdown(f"- {risk}")

def render_medium_term_details(strategy_data: Dict[str, Any]) -> None:
    """
    Render medium-term strategy details.
    
    Args:
        strategy_data: Medium-term strategy data
    """
    # Price targets
    price_targets = strategy_data.get("price_targets", {})
    if price_targets:
        st.markdown("### Price Targets")
        
        col1, col2 = st.columns(2)
        
        with col1:
            upper = price_targets.get("upper", 0)
            st.metric("Upper Target", f"${upper:.2f}")
        
        with col2:
            lower = price_targets.get("lower", 0)
            st.metric("Lower Target", f"${lower:.2f}")
    
    # Catalysts and sector trends
    col1, col2 = st.columns(2)
    
    with col1:
        # Catalysts
        catalysts = strategy_data.get("catalysts", [])
        if catalysts:
            st.markdown("### Potential Catalysts")
            for catalyst in catalysts:
                st.markdown(f"- {catalyst}")
        
        # Risk factors (below catalysts)
        risks = strategy_data.get("risk_factors", [])
        if risks:
            st.markdown("### Risk Factors")
            for risk in risks:
                st.markdown(f"- {risk}")
    
    with col2:
        # Sector trends
        sector_trends = strategy_data.get("sector_trends", [])
        if sector_trends:
            st.markdown("### Sector Trends")
            for trend in sector_trends:
                st.markdown(f"- {trend}")

def render_long_term_details(strategy_data: Dict[str, Any]) -> None:
    """
    Render long-term strategy details.
    
    Args:
        strategy_data: Long-term strategy data
    """
    # Growth and competitive position
    col1, col2, col3 = st.columns(3)
    
    with col1:
        growth = strategy_data.get("growth_potential", "Medium")
        st.metric("Growth Potential", growth)
    
    with col2:
        competitive = strategy_data.get("competitive_position", "Moderate")
        st.metric("Competitive Position", competitive)
    
    with col3:
        valuation = strategy_data.get("valuation_assessment", "Fairly Valued")
        st.metric("Valuation Assessment", valuation)
    
    # Long-term catalysts and headwinds
    col1, col2 = st.columns(2)
    
    with col1:
        # Catalysts
        catalysts = strategy_data.get("long_term_catalysts", [])
        if catalysts:
            st.markdown("### Long-term Catalysts")
            for catalyst in catalysts:
                st.markdown(f"- {catalyst}")
    
    with col2:
        # Headwinds
        headwinds = strategy_data.get("long_term_headwinds", [])
        if headwinds:
            st.markdown("### Long-term Headwinds")
            for headwind in headwinds:
                st.markdown(f"- {headwind}")

def render_entry_exit_points(
    entry_exit: Dict[str, Any], historical_data: pd.DataFrame
) -> None:
    """
    Render entry and exit points analysis.
    
    Args:
        entry_exit: Entry and exit points data
        historical_data: Historical price data
    """
    if not entry_exit:
        st.info("No entry and exit points analysis available")
        return
    
    # Extract current price for reference
    current_price = historical_data["close"].iloc[-1] if not historical_data.empty else 0
    
    # Extract entry points
    entry_points = entry_exit.get("entry_points", [])
    
    if entry_points:
        st.markdown("### Entry Points")
        
        for i, entry in enumerate(entry_points):
            col1, col2 = st.columns([1, 3])
            
            price = entry.get("price", 0)
            reasoning = entry.get("reasoning", "No reasoning provided")
            time_frame = entry.get("time_frame", "medium")
            
            with col1:
                # Calculate distance from current price
                if current_price > 0:
                    distance = (price - current_price) / current_price * 100
                    st.metric(
                        label=f"Entry Price {i+1}",
                        value=f"${price:.2f}",
                        delta=f"{distance:.1f}% from current" if distance else None,
                        delta_color="normal" if distance >= 0 else "inverse"
                    )
                else:
                    st.metric(label=f"Entry Price {i+1}", value=f"${price:.2f}")
                
                st.caption(f"Time frame: {time_frame}")
            
            with col2:
                st.markdown(f"**Reasoning:** {reasoning}")
    
    # Extract exit points
    exit_points = entry_exit.get("exit_points", [])
    
    if exit_points:
        st.markdown("### Exit Points")
        
        for i, exit in enumerate(exit_points):
            col1, col2 = st.columns([1, 3])
            
            price = exit.get("price", 0)
            reasoning = exit.get("reasoning", "No reasoning provided")
            time_frame = exit.get("time_frame", "medium")
            
            with col1:
                # Calculate distance from current price
                if current_price > 0:
                    distance = (price - current_price) / current_price * 100
                    st.metric(
                        label=f"Exit Price {i+1}",
                        value=f"${price:.2f}",
                        delta=f"{distance:.1f}% from current" if distance else None,
                        delta_color="normal" if distance >= 0 else "inverse"
                    )
                else:
                    st.metric(label=f"Exit Price {i+1}", value=f"${price:.2f}")
                
                st.caption(f"Time frame: {time_frame}")
            
            with col2:
                st.markdown(f"**Reasoning:** {reasoning}")
    
    # Stop loss
    stop_loss = entry_exit.get("stop_loss", {})
    
    if stop_loss:
        st.markdown("### Stop Loss")
        
        col1, col2 = st.columns([1, 3])
        
        price = stop_loss.get("price", 0)
        reasoning = stop_loss.get("reasoning", "No reasoning provided")
        
        with col1:
            # Calculate distance from current price
            if current_price > 0:
                distance = (price - current_price) / current_price * 100
                st.metric(
                    label="Stop Loss",
                    value=f"${price:.2f}",
                    delta=f"{distance:.1f}% from current" if distance else None,
                    delta_color="inverse"  # Always red for stop loss
                )
            else:
                st.metric(label="Stop Loss", value=f"${price:.2f}")
        
        with col2:
            st.markdown(f"**Reasoning:** {reasoning}")

def render_risk_management(
    risk_management: Dict[str, Any], risk_insights: str
) -> None:
    """
    Render risk management analysis.
    
    Args:
        risk_management: Risk management data
        risk_insights: Risk assessment insights
    """
    if not risk_management:
        st.info("No risk management analysis available")
        return
    
    # Display risk insights
    if risk_insights:
        st.markdown("### Risk Assessment")
        st.write(risk_insights)
    
    # Risk level and score
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk level
        risk_level = risk_management.get("risk_level", "medium")
        risk_score = risk_management.get("risk_score", 5.0)
        
        st.markdown(f"### Risk Level: {risk_level.upper()}")
        st.write(f"Risk Score: {risk_score}/10")
        
        # Position sizing
        st.markdown("### Position Sizing")
        st.write(risk_management.get("position_sizing", "Standard position sizing recommended"))
    
    with col2:
        # Risk gauge
        fig = risk_gauge(
            risk_score=risk_score,
            title="Risk Assessment"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Diversification and hedging
    col1, col2 = st.columns(2)
    
    with col1:
        # Diversification
        st.markdown("### Diversification Strategy")
        st.write(risk_management.get("diversification", "No diversification strategy provided"))
    
    with col2:
        # Hedging strategies
        hedging = risk_management.get("hedging_strategies", [])
        if hedging:
            st.markdown("### Hedging Strategies")
            for strategy in hedging:
                st.markdown(f"- {strategy}")
    
    # Risk mitigation tactics
    tactics = risk_management.get("risk_mitigation_tactics", [])
    if tactics:
        st.markdown("### Risk Mitigation Tactics")
        for tactic in tactics:
            st.markdown(f"- {tactic}")
