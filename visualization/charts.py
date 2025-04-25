"""
Chart generation functions for financial data visualization.
"""
import logging
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# Set up logging
logger = logging.getLogger(__name__)

def create_candlestick_chart(data: pd.DataFrame, title: str = "Stock Price") -> go.Figure:
    """
    Create a candlestick chart with volume subplot.
    
    Args:
        data: DataFrame with OHLCV data
        title: Chart title
        
    Returns:
        Plotly figure
    """
    try:
        # Create subplots with 2 rows (price and volume)
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(title, "Volume"),
            row_heights=[0.7, 0.3]
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Add volume bar chart
        colors = ['red' if row['open'] > row['close'] else 'green' for _, row in data.iterrows()]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                marker_color=colors,
                name="Volume"
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=600,
            width=900,
            showlegend=False,
            template="plotly_white"
        )
        
        # Update y-axis format
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating candlestick chart: {e}")
        # Return simple empty figure
        return go.Figure()

def add_technical_indicators(
    fig: go.Figure, data: pd.DataFrame, indicators: List[str] = None
) -> go.Figure:
    """
    Add technical indicators to a candlestick chart.
    
    Args:
        fig: Plotly figure with candlestick chart
        data: DataFrame with price data and indicators
        indicators: List of indicators to add
        
    Returns:
        Updated Plotly figure
    """
    try:
        if indicators is None:
            # Default indicators
            indicators = ['ma_20', 'ma_50', 'ma_200']
        
        # Add selected indicators
        for indicator in indicators:
            if indicator in data.columns:
                # Moving averages
                if indicator.startswith('ma_') or indicator.startswith('ema_'):
                    period = indicator.split('_')[1]
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data[indicator],
                            name=f"{indicator.split('_')[0].upper()} {period}",
                            line=dict(width=1)
                        ),
                        row=1, col=1
                    )
                
                # Bollinger Bands
                elif indicator == 'bb' and all(col in data.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['bb_upper'],
                            name="BB Upper",
                            line=dict(width=1, dash='dash', color='rgba(0,176,246,0.7)')
                        ),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['bb_middle'],
                            name="BB Middle",
                            line=dict(width=1, color='rgba(0,176,246,0.7)')
                        ),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['bb_lower'],
                            name="BB Lower",
                            line=dict(width=1, dash='dash', color='rgba(0,176,246,0.7)')
                        ),
                        row=1, col=1
                    )
        
        # Update layout
        fig.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        
        return fig
    
    except Exception as e:
        logger.error(f"Error adding technical indicators: {e}")
        # Return the original figure
        return fig

def create_indicator_chart(
    data: pd.DataFrame, indicators: List[str], title: str = "Technical Indicators"
) -> go.Figure:
    """
    Create a chart with technical indicators.
    
    Args:
        data: DataFrame with price data and indicators
        indicators: List of indicators to include
        title: Chart title
        
    Returns:
        Plotly figure
    """
    try:
        # Create subplots
        num_indicators = len(indicators)
        fig = make_subplots(
            rows=num_indicators,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=indicators,
            row_heights=[1/num_indicators] * num_indicators
        )
        
        # Add indicators
        for i, indicator in enumerate(indicators, 1):
            if indicator in data.columns:
                # Simple indicators
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[indicator],
                        name=indicator,
                        line=dict(width=1)
                    ),
                    row=i, col=1
                )
                
                # Add reference lines for certain indicators
                if indicator == 'rsi_14':
                    # Add overbought/oversold lines
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=i, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=i, col=1)
                elif indicator == 'macd':
                    # Add signal and histogram
                    if 'macd_signal' in data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=data['macd_signal'],
                                name='MACD Signal',
                                line=dict(width=1, color='orange')
                            ),
                            row=i, col=1
                        )
                    if 'macd_histogram' in data.columns:
                        colors = ['red' if val < 0 else 'green' for val in data['macd_histogram']]
                        fig.add_trace(
                            go.Bar(
                                x=data.index,
                                y=data['macd_histogram'],
                                name='MACD Histogram',
                                marker_color=colors
                            ),
                            row=i, col=1
                        )
        
        # Update layout
        fig.update_layout(
            height=100 * num_indicators + 100,
            width=900,
            showlegend=True,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating indicator chart: {e}")
        # Return simple empty figure
        return go.Figure()

def create_comparison_chart(
    data_dict: Dict[str, pd.DataFrame], column: str = 'close', 
    normalize: bool = True, title: str = "Price Comparison"
) -> go.Figure:
    """
    Create a comparison chart for multiple symbols.
    
    Args:
        data_dict: Dictionary of {symbol: DataFrame}
        column: Column to compare
        normalize: Whether to normalize to percentage change
        title: Chart title
        
    Returns:
        Plotly figure
    """
    try:
        # Create figure
        fig = go.Figure()
        
        # Add lines for each symbol
        for symbol, data in data_dict.items():
            if column in data.columns:
                y_values = data[column]
                
                # Normalize to percentage change from first value if requested
                if normalize and len(y_values) > 0:
                    first_value = y_values.iloc[0]
                    if first_value != 0:
                        y_values = (y_values / first_value - 1) * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=y_values,
                        name=symbol,
                        line=dict(width=2)
                    )
                )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=500,
            width=900,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_title="% Change" if normalize else column.capitalize()
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating comparison chart: {e}")
        # Return simple empty figure
        return go.Figure()

def create_performance_chart(performance: Dict[str, float], title: str = "Performance Metrics") -> go.Figure:
    """
    Create a bar chart of performance metrics.
    
    Args:
        performance: Dictionary of {period: return_pct}
        title: Chart title
        
    Returns:
        Plotly figure
    """
    try:
        # Extract periods and values
        periods = []
        values = []
        
        # Sort by time period
        period_order = {
            "1d_return_pct": 0, "1w_return_pct": 1, "1m_return_pct": 2,
            "3m_return_pct": 3, "6m_return_pct": 4, "1y_return_pct": 5,
            "3y_return_pct": 6, "5y_return_pct": 7
        }
        
        # Filter for only return percentage metrics
        return_metrics = {k: v for k, v in performance.items() if k.endswith('_return_pct')}
        
        # Sort by time period
        for key, value in sorted(return_metrics.items(), key=lambda x: period_order.get(x[0], 99)):
            # Clean up period name for display
            display_name = key.replace('_return_pct', '').upper()
            periods.append(display_name)
            values.append(value)
        
        # Create colors based on values
        colors = ['green' if val >= 0 else 'red' for val in values]
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=periods,
                y=values,
                marker_color=colors,
                text=[f"{v:.2f}%" for v in values],
                textposition='auto'
            )
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=400,
            width=700,
            template="plotly_white",
            yaxis_title="Return (%)",
            xaxis_title="Time Period"
        )
        
        # Add reference line at 0
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating performance chart: {e}")
        # Return simple empty figure
        return go.Figure()

def create_volume_profile_chart(
    data: pd.DataFrame, bins: int = 20, title: str = "Volume Profile"
) -> go.Figure:
    """
    Create a volume profile chart.
    
    Args:
        data: DataFrame with price and volume data
        bins: Number of price bins
        title: Chart title
        
    Returns:
        Plotly figure
    """
    try:
        # Create price bins
        price_min = data['low'].min()
        price_max = data['high'].max()
        bin_size = (price_max - price_min) / bins
        
        price_bins = [price_min + i * bin_size for i in range(bins + 1)]
        bin_centers = [(price_bins[i] + price_bins[i+1]) / 2 for i in range(bins)]
        
        # Calculate volume in each price bin
        volume_profile = np.zeros(bins)
        
        for _, row in data.iterrows():
            # Determine which bins this candle spans
            low_bin = max(0, min(bins - 1, int((row['low'] - price_min) / bin_size)))
            high_bin = max(0, min(bins - 1, int((row['high'] - price_min) / bin_size)))
            
            # Simple approach: distribute volume equally across spanned bins
            if high_bin >= low_bin:
                vol_per_bin = row['volume'] / (high_bin - low_bin + 1)
                for b in range(low_bin, high_bin + 1):
                    volume_profile[b] += vol_per_bin
        
        # Create figure with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price"
            ),
            secondary_y=False
        )
        
        # Add volume profile as horizontal bar chart
        fig.add_trace(
            go.Bar(
                x=volume_profile,
                y=bin_centers,
                orientation='h',
                name="Volume Profile",
                marker=dict(color='rgba(0,0,255,0.3)')
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=600,
            width=900,
            template="plotly_white",
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        # Update axes
        fig.update_yaxes(title_text="Price", secondary_y=False)
        fig.update_yaxes(title_text="", showgrid=False, secondary_y=True)
        fig.update_xaxes(title_text="Date")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating volume profile chart: {e}")
        # Return simple empty figure
        return go.Figure()

def create_correlation_heatmap(
    data_dict: Dict[str, pd.DataFrame], column: str = 'return',
    title: str = "Correlation Heatmap"
) -> go.Figure:
    """
    Create a correlation heatmap for multiple symbols.
    
    Args:
        data_dict: Dictionary of {symbol: DataFrame}
        column: Column to calculate correlation for
        title: Chart title
        
    Returns:
        Plotly figure
    """
    try:
        # Extract the specified column from each DataFrame
        symbol_data = {}
        for symbol, df in data_dict.items():
            if column in df.columns:
                symbol_data[symbol] = df[column]
        
        # Create a DataFrame with all symbols' data
        correlation_df = pd.DataFrame(symbol_data)
        
        # Calculate correlation matrix
        corr_matrix = correlation_df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            height=600,
            width=800,
            template="plotly_white",
            xaxis_title="",
            yaxis_title=""
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {e}")
        # Return simple empty figure
        return go.Figure()

def create_risk_return_scatter(
    data_dict: Dict[str, pd.DataFrame], window: int = 90,
    title: str = "Risk vs Return"
) -> go.Figure:
    """
    Create a risk-return scatter plot for multiple symbols.
    
    Args:
        data_dict: Dictionary of {symbol: DataFrame}
        window: Window for calculating risk and return (days)
        title: Chart title
        
    Returns:
        Plotly figure
    """
    try:
        # Calculate risk and return for each symbol
        risk_return_data = []
        
        for symbol, df in data_dict.items():
            if 'return' in df.columns and len(df) >= window:
                # Use the most recent 'window' days
                recent_data = df.iloc[-window:]
                
                # Calculate average return
                avg_return = recent_data['return'].mean() * 252 * 100  # Annualized, in percent
                
                # Calculate volatility (risk)
                volatility = recent_data['return'].std() * np.sqrt(252) * 100  # Annualized, in percent
                
                risk_return_data.append({
                    'symbol': symbol,
                    'return': avg_return,
                    'risk': volatility
                })
        
        # Create DataFrame
        risk_return_df = pd.DataFrame(risk_return_data)
        
        # Create scatter plot
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=risk_return_df['risk'],
                y=risk_return_df['return'],
                mode='markers+text',
                marker=dict(size=10),
                text=risk_return_df['symbol'],
                textposition="top center"
            )
        )
        
        # Add diagonal reference line
        max_val = max(risk_return_df['risk'].max(), risk_return_df['return'].max())
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name="Risk = Return"
            )
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=600,
            width=800,
            template="plotly_white",
            xaxis_title="Risk (Annualized Volatility %)",
            yaxis_title="Return (Annualized %)",
            showlegend=False
        )
        
        # Add quadrant labels
        midpoint_x = risk_return_df['risk'].median()
        midpoint_y = risk_return_df['return'].median()
        
        fig.add_annotation(
            x=midpoint_x / 2,
            y=midpoint_y + (risk_return_df['return'].max() - midpoint_y) / 2,
            text="Low Risk<br>High Return",
            showarrow=False,
            font=dict(size=10, color="green")
        )
        
        fig.add_annotation(
            x=midpoint_x + (risk_return_df['risk'].max() - midpoint_x) / 2,
            y=midpoint_y + (risk_return_df['return'].max() - midpoint_y) / 2,
            text="High Risk<br>High Return",
            showarrow=False,
            font=dict(size=10, color="orange")
        )
        
        fig.add_annotation(
            x=midpoint_x / 2,
            y=midpoint_y / 2,
            text="Low Risk<br>Low Return",
            showarrow=False,
            font=dict(size=10, color="blue")
        )
        
        fig.add_annotation(
            x=midpoint_x + (risk_return_df['risk'].max() - midpoint_x) / 2,
            y=midpoint_y / 2,
            text="High Risk<br>Low Return",
            showarrow=False,
            font=dict(size=10, color="red")
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating risk-return scatter: {e}")
        # Return simple empty figure
        return go.Figure()

def create_sentiment_timeline(
    news_data: List[Dict[str, Any]], title: str = "News Sentiment Timeline"
) -> go.Figure:
    """
    Create a timeline of news sentiment.
    
    Args:
        news_data: List of news items with sentiment scores
        title: Chart title
        
    Returns:
        Plotly figure
    """
    try:
        # Extract dates and sentiment scores
        dates = []
        sentiments = []
        titles = []
        sources = []
        
        for item in news_data:
            if 'published_at' in item and 'sentiment_score' in item:
                dates.append(item['published_at'])
                sentiments.append(item['sentiment_score'])
                titles.append(item.get('title', 'No title'))
                sources.append(item.get('source', 'Unknown'))
        
        # Convert dates to datetime if they're strings
        if dates and isinstance(dates[0], str):
            dates = [pd.to_datetime(d) for d in dates]
        
        # Sort by date
        data = pd.DataFrame({
            'date': dates,
            'sentiment': sentiments,
            'title': titles,
            'source': sources
        }).sort_values('date')
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot with color based on sentiment
        colors = ['red' if s < -0.2 else 'green' if s > 0.2 else 'gray' for s in data['sentiment']]
        
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['sentiment'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=colors
                ),
                text=[f"Title: {t}<br>Source: {s}<br>Sentiment: {sent:.2f}" 
                      for t, s, sent in zip(data['title'], data['source'], data['sentiment'])],
                hoverinfo='text'
            )
        )
        
        # Add reference lines
        fig.add_hline(y=0.2, line_dash="dash", line_color="green", line_width=1)
        fig.add_hline(y=-0.2, line_dash="dash", line_color="red", line_width=1)
        fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
        
        # Add annotations for reference lines
        fig.add_annotation(
            x=data['date'].min(),
            y=0.2,
            text="Positive",
            showarrow=False,
            xanchor="left",
            font=dict(size=10, color="green")
        )
        
        fig.add_annotation(
            x=data['date'].min(),
            y=-0.2,
            text="Negative",
            showarrow=False,
            xanchor="left",
            font=dict(size=10, color="red")
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=500,
            width=900,
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            yaxis=dict(range=[-1, 1])
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating sentiment timeline: {e}")
        # Return simple empty figure
        return go.Figure()

def create_financial_metrics_radar(
    metrics: Dict[str, float], benchmark: Dict[str, float] = None,
    title: str = "Financial Metrics"
) -> go.Figure:
    """
    Create a radar chart of financial metrics.
    
    Args:
        metrics: Dictionary of {metric_name: value}
        benchmark: Dictionary of benchmark values (optional)
        title: Chart title
        
    Returns:
        Plotly figure
    """
    try:
        # Extract metric names and values
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        # Normalize values to 0-1 scale (better comparison)
        def normalize_values(values, min_threshold=0.01, max_threshold=0.99):
            normalized = []
            for val in values:
                # Normalize based on typical ranges for financial metrics
                if val <= 0:
                    norm_val = min_threshold
                elif val > 2:
                    norm_val = max_threshold
                else:
                    norm_val = (val / 2) * (max_threshold - min_threshold) + min_threshold
                normalized.append(norm_val)
            return normalized
        
        metric_values_norm = normalize_values(metric_values)
        
        # Create figure
        fig = go.Figure()
        
        # Add main metrics
        fig.add_trace(
            go.Scatterpolar(
                r=metric_values_norm,
                theta=metric_names,
                fill='toself',
                name="Company"
            )
        )
        
        # Add benchmark if provided
        if benchmark:
            benchmark_values = [benchmark.get(metric, 0) for metric in metric_names]
            benchmark_values_norm = normalize_values(benchmark_values)
            
            fig.add_trace(
                go.Scatterpolar(
                    r=benchmark_values_norm,
                    theta=metric_names,
                    fill='toself',
                    name="Benchmark"
                )
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            height=500,
            width=700
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating financial metrics radar: {e}")
        # Return simple empty figure
        return go.Figure()

def create_strategy_visualization(
    strategy: Dict[str, Any], data: pd.DataFrame, title: str = "Investment Strategy"
) -> Tuple[go.Figure, str]:
    """
    Create a visualization of the investment strategy.
    
    Args:
        strategy: Strategy recommendations
        data: Stock price data
        title: Chart title
        
    Returns:
        Tuple of (Plotly figure, strategy summary text)
    """
    try:
        # Extract key information from strategy
        overall = strategy.get('overall', {})
        entry_exit = strategy.get('entry_exit', {})
        
        recommendation = overall.get('recommendation', 'HOLD')
        confidence = overall.get('confidence', 0.5)
        reasoning = overall.get('reasoning', 'No detailed reasoning available.')
        time_horizon = overall.get('time_horizon', 'Medium')
        
        # Extract price targets
        price_targets = overall.get('price_targets', {})
        entry_price = price_targets.get('entry', 0)
        exit_price = price_targets.get('exit', 0)
        stop_loss = price_targets.get('stop_loss', 0)
        
        # Ensure price targets are valid
        current_price = data['close'].iloc[-1] if not data.empty else 0
        
        if entry_price == 0:
            entry_price = current_price
        if exit_price == 0:
            exit_price = current_price * 1.1  # Default: 10% above current
        if stop_loss == 0:
            stop_loss = current_price * 0.9  # Default: 10% below current
        
        # Create candlestick chart
        fig = create_candlestick_chart(data, title=title)
        
        # Add horizontal lines for price targets
        fig.add_hline(y=entry_price, line_dash="dash", line_color="blue", 
                     annotation_text="Entry", annotation_position="right")
        fig.add_hline(y=exit_price, line_dash="dash", line_color="green", 
                     annotation_text="Target", annotation_position="right")
        fig.add_hline(y=stop_loss, line_dash="dash", line_color="red", 
                     annotation_text="Stop Loss", annotation_position="right")
        
        # Add annotation for recommendation
        color_map = {"BUY": "green", "HOLD": "blue", "SELL": "red"}
        rec_color = color_map.get(recommendation, "gray")
        
        fig.add_annotation(
            x=data.index[-1],
            y=data['high'].max(),
            text=f"{recommendation} (Confidence: {confidence:.2f})",
            showarrow=True,
            arrowhead=1,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=rec_color,
            font=dict(size=12, color=rec_color),
            align="right"
        )
        
        # Create strategy summary text
        summary = f"""
        <h3>Strategy Summary: {recommendation} - {confidence:.0%} Confidence</h3>
        <p><strong>Time Horizon:</strong> {time_horizon}</p>
        <p><strong>Entry Price:</strong> ${entry_price:.2f}</p>
        <p><strong>Target Price:</strong> ${exit_price:.2f}</p>
        <p><strong>Stop Loss:</strong> ${stop_loss:.2f}</p>
        <p><strong>Potential Return:</strong> {((exit_price / entry_price) - 1) * 100:.1f}%</p>
        <p><strong>Risk/Reward Ratio:</strong> {(exit_price - entry_price) / (entry_price - stop_loss):.2f}</p>
        <hr>
        <p><strong>Reasoning:</strong> {reasoning}</p>
        """
        
        return fig, summary
    
    except Exception as e:
        logger.error(f"Error creating strategy visualization: {e}")
        # Return simple empty figure and error message
        return go.Figure(), f"Error creating strategy visualization: {e}"
