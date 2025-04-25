"""
Reusable UI widgets for the Streamlit application.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Union, Callable

def metric_card(
    title: str, 
    value: Union[str, float, int], 
    delta: Optional[Union[str, float, int]] = None,
    delta_color: str = "normal",
    help_text: Optional[str] = None
) -> None:
    """
    Display a metric card with title, value, and optional delta.
    
    Args:
        title: Card title
        value: Main value to display
        delta: Optional delta value
        delta_color: Color of delta (normal, inverse, off)
        help_text: Optional tooltip text
    """
    st.metric(
        label=title,
        value=value,
        delta=delta,
        delta_color=delta_color,
        help=help_text
    )

def info_card(title: str, content: str, icon: str = "ℹ️") -> None:
    """
    Display an information card with icon.
    
    Args:
        title: Card title
        content: Card content
        icon: Emoji icon
    """
    st.markdown(
        f"""
        <div style="padding: 0.5rem; border-radius: 0.5rem; background-color: #f0f2f6; margin-bottom: 1rem;">
            <h3 style="margin: 0 0 0.5rem 0;">{icon} {title}</h3>
            <p style="margin: 0;">{content}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def data_table(
    data: pd.DataFrame, 
    height: Optional[int] = None, 
    column_config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Display a data table with customizable columns.
    
    Args:
        data: DataFrame to display
        height: Optional height constraint
        column_config: Optional column configuration
    """
    # Default column formatting
    if column_config is None:
        column_config = {}
    
    # Apply default formatting based on column types
    for col in data.columns:
        if col not in column_config:
            if pd.api.types.is_numeric_dtype(data[col]):
                if data[col].abs().max() > 100000:
                    # Large numbers
                    column_config[col] = st.column_config.NumberColumn(
                        format="%.2f"
                    )
                else:
                    # Regular numbers
                    column_config[col] = st.column_config.NumberColumn(
                        format="%.2f"
                    )
            elif pd.api.types.is_datetime64_dtype(data[col]):
                # Date columns
                column_config[col] = st.column_config.DatetimeColumn(
                    format="YYYY-MM-DD"
                )
    
    # Display the table
    st.dataframe(
        data,
        height=height,
        column_config=column_config,
        use_container_width=True
    )

def expandable_section(
    title: str, 
    content_func: Callable[[], None], 
    expanded: bool = False
) -> None:
    """
    Create an expandable section with custom content.
    
    Args:
        title: Section title
        content_func: Function that populates the section content
        expanded: Whether the section is expanded by default
    """
    with st.expander(title, expanded=expanded):
        content_func()

def tabbed_sections(
    titles: List[str], 
    content_funcs: List[Callable[[], None]]
) -> None:
    """
    Create tabbed sections with custom content.
    
    Args:
        titles: List of tab titles
        content_funcs: List of functions that populate each tab
    """
    if len(titles) != len(content_funcs):
        raise ValueError("Number of titles must match number of content functions")
    
    tabs = st.tabs(titles)
    
    for i, tab in enumerate(tabs):
        with tab:
            content_funcs[i]()

def alert(message: str, type: str = "info") -> None:
    """
    Display an alert message with specified type.
    
    Args:
        message: Alert message
        type: Alert type (info, success, warning, error)
    """
    if type == "info":
        st.info(message)
    elif type == "success":
        st.success(message)
    elif type == "warning":
        st.warning(message)
    elif type == "error":
        st.error(message)
    else:
        st.write(message)

def progress_indicator(
    label: str, 
    value: float, 
    min_value: float = 0.0, 
    max_value: float = 1.0,
    format_func: Optional[Callable[[float], str]] = None
) -> None:
    """
    Display a progress indicator.
    
    Args:
        label: Progress label
        value: Current value
        min_value: Minimum value
        max_value: Maximum value
        format_func: Optional function to format the displayed value
    """
    # Normalize value to 0-1 range for progress bar
    normalized = (value - min_value) / (max_value - min_value)
    normalized = max(0, min(1, normalized))  # Clamp to 0-1
    
    # Format display value
    if format_func:
        display_value = format_func(value)
    else:
        display_value = f"{value:.2f}"
    
    # Display progress
    st.write(f"{label}: {display_value}")
    st.progress(normalized)

def gauge_chart(
    value: float, 
    title: str, 
    min_value: float = 0, 
    max_value: float = 100,
    threshold_ranges: Optional[List[Tuple[float, float, str]]] = None
) -> go.Figure:
    """
    Create a gauge chart with customizable thresholds.
    
    Args:
        value: Gauge value
        title: Chart title
        min_value: Minimum value
        max_value: Maximum value
        threshold_ranges: List of (min, max, color) tuples for thresholds
        
    Returns:
        Plotly figure
    """
    # Default threshold ranges if not provided
    if threshold_ranges is None:
        # Default: red (0-33), yellow (33-66), green (66-100)
        threshold_ranges = [
            (min_value, min_value + (max_value - min_value) / 3, "red"),
            (min_value + (max_value - min_value) / 3, min_value + 2 * (max_value - min_value) / 3, "yellow"),
            (min_value + 2 * (max_value - min_value) / 3, max_value, "green")
        ]
    
    # Determine gauge color based on value
    gauge_color = threshold_ranges[0][2]  # Default to first range color
    for range_min, range_max, color in threshold_ranges:
        if range_min <= value <= range_max:
            gauge_color = color
            break
    
    # Calculate gauge value as percentage
    gauge_value = (value - min_value) / (max_value - min_value) * 100
    
    # Create the gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': gauge_color},
            'steps': [
                {'range': [(r[0] - min_value) / (max_value - min_value) * 100, 
                          (r[1] - min_value) / (max_value - min_value) * 100], 
                 'color': f"rgba({','.join(['255' if c == r[2] else '200' for c in ['red', 'yellow', 'green']])},0.2)"} 
                for r in threshold_ranges
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': gauge_value
            }
        }
    ))
    
    # Update layout
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig

def sentiment_gauge(
    sentiment_score: float, 
    title: str = "Sentiment Score"
) -> go.Figure:
    """
    Create a sentiment gauge chart (-1 to 1 scale).
    
    Args:
        sentiment_score: Sentiment score (-1 to 1)
        title: Chart title
        
    Returns:
        Plotly figure
    """
    # Define threshold ranges for sentiment
    ranges = [
        (-1.0, -0.2, "red"),
        (-0.2, 0.2, "yellow"),
        (0.2, 1.0, "green")
    ]
    
    return gauge_chart(
        value=sentiment_score,
        title=title,
        min_value=-1.0,
        max_value=1.0,
        threshold_ranges=ranges
    )

def risk_gauge(
    risk_score: float, 
    title: str = "Risk Score"
) -> go.Figure:
    """
    Create a risk gauge chart (0 to 10 scale).
    
    Args:
        risk_score: Risk score (0 to 10)
        title: Chart title
        
    Returns:
        Plotly figure
    """
    # Define threshold ranges for risk (inverse colors)
    ranges = [
        (0, 3.33, "green"),
        (3.33, 6.67, "yellow"),
        (6.67, 10, "red")
    ]
    
    return gauge_chart(
        value=risk_score,
        title=title,
        min_value=0,
        max_value=10,
        threshold_ranges=ranges
    )

def chat_message(
    content: str, 
    is_user: bool = False, 
    avatar: Optional[str] = None
) -> None:
    """
    Display a chat message with optional avatar.
    
    Args:
        content: Message content
        is_user: Whether the message is from the user
        avatar: Optional avatar emoji
    """
    if is_user:
        message_type = "user"
        avatar = avatar or "👤"
    else:
        message_type = "assistant"
        avatar = avatar or "🤖"
    
    st.chat_message(message_type, avatar=avatar).markdown(content)

def create_chat_interface(
    submit_func: Callable[[str], str],
    placeholder: str = "Type your message here...",
    submit_button_text: str = "Send",
    clear_on_submit: bool = True
) -> None:
    """
    Create a chat interface with input and submit button.
    
    Args:
        submit_func: Function to process the submitted message
        placeholder: Placeholder text for input
        submit_button_text: Text for submit button
        clear_on_submit: Whether to clear input after submission
    """
    # Initialize chat history in session state if it doesn't exist
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Display chat history
    for message in st.session_state.chat_messages:
        chat_message(
            content=message["content"],
            is_user=message["is_user"],
            avatar=message.get("avatar", None)
        )
    
    # Chat input
    user_input = st.text_input(
        "Message",
        key="chat_input",
        placeholder=placeholder
    )
    
    # Submit button
    if st.button(submit_button_text, key="chat_submit"):
        if user_input:
            # Add user message to chat history
            st.session_state.chat_messages.append({
                "content": user_input,
                "is_user": True
            })
            
            # Process input and get response
            response = submit_func(user_input)
            
            # Add assistant response to chat history
            st.session_state.chat_messages.append({
                "content": response,
                "is_user": False
            })
            
            # Clear input if requested
            if clear_on_submit:
                st.session_state.chat_input = ""
            
            # Force rerun to update chat display
            st.rerun()
