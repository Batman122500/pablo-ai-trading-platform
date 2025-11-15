import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ..config.performance import performance_monitor
from ..components.header import show_connection_status

@performance_monitor
def show_dashboard():
    """Main dashboard view"""
    st.markdown('<div class="section-header">📊 Trading Dashboard</div>', unsafe_allow_html=True)
    is_online = show_connection_status()

    # Enhanced Metrics with gradient cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Balance", "$1000000000000000000", "+2.3%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Active Bots", "3", "0")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("24h Profit", "$234.56", "+1.2%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Open Positions", "5", "-1")
        st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced Charts
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.subheader("Portfolio Value")
        chart_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'value': [10000 + i * 100 + (i % 3) * 50 for i in range(30)]
        })
        st.line_chart(chart_data.set_index('date'))
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.subheader("Asset Distribution")
        fig = go.Figure(data=[go.Pie(labels=['BTC', 'ETH', 'ADA', 'DOT'], values=[40, 30, 20, 10])])
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)