import streamlit as st
from ..config.performance import performance_monitor
from ..components.header import show_connection_status

@performance_monitor
def show_trading_monitor():
    """Trading monitor page"""
    st.markdown('<div class="section-header">📈 Trading Monitor</div>', unsafe_allow_html=True)
    is_online = show_connection_status()

    if not is_online:
        st.error("🔴 Trading monitor requires internet connection for real-time data")
        return

    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    st.markdown("#### 🎯 Active Trading Bots")

    # Placeholder for trading bots display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Bots", "3", "0")
    with col2:
        st.metric("Total Positions", "12", "+2")
    with col3:
        st.metric("24h P&L", "$1,234.56", "+5.2%")

    st.markdown("#### 📊 Performance Overview")
    st.info("Real-time trading performance charts will be displayed here")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    st.markdown("#### ⚡ Real-time Positions")
    st.info("Active positions and order book will be displayed here")
    st.markdown('</div>', unsafe_allow_html=True)