import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ...config.performance import performance_monitor
from ...config.session_state import get_data_manager

@performance_monitor
def show_historical_data_tab():
    """Historical Data tab with performance optimizations"""
    st.markdown('<div class="subsection-header">Historical Data Analysis</div>', unsafe_allow_html=True)

    data_manager = get_data_manager()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)

        available_exchanges = data_manager.get_available_exchanges()
        selected_exchange = st.selectbox(
            "🏢 Select Exchange",
            available_exchanges,
            key="historical_exchange"
        )

        if selected_exchange:
            available_pairs = data_manager.get_trading_pairs(selected_exchange)

            if available_pairs:
                selected_pair = st.selectbox(
                    "💰 Select Trading Pair",
                    available_pairs,
                    key="historical_pair"
                )

                if selected_pair:
                    timeframes = data_manager.get_available_timeframes(selected_exchange, selected_pair)

                    if timeframes:
                        selected_timeframe = st.selectbox(
                            "⏱️ Select Timeframe",
                            timeframes,
                            key="historical_timeframe"
                        )
                    else:
                        st.warning("No timeframes available for this pair")
                        selected_timeframe = None
            else:
                st.warning("No trading pairs available for this exchange")
                selected_pair = None

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if selected_exchange and selected_pair and selected_timeframe:
            with st.spinner("Loading data..."):
                df = data_manager.load_pair_data(selected_exchange, selected_pair, selected_timeframe)

            if df is not None and not df.empty:
                st.markdown('<div class="data-card">', unsafe_allow_html=True)
                st.success(f"✅ Loaded {len(df)} records")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Records", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    null_count = df.isnull().sum().sum()
                    st.metric("Null Values", null_count)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="data-card">', unsafe_allow_html=True)
                st.markdown("#### Dataset Preview")
                st.dataframe(df, use_container_width=True, height=400)
                st.markdown('</div>', unsafe_allow_html=True)

                if all(col in df.columns for col in ['close', 'timestamp']):
                    st.markdown('<div class="data-card">', unsafe_allow_html=True)
                    st.markdown("#### Price Chart")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], name='Close Price'))
                    fig.update_layout(
                        title=f"{selected_pair} - {selected_timeframe}",
                        template="plotly_dark",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("No data found for the selected parameters")