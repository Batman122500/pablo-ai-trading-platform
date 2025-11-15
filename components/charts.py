import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st


def create_preprocessed_data_chart(df, selected_indicators, timeframe, algorithm):
    """Enhanced chart visualization with fixed indicators"""
    if df is None or df.empty:
        st.warning("No data available for chart visualization")
        return None

    # Ensure we have required columns
    required_cols = ['timestamp', 'open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        st.error("Missing required columns for chart")
        return None

    # Get pair from session state for the title
    pair = st.session_state.get('current_pair', 'Unknown Pair')

    # Create subplots based on available indicators
    subplot_titles = ['Price & Moving Averages']
    rows = 1
    row_heights = [0.4]

    # Check which indicator groups we have data for
    has_momentum = any(ind in df.columns for ind in ['RSI_14', 'RSI_21', 'MACD', 'MACD_signal', 'MACD_hist',
                                                    'Stoch_%K', 'Stoch_%D', 'Williams_%R', 'CCI_20', 'ROC_10',
                                                    'ROC_21', 'MFI_14', 'TSI'])
    has_volatility = any(ind in df.columns for ind in ['BB_upper', 'BB_lower', 'BB_middle', 'BB_width',
                                                      'ATR_14', 'KC_upper', 'KC_lower', 'STD_20', 'VAR_20', 'CHV'])
    has_volume = 'volume' in df.columns

    if has_momentum:
        subplot_titles.append('Momentum Indicators')
        rows += 1
        row_heights.append(0.2)
    if has_volatility:
        subplot_titles.append('Volatility Indicators')
        rows += 1
        row_heights.append(0.2)
    if has_volume:
        subplot_titles.append('Volume Indicators')
        rows += 1
        row_heights.append(0.2)

    fig = make_subplots(
        rows=rows, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.05,
        row_heights=row_heights
    )

    current_row = 1

    # Price and Moving Averages with Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC'
        ), row=current_row, col=1
    )

    # Add selected moving averages
    ma_indicators = {
        'SMA_20': ('SMA 20', 'orange'),
        'SMA_50': ('SMA 50', 'red'),
        'SMA_200': ('SMA 200', 'purple'),
        'EMA_20': ('EMA 20', 'blue'),
        'EMA_50': ('EMA 50', 'green')
    }

    for col, (name, color) in ma_indicators.items():
        if col in df.columns and any(ind in selected_indicators for ind in ['SMA', 'EMA', 'WMA']):
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df[col], name=name,
                           line=dict(color=color, width=1)),
                row=current_row, col=1
            )

    # Bollinger Bands
    if all(col in df.columns for col in ['BB_upper', 'BB_lower', 'BB_middle']) and 'Bollinger_Bands' in selected_indicators:
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['BB_upper'], name='BB Upper',
                       line=dict(color='gray', width=1, dash='dash'),
                       showlegend=True),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['BB_lower'], name='BB Lower',
                       line=dict(color='gray', width=1, dash='dash'),
                       showlegend=True),
            row=current_row, col=1
        )
        # Fill between Bollinger Bands
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['BB_upper'], fill=None,
                       mode='lines', line=dict(color='gray', width=0),
                       showlegend=False),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['BB_lower'], fill='tonexty',
                       mode='lines', line=dict(color='gray', width=0),
                       showlegend=False, fillcolor='rgba(128,128,128,0.2)'),
            row=current_row, col=1
        )

    current_row += 1

    # Momentum Indicators
    if has_momentum:
        # RSI
        if 'RSI_14' in df.columns and 'RSI' in selected_indicators:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['RSI_14'], name='RSI 14',
                           line=dict(color='purple', width=1)),
                row=current_row, col=1
            )
            # Add RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red",
                          annotation_text="Overbought", row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green",
                          annotation_text="Oversold", row=current_row, col=1)

        # MACD
        if all(col in df.columns for col in ['MACD', 'MACD_signal']) and 'MACD' in selected_indicators:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['MACD'], name='MACD',
                           line=dict(color='blue', width=1)),
                row=current_row, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['MACD_signal'], name='MACD Signal',
                           line=dict(color='red', width=1)),
                row=current_row, col=1
            )
            # MACD Histogram
            if 'MACD_hist' in df.columns:
                colors = ['green' if x >= 0 else 'red' for x in df['MACD_hist']]
                fig.add_trace(
                    go.Bar(x=df['timestamp'], y=df['MACD_hist'], name='MACD Hist',
                           marker_color=colors, opacity=0.6),
                    row=current_row, col=1
                )

        # Stochastic
        if all(col in df.columns for col in ['Stoch_%K', 'Stoch_%D']) and 'Stochastic' in selected_indicators:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['Stoch_%K'], name='Stoch %K',
                           line=dict(color='cyan', width=1)),
                row=current_row, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['Stoch_%D'], name='Stoch %D',
                           line=dict(color='magenta', width=1, dash='dash')),
                row=current_row, col=1
            )
            fig.add_hline(y=80, line_dash="dash", line_color="red", row=current_row, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="green", row=current_row, col=1)

        current_row += 1

    # Volatility Indicators
    if has_volatility:
        # ATR
        if 'ATR_14' in df.columns and 'ATR' in selected_indicators:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['ATR_14'], name='ATR',
                           line=dict(color='orange', width=1)),
                row=current_row, col=1
            )

        # Keltner Channel
        if all(col in df.columns for col in ['KC_upper', 'KC_lower']) and 'Keltner_Channel' in selected_indicators:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['KC_upper'], name='KC Upper',
                           line=dict(color='cyan', width=1, dash='dot')),
                row=current_row, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['KC_lower'], name='KC Lower',
                           line=dict(color='cyan', width=1, dash='dot')),
                row=current_row, col=1
            )

        # Standard Deviation
        if 'STD_20' in df.columns and 'STD' in selected_indicators:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['STD_20'], name='STD 20',
                           line=dict(color='yellow', width=1)),
                row=current_row, col=1
            )

        current_row += 1

    # Volume Indicators with fixed colors
    if has_volume:
        # Calculate colors based on price movement - FIXED
        colors = ['green' if close > open else 'red' for close, open in zip(df['close'], df['open'])]

        fig.add_trace(
            go.Bar(x=df['timestamp'], y=df['volume'], name='Volume',
                   marker_color=colors, opacity=0.7),
            row=current_row, col=1
        )

        # Volume SMA
        if 'Volume_SMA_20' in df.columns and 'Volume_SMA' in selected_indicators:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['Volume_SMA_20'], name='Volume SMA 20',
                           line=dict(color='yellow', width=2)),
                row=current_row, col=1
            )

        # OBV
        if 'OBV' in df.columns and 'OBV' in selected_indicators:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['OBV'], name='OBV',
                           line=dict(color='white', width=1)),
                row=current_row, col=1
            )

        current_row += 1

    # Update layout
    fig.update_layout(
        title=f"Preprocessed Data - {pair} {timeframe} - {algorithm}",
        template="plotly_dark",
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )

    # Update y-axis titles
    fig.update_yaxes(title_text="Price", row=1, col=1)
    if has_momentum:
        fig.update_yaxes(title_text="Momentum", row=2, col=1)
    if has_volatility:
        vol_row = 3 if has_momentum else 2
        fig.update_yaxes(title_text="Volatility", row=vol_row, col=1)
    if has_volume:
        vol_row = 4 if has_momentum and has_volatility else (3 if has_momentum or has_volatility else 2)
        fig.update_yaxes(title_text="Volume", row=vol_row, col=1)

    return fig


def show_chart_visualization():
    """Show chart visualization for preprocessed data"""
    st.markdown('<div class="subsection-header">📊 Preprocessed Data Visualization</div>', unsafe_allow_html=True)

    if not st.session_state.preprocessing_results:
        st.info("👆 Process data first to visualize charts")
        return

    st.markdown('<div class="data-card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        available_timeframes = list(st.session_state.preprocessing_results.keys())
        selected_timeframe = st.selectbox(
            "Select Timeframe for Chart",
            available_timeframes,
            key="chart_timeframe_selector"
        )
        st.session_state.selected_timeframe_for_chart = selected_timeframe

    with col2:
        if selected_timeframe:
            result = st.session_state.preprocessing_results[selected_timeframe]
            st.metric("Processed Records", result['processed_records'])
            st.metric("Algorithm", st.session_state.current_algorithm)

    if selected_timeframe:
        result = st.session_state.preprocessing_results[selected_timeframe]
        processed_data = result.get('processed_data')

        if processed_data is not None and not processed_data.empty:
            # Get available indicators from the actual dataframe columns
            available_indicators = [col for col in processed_data.columns
                                    if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trend_label']]

            # Categorize indicators based on actual column names in the dataframe
            indicator_categories = {
                'Moving Averages': [ind for ind in available_indicators if
                                    any(ma in ind for ma in ['SMA', 'EMA', 'WMA'])],
                'Momentum': [ind for ind in available_indicators if
                             any(mom in ind for mom in ['RSI', 'MACD', 'Stoch', 'Williams', 'CCI', 'ROC', 'MFI', 'TSI'])],
                'Volatility': [ind for ind in available_indicators if
                               any(vol in ind for vol in ['BB', 'ATR', 'KC', 'STD', 'VAR', 'CHV'])],
                'Volume': [ind for ind in available_indicators if
                           any(vol in ind for vol in ['Volume_SMA', 'OBV', 'ADL', 'CMF'])]
            }

            st.markdown("#### 📈 Select Indicators to Visualize")

            selected_indicators = []
            for category, indicators in indicator_categories.items():
                if indicators:
                    with st.expander(f"{category} ({len(indicators)} indicators)"):
                        # Show which indicators are actually available
                        st.caption(f"Available: {', '.join(indicators)}")
                        category_selected = st.multiselect(
                            f"Select {category} indicators:",
                            indicators,
                            default=indicators[:min(3, len(indicators))],  # Show first 3 by default
                            key=f"indicators_{category}_{selected_timeframe}"
                        )
                        selected_indicators.extend(category_selected)

            # Always include basic moving averages if available
            base_indicators = []
            if any('SMA' in ind for ind in available_indicators):
                base_indicators.append('SMA')
            if any('EMA' in ind for ind in available_indicators):
                base_indicators.append('EMA')

            if selected_indicators or base_indicators:
                fig = create_preprocessed_data_chart(
                    processed_data,
                    selected_indicators + base_indicators,
                    selected_timeframe,
                    st.session_state.current_algorithm
                )

                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                    # Display statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Data Points", len(processed_data))
                    with col2:
                        st.metric("Start Date", processed_data['timestamp'].min().strftime('%Y-%m-%d'))
                    with col3:
                        st.metric("End Date", processed_data['timestamp'].max().strftime('%Y-%m-%d'))

                    # Show data quality metrics
                    st.markdown("#### 📊 Data Quality")
                    qual_col1, qual_col2, qual_col3, qual_col4 = st.columns(4)
                    with qual_col1:
                        null_count = processed_data.isnull().sum().sum()
                        st.metric("Null Values", null_count)
                    with qual_col2:
                        completeness = (1 - null_count / (len(processed_data) * len(processed_data.columns))) * 100
                        st.metric("Completeness", f"{completeness:.1f}%")
                    with qual_col3:
                        duplicate_count = processed_data.duplicated().sum()
                        st.metric("Duplicates", duplicate_count)
                    with qual_col4:
                        memory_usage = processed_data.memory_usage(deep=True).sum() / 1024 ** 2
                        st.metric("Memory (MB)", f"{memory_usage:.2f}")

                    # Trend classification if available
                    if 'trend_label' in processed_data.columns:
                        st.markdown("#### 🏷️ Trend Classification Distribution")
                        label_counts = processed_data['trend_label'].value_counts().sort_index()
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Downward Trends", label_counts.get(-1, 0))
                        with col2:
                            st.metric("Neutral Trends", label_counts.get(0, 0))
                        with col3:
                            st.metric("Upward Trends", label_counts.get(1, 0))
            else:
                st.info("No technical indicators available for visualization. Try selecting different indicators during preprocessing.")
        else:
            st.warning("No processed data available for the selected timeframe")

    st.markdown('</div>', unsafe_allow_html=True)