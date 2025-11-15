import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ...config.session_state import get_data_manager, mark_data_saved
from ...components.charts import create_preprocessed_data_chart


def show_preprocessed_data_management():
    """Enhanced interface for managing preprocessed datasets with consistent buttons"""
    st.markdown('<div class="subsection-header">💾 Preprocessed Data Management</div>', unsafe_allow_html=True)

    data_manager = get_data_manager()

    # Show recently saved datasets
    show_recently_saved_datasets()

    # Get available preprocessed datasets
    datasets = data_manager.get_available_preprocessed_datasets()

    if datasets:
        st.markdown(f"#### 📁 Available Preprocessed Datasets ({len(datasets)} total)")

        # Group datasets by exchange and algorithm
        grouped_datasets = {}
        for dataset in datasets:
            key = f"{dataset['exchange']}_{dataset['algorithm']}"
            if key not in grouped_datasets:
                grouped_datasets[key] = []
            grouped_datasets[key].append(dataset)

        for group_key, group_datasets in grouped_datasets.items():
            exchange, algorithm = group_key.split('_', 1)
            with st.expander(f"🏢 {exchange} - 🤖 {algorithm} ({len(group_datasets)} pairs)"):
                for idx, dataset in enumerate(group_datasets):
                    # Use consistent columns with equal button widths
                    col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 1.2, 1.2])
                    with col1:
                        st.write(f"**{dataset['pair']}**")
                    with col2:
                        st.write(f"`{dataset['timeframe']}`")
                    with col3:
                        # Load metadata to show record count
                        _, metadata = data_manager.load_preprocessed_data(
                            dataset['exchange'], dataset['algorithm'], dataset['pair'], dataset['timeframe']
                        )
                        record_count = metadata.get('records_count', 'N/A') if metadata else 'N/A'
                        st.write(f"📊 {record_count} records")
                    with col4:
                        # Consistent View button
                        if st.button("👁️ View",
                                     key=f"view_{group_key}_{idx}",
                                     use_container_width=True,
                                     help="View detailed dataset with charts"):
                            view_dataset(data_manager, dataset)
                    with col5:
                        # Consistent Delete button
                        if st.button("🗑️ Delete",
                                     key=f"delete_{group_key}_{idx}",
                                     use_container_width=True,
                                     help="Delete this dataset"):
                            if data_manager.delete_preprocessed_data(
                                    dataset['exchange'], dataset['algorithm'], dataset['pair'], dataset['timeframe']
                            ):
                                st.success(
                                    f"✅ Deleted {dataset['exchange']} {dataset['algorithm']} {dataset['pair']} {dataset['timeframe']}")
                                st.rerun()
                            else:
                                st.error("❌ Failed to delete dataset")

        # Show the viewed dataset if one was selected
        if 'viewed_dataset' in st.session_state:
            display_enhanced_viewed_dataset()

    else:
        st.info("📭 No preprocessed datasets found. Process some data first!")

    # Enhanced save functionality with automatic detection
    if st.session_state.get('preprocessing_results') and st.session_state.get('current_algorithm'):
        st.markdown("#### 💾 Save Current Preprocessed Data")

        with st.expander("💾 Save Options", expanded=True):
            # Auto-detect exchange and pair from current context
            default_exchange = st.session_state.get('current_exchange', 'binance')
            default_pair = st.session_state.get('current_pair', 'BTC_USDT')
            default_algorithm = st.session_state.current_algorithm

            col1, col2, col3 = st.columns(3)
            with col1:
                exchange = st.text_input("Exchange", value=default_exchange, key="preprocessed_exchange_save")
            with col2:
                pair = st.text_input("Trading Pair", value=default_pair, key="preprocessed_pair_save")
            with col3:
                algorithm = st.text_input("Algorithm", value=default_algorithm, key="preprocessed_algorithm_save")

            # Show processing summary
            total_timeframes = len(st.session_state.preprocessing_results)
            total_records = sum(
                len(result['processed_data']) for result in st.session_state.preprocessing_results.values())

            st.info(f"📊 Ready to save: {total_timeframes} timeframes, {total_records} total records")

            if st.button("💾 Save All Processed Timeframes",
                         type="primary",
                         key="save_preprocessed_btn_enhanced",
                         use_container_width=True,
                         help="Save all processed timeframes to the preprocessed database"):
                with st.spinner(f"Saving {total_timeframes} timeframes..."):
                    save_preprocessed_data(data_manager, exchange, pair, algorithm)

                # Refresh the datasets list
                st.rerun()


def show_recently_saved_datasets():
    """Show recently saved datasets"""
    if st.session_state.get('last_saved_datasets'):
        st.markdown("#### 🕒 Recently Saved Datasets")

        recent_datasets = st.session_state.last_saved_datasets[-5:]  # Show last 5

        for saved_item in reversed(recent_datasets):
            dataset = saved_item['dataset']
            timestamp = saved_item['timestamp'].strftime("%Y-%m-%d %H:%M:%S")

            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            with col1:
                st.write(f"**{dataset['pair']}**")
            with col2:
                st.write(f"`{dataset['timeframe']}`")
            with col3:
                st.write(f"`{dataset['algorithm']}`")
            with col4:
                st.write(f"`{timestamp}`")

        st.markdown("---")


def view_dataset(data_manager, dataset):
    """View a specific dataset"""
    try:
        processed_data, metadata = data_manager.load_preprocessed_data(
            dataset['exchange'], dataset['algorithm'], dataset['pair'], dataset['timeframe']
        )

        if processed_data is not None:
            # Store the viewed dataset in session state
            st.session_state.viewed_dataset = {
                'data': processed_data,
                'metadata': metadata,
                'dataset_info': dataset
            }
            st.success(f"✅ Loaded dataset: {dataset['pair']} {dataset['timeframe']}")
            st.rerun()
        else:
            st.error("❌ Failed to load dataset")

    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")


def display_enhanced_viewed_dataset():
    """Display the currently viewed dataset with charts and detailed information"""
    viewed_data = st.session_state.viewed_dataset
    dataset_info = viewed_data['dataset_info']
    processed_data = viewed_data['data']
    metadata = viewed_data['metadata']

    st.markdown("---")
    st.markdown(f"#### 👁️ Detailed View: {dataset_info['pair']} {dataset_info['timeframe']}")

    # Basic information in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records", len(processed_data))
    with col2:
        st.metric("Algorithm", dataset_info['algorithm'])
    with col3:
        st.metric("Exchange", dataset_info['exchange'])
    with col4:
        st.metric("Timeframe", dataset_info['timeframe'])

    # Dataset Chart
    st.markdown("##### 📈 Dataset Chart")

    # Get available indicators from the dataset
    available_indicators = [col for col in processed_data.columns
                            if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trend_label']]

    # Categorize indicators
    indicator_categories = {
        'Moving Averages': [ind for ind in available_indicators if any(ma in ind for ma in ['SMA', 'EMA', 'WMA'])],
        'Momentum': [ind for ind in available_indicators if
                     any(mom in ind for mom in ['RSI', 'MACD', 'Stoch', 'Williams', 'CCI', 'ROC', 'MFI', 'TSI'])],
        'Volatility': [ind for ind in available_indicators if
                       any(vol in ind for vol in ['BB', 'ATR', 'KC', 'STD', 'VAR', 'CHV'])],
        'Volume': [ind for ind in available_indicators if
                   any(vol in ind for vol in ['Volume_SMA', 'OBV', 'ADL', 'CMF'])]
    }

    # Create chart with available indicators
    if all(col in processed_data.columns for col in ['timestamp', 'open', 'high', 'low', 'close']):
        # Use all available indicators for the chart
        selected_indicators = []
        for category_indicators in indicator_categories.values():
            selected_indicators.extend(category_indicators)

        fig = create_preprocessed_data_chart(
            processed_data,
            selected_indicators,
            dataset_info['timeframe'],
            dataset_info['algorithm']
        )

        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not generate chart for this dataset")
    else:
        st.warning("Insufficient data columns for chart generation")

    # Detailed Dataset Information
    st.markdown("##### 📊 Dataset Details")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Basic Info", "🧮 Indicators", "📋 Data Quality", "🔍 Sample Data"])

    with tab1:
        display_basic_dataset_info(processed_data, metadata, dataset_info)

    with tab2:
        display_indicators_info(processed_data, indicator_categories)

    with tab3:
        display_data_quality_info(processed_data)

    with tab4:
        display_sample_data(processed_data)

    # Clear button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Close Detailed View", key="close_enhanced_view", use_container_width=True):
            del st.session_state.viewed_dataset
            st.rerun()


def display_basic_dataset_info(processed_data, metadata, dataset_info):
    """Display basic dataset information"""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Date Range:**")
        if 'timestamp' in processed_data.columns:
            start_date = processed_data['timestamp'].min()
            end_date = processed_data['timestamp'].max()
            st.write(f"Start: `{start_date}`")
            st.write(f"End: `{end_date}`")
            st.write(f"Duration: `{(end_date - start_date).days} days`")
        else:
            st.write("No timestamp column")

    with col2:
        st.markdown("**Data Statistics:**")
        st.write(f"Total Columns: `{len(processed_data.columns)}`")
        st.write(f"Memory Usage: `{processed_data.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB`")

        if metadata:
            st.write(f"Processed: `{metadata.get('processed_timestamp', 'Unknown')}`")


def display_indicators_info(processed_data, indicator_categories):
    """Display information about calculated indicators"""
    st.markdown("#### 🧮 Calculated Technical Indicators")

    for category, indicators in indicator_categories.items():
        if indicators:
            with st.expander(f"{category} ({len(indicators)} indicators)"):
                # Display indicators in a grid
                cols = st.columns(3)
                for i, indicator in enumerate(indicators):
                    with cols[i % 3]:
                        if indicator in processed_data.columns:
                            # Show basic stats for the indicator
                            indicator_data = processed_data[indicator].dropna()
                            if len(indicator_data) > 0:
                                st.metric(
                                    label=indicator,
                                    value=f"{len(indicator_data)} vals",
                                    delta=f"μ: {indicator_data.mean():.2f}"
                                )
                            else:
                                st.write(f"`{indicator}`: No data")
                        else:
                            st.write(f"`{indicator}`: Missing")

    # Show indicator summary
    total_indicators = sum(len(indicators) for indicators in indicator_categories.values())
    if total_indicators > 0:
        st.success(f"✅ Total calculated indicators: {total_indicators}")
    else:
        st.info("No technical indicators calculated in this dataset")


def display_data_quality_info(processed_data):
    """Display data quality metrics"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        null_count = processed_data.isnull().sum().sum()
        st.metric("Null Values", null_count)

    with col2:
        completeness = (1 - null_count / (len(processed_data) * len(processed_data.columns))) * 100
        st.metric("Completeness", f"{completeness:.1f}%")

    with col3:
        duplicate_count = processed_data.duplicated().sum()
        st.metric("Duplicates", duplicate_count)

    with col4:
        numeric_cols = processed_data.select_dtypes(include=['number']).columns
        st.metric("Numeric Columns", len(numeric_cols))

    # Column-specific null information
    st.markdown("**Column-wise Null Counts:**")
    null_counts = processed_data.isnull().sum()
    null_counts = null_counts[null_counts > 0]

    if len(null_counts) > 0:
        for col, count in null_counts.items():
            st.write(f"- `{col}`: {count} null values ({count / len(processed_data) * 100:.1f}%)")
    else:
        st.success("✅ No null values in any column!")

    # Trend classification info if available
    if 'trend_label' in processed_data.columns:
        st.markdown("**Trend Classification:**")
        label_counts = processed_data['trend_label'].value_counts().sort_index()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Downward Trends", label_counts.get(-1, 0))
        with col2:
            st.metric("Neutral Trends", label_counts.get(0, 0))
        with col3:
            st.metric("Upward Trends", label_counts.get(1, 0))


def display_sample_data(processed_data):
    """Display sample of the dataset"""
    st.markdown("#### 🔍 Data Sample")

    # Configuration for display
    display_config = st.expander("Display Configuration", expanded=False)
    with display_config:
        max_rows = st.slider("Max rows to display", 10, 1000, 100, 10, key="sample_display_slider")
        show_all_columns = st.checkbox("Show all columns", value=False, key="sample_show_all_columns")

    # Prepare data for display
    if len(processed_data) > max_rows:
        st.warning(f"Showing first {max_rows} rows of {len(processed_data)} total rows")
        display_data = processed_data.head(max_rows)
    else:
        display_data = processed_data

    if not show_all_columns and len(display_data.columns) > 12:
        # Show only key columns by default
        key_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        available_key_cols = [col for col in key_columns if col in display_data.columns]
        other_cols = [col for col in display_data.columns if col not in key_columns]

        if available_key_cols:
            st.info(f"Showing key columns. Enable 'Show all columns' to see {len(other_cols)} additional columns.")
            display_data = display_data[available_key_cols]

    st.dataframe(display_data, use_container_width=True, height=400)


def save_preprocessed_data(data_manager, exchange, pair, algorithm):
    """Save preprocessed data with enhanced error handling and feedback"""
    saved_count = 0
    total_timeframes = len(st.session_state.preprocessing_results)
    save_results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (timeframe, result) in enumerate(st.session_state.preprocessing_results.items()):
        try:
            status_text.text(f"Saving {timeframe}...")

            success = data_manager.save_preprocessed_data(
                exchange=exchange,
                pair=pair,
                algorithm=algorithm,
                timeframe=timeframe,
                processed_data=result['processed_data'],
                scaler=result.get('scaler'),
                validation_results=result.get('validation_results', {})
            )

            if success:
                saved_count += 1
                dataset_info = {
                    'exchange': exchange,
                    'pair': pair,
                    'algorithm': algorithm,
                    'timeframe': timeframe,
                    'records': len(result['processed_data'])
                }
                mark_data_saved(dataset_info)
                save_results.append(f"✅ {timeframe} ({len(result['processed_data'])} records)")
            else:
                save_results.append(f"❌ {timeframe} failed")

            progress_bar.progress((i + 1) / total_timeframes)

        except Exception as e:
            save_results.append(f"❌ {timeframe} error: {str(e)}")
            progress_bar.progress((i + 1) / total_timeframes)

    progress_bar.empty()
    status_text.empty()

    # Show detailed results
    st.markdown("##### 📋 Save Results:")
    for result in save_results:
        st.write(result)

    if saved_count > 0:
        st.balloons()
        st.success(f"🎉 Successfully saved {saved_count}/{total_timeframes} timeframes!")

        # Clear processing results after successful save
        st.session_state.preprocessing_results = {}

        # Refresh the page to show updated datasets
        st.rerun()
    else:
        st.error("❌ No timeframes were saved successfully")