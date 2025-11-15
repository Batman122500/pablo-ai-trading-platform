import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import threading
import time
from ...config.performance import performance_monitor
from ...config.session_state import get_data_manager, get_algo_preprocessor, get_preprocessor, mark_data_saved
from ...components.charts import show_chart_visualization
from ...components.task_monitor import task_monitor


@performance_monitor
def show_data_preprocessing_tab():
    """Enhanced Data Preprocessing tab with timeframe selection and technical indicators"""
    st.markdown('<div class="subsection-header">🧹 Data Cleaning & Preprocessing</div>', unsafe_allow_html=True)

    data_manager = get_data_manager()
    algo_preprocessor = get_algo_preprocessor()
    preprocessor = get_preprocessor()

    # Algorithm Selection
    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    st.markdown("#### 🎯 Select Learning Algorithm")

    algorithms = list(algo_preprocessor.algorithm_configs.keys())
    selected_algorithm = st.selectbox(
        "Choose Algorithm",
        algorithms,
        key="preprocessing_algo_select"
    )

    if selected_algorithm:
        config = algo_preprocessor.get_algorithm_config(selected_algorithm)
        st.info(f"**Configuration:** {config['description']}")
        st.info(f"**Type:** {config.get('type', 'regression').upper()}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Data Selection
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.markdown("#### 🏢 Select Data Source")

        available_exchanges = data_manager.get_available_exchanges()
        selected_exchange = st.selectbox(
            "Exchange",
            available_exchanges,
            key="preprocessing_exchange_select"
        )

        if selected_exchange:
            available_pairs = data_manager.get_trading_pairs(selected_exchange)
            if available_pairs:
                selected_pair = st.selectbox(
                    "Trading Pair",
                    available_pairs,
                    key="preprocessing_pair_select"
                )
                # Store current exchange and pair for automatic detection
                st.session_state.current_exchange = selected_exchange
                st.session_state.current_pair = selected_pair
            else:
                st.warning("No pairs available")
                selected_pair = None
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.markdown("#### ⚙️ Processing Configuration")

        # Enhanced technical indicators selection with checkbox
        st.markdown("##### 📊 Technical Indicators")

        all_indicators = ['SMA', 'EMA', 'WMA', 'RSI', 'MACD', 'Stochastic', 'Williams %R', 'CCI', 'ROC', 'MFI', 'TSI',
                          'Bollinger_Bands', 'ATR', 'Keltner_Channel', 'STD', 'VAR', 'CHV', 'Volume_SMA', 'OBV', 'ADL',
                          'CMF']

        # Checkbox for select all
        select_all = st.checkbox("Select All Technical Indicators", value=False, key="select_all_indicators_checkbox")

        if select_all:
            selected_indicators = all_indicators
            # Show disabled multiselect with all indicators selected
            st.multiselect(
                "Select indicators to calculate:",
                all_indicators,
                default=all_indicators,
                disabled=True,
                key="preprocessing_indicators_select_disabled"
            )
        else:
            selected_indicators = st.multiselect(
                "Select indicators to calculate:",
                all_indicators,
                default=['SMA', 'RSI', 'MACD', 'Bollinger_Bands', 'ATR'],
                key="preprocessing_indicators_select"
            )

        # Classification-specific settings
        if selected_algorithm == 'Classification':
            st.markdown("##### 🏷️ Classification Settings")
            col_lookforward, col_threshold = st.columns(2)
            with col_lookforward:
                lookforward_period = st.number_input("Lookforward Period", min_value=1, max_value=20, value=5,
                                                     key="lookforward_period_input")
            with col_threshold:
                classification_threshold = st.number_input("Trend Threshold (%)", min_value=0.1, max_value=10.0,
                                                           value=2.0, step=0.1,
                                                           key="classification_threshold_input") / 100

        # Data splitting configuration
        st.markdown("##### 📊 Data Splitting")
        col_train, col_val, col_test = st.columns(3)
        with col_train:
            train_ratio = st.number_input("Train %", min_value=0.0, max_value=1.0, value=0.7, step=0.05,
                                          key="train_ratio_input")
        with col_val:
            val_ratio = st.number_input("Validation %", min_value=0.0, max_value=1.0, value=0.15, step=0.05,
                                        key="val_ratio_input")
        with col_test:
            test_ratio = st.number_input("Test %", min_value=0.0, max_value=1.0, value=0.15, step=0.05,
                                         key="test_ratio_input")

        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            st.warning(f"Ratios sum to {total_ratio:.2f}. Please adjust to sum to 1.0")

        st.markdown('</div>', unsafe_allow_html=True)

    # Timeframe Selection Section
    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    st.markdown("##### ⏱️ Timeframe Selection")

    if selected_exchange and selected_pair:
        timeframes = data_manager.get_available_timeframes(selected_exchange, selected_pair)
        if timeframes:
            # Checkbox for select all timeframes
            select_all_timeframes = st.checkbox(
                "Select All Timeframes",
                value=False,
                key="select_all_timeframes_checkbox"
            )

            if select_all_timeframes:
                selected_timeframes = timeframes
                # Show disabled multiselect with all timeframes selected
                st.multiselect(
                    "Select timeframes to process:",
                    timeframes,
                    default=timeframes,
                    disabled=True,
                    key="preprocessing_timeframes_select_disabled"
                )
                st.info(f"✅ All {len(timeframes)} timeframes selected")
            else:
                selected_timeframes = st.multiselect(
                    "Select timeframes to process:",
                    timeframes,
                    default=timeframes[:3] if len(timeframes) > 3 else timeframes,
                    key="preprocessing_timeframes_select"
                )
        else:
            st.warning("No timeframes available for selected pair")
            selected_timeframes = []
    else:
        st.info("👆 Select exchange and trading pair first")
        selected_timeframes = []
    st.markdown('</div>', unsafe_allow_html=True)

    # Process Data Button with validation and threading
    process_disabled = not (selected_exchange and selected_pair and selected_timeframes)

    if st.button("🚀 Process Selected Timeframes",
                 use_container_width=True,
                 type="primary" if not process_disabled else "secondary",
                 disabled=process_disabled,
                 key="process_timeframes_btn"):

        if selected_exchange and selected_pair and selected_timeframes:
            st.session_state.current_algorithm = selected_algorithm
            st.session_state.current_exchange = selected_exchange
            st.session_state.current_pair = selected_pair

            # Start processing in a thread to prevent UI blocking
            thread = threading.Thread(
                target=process_selected_timeframes_thread,
                args=(
                    data_manager, algo_preprocessor, preprocessor,
                    selected_exchange, selected_pair, selected_algorithm,
                    selected_indicators, train_ratio, val_ratio, test_ratio,
                    selected_timeframes,
                    lookforward_period if selected_algorithm == 'Classification' else 5,
                    classification_threshold if selected_algorithm == 'Classification' else 0.02
                )
            )
            thread.daemon = True
            thread.start()

            st.success(f"🔄 Started processing {len(selected_timeframes)} timeframes in background...")
            st.info("Check the task monitor above for progress updates.")
        else:
            st.error("Please select exchange, trading pair, and timeframes")

    # Show chart visualization if we have results
    if st.session_state.preprocessing_results:
        show_chart_visualization()

    # Show preprocessed data management
    from ...pages.data_management.preprocessed_data import show_preprocessed_data_management
    show_preprocessed_data_management()


def process_selected_timeframes_thread(data_manager, algo_preprocessor, preprocessor, exchange, pair, algorithm,
                                       indicators,
                                       train_ratio, val_ratio, test_ratio, timeframes, lookforward_period=5,
                                       classification_threshold=0.02):
    """Thread function for processing timeframes"""
    process_selected_timeframes(data_manager, algo_preprocessor, preprocessor, exchange, pair, algorithm, indicators,
                                train_ratio, val_ratio, test_ratio, timeframes, lookforward_period,
                                classification_threshold)


def save_processed_data_automatically(data_manager, exchange, pair, algorithm, results):
    """Automatically save processed data after successful preprocessing"""
    try:
        saved_count = 0
        total_timeframes = len(results)
        save_details = []

        for timeframe, result in results.items():
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
                save_details.append(f"✅ {timeframe} ({len(result['processed_data'])} records)")
            else:
                save_details.append(f"❌ Failed to save {timeframe}")

        return saved_count, total_timeframes, save_details

    except Exception as e:
        print(f"❌ Error in auto-save: {e}")
        return 0, 0, [f"❌ Auto-save error: {str(e)}"]


def process_selected_timeframes(data_manager, algo_preprocessor, preprocessor, exchange, pair, algorithm, indicators,
                                train_ratio, val_ratio, test_ratio, timeframes, lookforward_period=5,
                                classification_threshold=0.02):
    """Process selected timeframes for the selected pair with auto-save"""
    try:
        if not timeframes:
            st.error("No timeframes selected for processing")
            return

        task_id = f"preprocess_{exchange}_{pair}_{int(time.time())}"
        task_monitor.start_task(task_id, f"Processing {len(timeframes)} timeframes for {pair}")

        results = {}

        for i, timeframe in enumerate(timeframes):
            progress = int((i / len(timeframes)) * 100)
            task_monitor.update_task(task_id, progress, f"Processing {timeframe}...")

            try:
                df = data_manager.load_pair_data(exchange, pair, timeframe)

                if df is not None and not df.empty:
                    # Enhanced cleaning that preserves data
                    clean_df = preprocessor.clean_data(df)

                    # Algorithm-specific cleaning
                    algo_clean_df = algo_preprocessor.clean_data_algorithm_specific(clean_df, algorithm)

                    # Calculate indicators
                    processed_df = algo_preprocessor.calculate_technical_indicators(algo_clean_df, indicators)

                    if algorithm == 'Classification':
                        processed_df = algo_preprocessor.add_classification_labels(
                            processed_df, lookforward_period, classification_threshold
                        )

                    normalized_df, scaler = algo_preprocessor.normalize_data(processed_df, algorithm)
                    validation_results = algo_preprocessor.validate_processed_data(normalized_df, algorithm)

                    # Only split if we have enough data
                    if len(normalized_df) > 100:
                        train_data, val_data, test_data = preprocessor.split_data(
                            normalized_df, train_ratio, val_ratio, test_ratio
                        )
                    else:
                        train_data, val_data, test_data = normalized_df, None, None
                        st.warning(f"⚠️ Insufficient data for proper splitting in {timeframe}")

                    results[timeframe] = {
                        'original_records': len(df),
                        'processed_records': len(normalized_df),
                        'train_records': len(train_data) if train_data is not None else 0,
                        'val_records': len(val_data) if val_data is not None else 0,
                        'test_records': len(test_data) if test_data is not None else 0,
                        'validation_results': validation_results,
                        'processed_data': normalized_df,
                        'scaler': scaler
                    }

                    st.success(f"✅ {timeframe}: {len(normalized_df)} records processed")

                else:
                    st.warning(f"⚠️ No data found for {timeframe}")

            except Exception as e:
                st.error(f"❌ Error processing {timeframe}: {str(e)}")

        # AUTO-SAVE AFTER SUCCESSFUL PROCESSING
        if results:
            task_monitor.update_task(task_id, 95, "Auto-saving processed data...")
            saved_count, total_timeframes, save_details = save_processed_data_automatically(
                data_manager, exchange, pair, algorithm, results
            )

            # Store results in session state for chart visualization
            st.session_state.preprocessing_results = results

            if saved_count > 0:
                task_monitor.complete_task(task_id,
                                           f"Processed and auto-saved {saved_count}/{total_timeframes} timeframes")

                # Show save details
                st.markdown("#### 💾 Auto-Save Results:")
                for detail in save_details:
                    st.write(detail)

                st.success(f"🎉 Successfully processed and auto-saved {saved_count}/{total_timeframes} timeframes!")
                st.balloons()
            else:
                task_monitor.fail_task(task_id, "Processing completed but auto-saving failed")
                st.error("❌ Processing completed but failed to auto-save data")
                # Still store results for manual saving
                st.session_state.preprocessing_results = results
        else:
            task_monitor.fail_task(task_id, "No data was processed")
            st.error("❌ No data was processed")

        display_preprocessing_summary(results, exchange, pair, algorithm)

    except Exception as e:
        task_monitor.fail_task(task_id, f"Processing failed: {str(e)}")
        st.error(f"Error during batch processing: {str(e)}")


def display_preprocessing_summary(results, exchange, pair, algorithm):
    """Display comprehensive preprocessing summary"""
    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    st.markdown("#### 📊 Processing Summary")

    if not results:
        st.warning("No data was processed")
        return

    total_original = sum(result['original_records'] for result in results.values())
    total_processed = sum(result['processed_records'] for result in results.values())
    total_train = sum(result['train_records'] for result in results.values())
    total_val = sum(result['val_records'] for result in results.values())
    total_test = sum(result['test_records'] for result in results.values())

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Original Records", f"{total_original:,}")
    with col2:
        st.metric("Total Processed Records", f"{total_processed:,}")
    with col3:
        st.metric("Training Samples", f"{total_train:,}")
    with col4:
        st.metric("Validation/Test Samples", f"{total_val + total_test:,}")

    st.markdown("#### 📈 Detailed Results by Timeframe")

    summary_data = []
    for timeframe, result in results.items():
        summary_data.append({
            'Timeframe': timeframe,
            'Original': result['original_records'],
            'Processed': result['processed_records'],
            'Train': result['train_records'],
            'Validation': result['val_records'],
            'Test': result['test_records'],
            'Status': '✅' if result['validation_results']['has_required_columns'] else '⚠️'
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

    st.markdown("#### ✅ Data Quality Validation")

    for timeframe, result in results.items():
        with st.expander(f"Validation Results - {timeframe}"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Records", result['processed_records'])
                st.metric("Null Values", result['validation_results']['null_count'])
            with col2:
                st.metric("Completeness", f"{result['validation_results']['completeness']:.1f}%")
                status = "✅ PASS" if result['validation_results']['has_required_columns'] else "❌ FAIL"
                st.metric("Required Columns", status)

            if algorithm == 'Classification' and 'trend_label' in result['processed_data'].columns:
                st.markdown("##### 🏷️ Classification Labels Distribution")
                label_counts = result['processed_data']['trend_label'].value_counts()
                fig = go.Figure(data=[go.Pie(labels=['Downward', 'Neutral', 'Upward'],
                                             values=[label_counts.get(-1, 0), label_counts.get(0, 0),
                                                     label_counts.get(1, 0)])])
                fig.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)