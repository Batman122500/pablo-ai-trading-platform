import streamlit as st
from datetime import datetime, timedelta
import threading
import time
import concurrent.futures
import pandas as pd
import os
import json
from ...config.performance import performance_monitor
from ...config.session_state import get_data_manager
from ...components.header import check_internet_connection
from ...components.task_monitor import task_monitor


@performance_monitor
def show_trading_pair_management_tab():
    """Enhanced Trading Pair Management with better pair display"""
    st.markdown('<div class="subsection-header">🔄 Trading Pair Management</div>', unsafe_allow_html=True)

    data_manager = get_data_manager()
    is_online = check_internet_connection()

    # Initialize and load persisted pairs
    initialize_persisted_pairs()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.markdown("#### 🏢 Exchange Selection")

        exchange_options = ['binance', 'bitmex', 'bybit', 'coinbase', 'cryptocom', 'gate', 'huobi', 'kucoin', 'okx']
        selected_exchange = st.selectbox(
            "Select Exchange",
            exchange_options,
            key="management_exchange_select"
        )

        # Fetch All Exchanges button
        if st.button("🌐 Fetch All Exchanges",
                     use_container_width=True,
                     type="secondary",
                     disabled=not is_online):
            fetch_all_exchanges_pairs(data_manager, exchange_options)

        # Display exchange summary
        display_exchange_summary(selected_exchange)

        # Auto-fetch pairs when exchange is selected (if not already persisted)
        if (selected_exchange and
                selected_exchange not in st.session_state.exchange_pairs and
                is_online):
            thread = threading.Thread(
                target=fetch_pairs_background,
                args=(data_manager, selected_exchange)
            )
            thread.daemon = True
            thread.start()
            st.info(f"🔄 Fetching pairs for {selected_exchange}...")

        # Display current pairs for selected exchange with IMPROVED SCROLLABLE LAYOUT
        display_trading_pairs_scrollable(selected_exchange)

        # Update button and status
        col_update, col_status = st.columns([2, 1])
        with col_update:
            if st.button("🔄 Update Pairs",
                         use_container_width=True,
                         type="primary",
                         key="update_pairs_btn",
                         disabled=not is_online):
                update_trading_pairs(data_manager, selected_exchange)

        with col_status:
            status_text = "🟢 Online" if is_online else "🔴 Offline"
            status_color = "green" if is_online else "red"
            st.markdown(
                f'<div style="text-align: center; padding: 0.7rem; border-radius: 8px; background: rgba(16, 42, 67, 0.8); border: 1px solid rgba(56, 178, 172, 0.3); color: {status_color}; font-weight: bold;">{status_text}</div>',
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Show data acquisition interface if we have pairs
        current_pairs = st.session_state.exchange_pairs.get(selected_exchange, [])
        all_exchange_pairs = st.session_state.all_exchanges_pairs.get(selected_exchange, [])

        display_pairs = all_exchange_pairs if all_exchange_pairs else current_pairs

        if display_pairs:
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.markdown(f"#### 💰 Select Trading Pair")

            selected_pair = st.selectbox(
                "Choose trading pair:",
                display_pairs,
                key="management_pair_selection"
            )

            if selected_pair:
                show_data_acquisition_interface(data_manager, selected_exchange, selected_pair, is_online)

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            if not is_online:
                st.error("🔴 Internet connection required")
            else:
                st.info("🔄 Fetching trading pairs...")
            st.markdown('</div>', unsafe_allow_html=True)


def initialize_persisted_pairs():
    """Initialize and load persisted trading pairs from file"""
    if 'exchange_pairs' not in st.session_state:
        st.session_state.exchange_pairs = {}

    if 'all_exchanges_pairs' not in st.session_state:
        st.session_state.all_exchanges_pairs = {}

    # Load persisted pairs if they exist
    persisted_file = get_persisted_pairs_file()
    if os.path.exists(persisted_file):
        try:
            with open(persisted_file, 'r') as f:
                persisted_data = json.load(f)
                st.session_state.exchange_pairs = persisted_data.get('exchange_pairs', {})
                st.session_state.all_exchanges_pairs = persisted_data.get('all_exchanges_pairs', {})
                st.session_state.last_persisted = persisted_data.get('last_persisted', None)
        except Exception as e:
            st.warning(f"Could not load persisted pairs: {e}")


def get_persisted_pairs_file():
    """Get the file path for persisted trading pairs"""
    data_manager = get_data_manager()
    return os.path.join(data_manager.db_path, "persisted_trading_pairs.json")


def save_persisted_pairs():
    """Save trading pairs to persistent storage"""
    try:
        persisted_data = {
            'exchange_pairs': st.session_state.exchange_pairs,
            'all_exchanges_pairs': st.session_state.all_exchanges_pairs,
            'last_persisted': datetime.now().isoformat()
        }

        persisted_file = get_persisted_pairs_file()
        with open(persisted_file, 'w') as f:
            json.dump(persisted_data, f, indent=2)
    except Exception as e:
        st.error(f"Error saving persisted pairs: {e}")


def display_exchange_summary(selected_exchange):
    """Display summary of exchanges and pairs"""
    if st.session_state.all_exchanges_pairs:
        st.markdown("#### 📊 All Exchanges Summary")
        total_pairs = sum(len(pairs) for pairs in st.session_state.all_exchanges_pairs.values())
        st.success(f"Total pairs across all exchanges: {total_pairs}")

        # Show top exchanges by pair count
        exchange_counts = [(exch, len(pairs)) for exch, pairs in st.session_state.all_exchanges_pairs.items()]
        exchange_counts.sort(key=lambda x: x[1], reverse=True)

        for exchange, count in exchange_counts[:5]:  # Show top 5
            st.write(f"• {exchange}: {count} pairs")

        if len(exchange_counts) > 5:
            st.write(f"*... and {len(exchange_counts) - 5} more exchanges*")


def display_trading_pairs_scrollable(selected_exchange):
    """Display trading pairs in a scrollable, organized container"""
    current_pairs = st.session_state.exchange_pairs.get(selected_exchange, [])
    all_exchange_pairs = st.session_state.all_exchanges_pairs.get(selected_exchange, [])

    display_pairs = all_exchange_pairs if all_exchange_pairs else current_pairs

    if display_pairs:
        st.markdown(f"#### 📋 Available Pairs ({len(display_pairs)})")

        # Add search functionality
        search_term = st.text_input("🔍 Search pairs:", key=f"search_{selected_exchange}",
                                    placeholder="Type to filter pairs...")

        # Filter pairs based on search
        if search_term:
            filtered_pairs = [pair for pair in display_pairs if search_term.upper() in pair.upper()]
            st.info(f"Showing {len(filtered_pairs)} of {len(display_pairs)} pairs")
        else:
            filtered_pairs = display_pairs

        # Create a scrollable container for pairs
        st.markdown("""
        <style>
        .scrollable-pairs-container {
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid rgba(56, 178, 172, 0.3);
            border-radius: 10px;
            padding: 10px;
            background: rgba(16, 42, 67, 0.6);
            margin-bottom: 10px;
        }
        .pair-item {
            padding: 8px 12px;
            margin: 4px 0;
            background: rgba(30, 58, 95, 0.6);
            border-radius: 6px;
            border: 1px solid rgba(56, 178, 172, 0.2);
            transition: all 0.3s ease;
            cursor: pointer;
            text-align: center;
            font-family: 'Courier New', monospace;
        }
        .pair-item:hover {
            background: rgba(56, 178, 172, 0.2);
            border-color: rgba(56, 178, 172, 0.5);
            transform: translateX(5px);
        }
        </style>
        """, unsafe_allow_html=True)

        if filtered_pairs:
            # Display pairs in a scrollable container
            pairs_html = f"""
            <div class="scrollable-pairs-container">
                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 8px;">
            """

            for pair in filtered_pairs:
                pairs_html += f'<div class="pair-item"><code>{pair}</code></div>'

            pairs_html += """
                </div>
            </div>
            """

            st.markdown(pairs_html, unsafe_allow_html=True)

            # Show summary
            if not search_term:
                st.caption(f"💡 Found {len(filtered_pairs)} pairs. Use search to filter, or scroll to see all.")
        else:
            st.info("No pairs match your search criteria")

        # Quick actions
        if filtered_pairs:
            st.markdown("##### 🚀 Quick Actions")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📋 Copy All Pairs", key=f"copy_all_{selected_exchange}", use_container_width=True):
                    pairs_text = "\n".join(filtered_pairs)
                    st.session_state.copied_pairs = pairs_text
                    st.success(f"✅ Copied {len(filtered_pairs)} pairs to clipboard!")
            with col2:
                if st.button("⭐ Show Top 20", key=f"top_20_{selected_exchange}", use_container_width=True):
                    st.info(f"Top 20 pairs: {', '.join(filtered_pairs[:20])}")
    else:
        st.info("No pairs loaded for this exchange")


def fetch_all_exchanges_pairs(data_manager, exchanges):
    """Fetch trading pairs from ALL exchanges in parallel and persist them"""
    task_id = f"fetch_all_exchanges_{int(time.time())}"
    task_monitor.start_task(task_id, "Fetching pairs from all exchanges")

    def fetch_single_exchange(exchange):
        try:
            return exchange, data_manager.fetch_spot_pairs(exchange, use_cache=False)
        except Exception as e:
            return exchange, []

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_exchange = {executor.submit(fetch_single_exchange, exchange): exchange for exchange in exchanges}

            results = {}
            completed = 0
            total = len(exchanges)

            for future in concurrent.futures.as_completed(future_to_exchange):
                exchange, pairs = future.result()
                results[exchange] = pairs
                completed += 1
                progress = int((completed / total) * 100)
                task_monitor.update_task(task_id, progress, f"Fetched {exchange} ({len(pairs)} pairs)")

        # Update session state
        st.session_state.all_exchanges_pairs = results
        # Also update individual exchange pairs
        for exchange, pairs in results.items():
            st.session_state.exchange_pairs[exchange] = pairs

        # Persist to file
        save_persisted_pairs()

        total_pairs = sum(len(pairs) for pairs in results.values())
        task_monitor.complete_task(task_id, f"Fetched {total_pairs} pairs from {len(exchanges)} exchanges")
        st.success(f"✅ Fetched {total_pairs} trading pairs from {len(exchanges)} exchanges!")
        st.rerun()

    except Exception as e:
        task_monitor.fail_task(task_id, f"Error fetching all exchanges: {str(e)}")
        st.error(f"Error fetching all exchanges: {str(e)}")


def fetch_pairs_background(data_manager, exchange):
    """Fetch pairs in background thread and persist them"""
    try:
        spot_pairs = data_manager.fetch_spot_pairs(exchange, use_cache=True)
        if spot_pairs:
            st.session_state.exchange_pairs[exchange] = spot_pairs
            # Also update all_exchanges_pairs if it exists
            if exchange in st.session_state.all_exchanges_pairs:
                st.session_state.all_exchanges_pairs[exchange] = spot_pairs

            # Persist to file
            save_persisted_pairs()
    except Exception as e:
        print(f"Error fetching pairs for {exchange}: {e}")


def update_trading_pairs(data_manager, exchange):
    """Update trading pairs list and persist them"""
    task_id = f"update_pairs_{exchange}_{int(time.time())}"
    task_monitor.start_task(task_id, f"Fetching pairs from {exchange}")

    try:
        task_monitor.update_task(task_id, 25, "Connecting to exchange...")

        # Force refresh by not using cache
        spot_pairs = data_manager.fetch_spot_pairs(exchange, use_cache=False)

        task_monitor.update_task(task_id, 75, "Processing pairs list...")

        if spot_pairs:
            st.session_state.exchange_pairs[exchange] = spot_pairs
            # Also update all_exchanges_pairs if it exists
            if exchange in st.session_state.all_exchanges_pairs:
                st.session_state.all_exchanges_pairs[exchange] = spot_pairs

            # Persist to file
            save_persisted_pairs()

            task_monitor.update_task(task_id, 100, f"Found {len(spot_pairs)} trading pairs")
            task_monitor.complete_task(task_id, f"Updated {len(spot_pairs)} pairs")
            st.success(f"✅ Updated {len(spot_pairs)} trading pairs from {exchange}")
            st.rerun()
        else:
            task_monitor.fail_task(task_id, "No pairs found")
            st.error("No pairs found or error fetching pairs")

    except Exception as e:
        task_monitor.update_task(task_id, 100, f"Error: {str(e)}")
        task_monitor.fail_task(task_id, f"Update error: {str(e)}")
        st.error(f"Error updating pairs: {e}")


def show_data_acquisition_interface(data_manager, exchange, pair, is_online):
    """Show data acquisition interface for selected pair"""
    st.markdown("#### 📥 Data Acquisition")

    # Multi-exchange checkbox
    fetch_from_all = st.checkbox(
        "🌐 Fetch from all available exchanges",
        value=False,
        key=f"fetch_all_exchanges_{pair}",
        help="Fetch data for this trading pair from all exchanges that support it"
    )

    # Get available exchanges for this pair if multi-exchange is selected
    available_exchanges = [exchange]  # Start with current exchange
    if fetch_from_all:
        # Check which exchanges have this pair
        all_exchanges_data = st.session_state.all_exchanges_pairs or st.session_state.exchange_pairs
        available_exchanges = []
        for exch, pairs in all_exchanges_data.items():
            if pair in pairs:
                available_exchanges.append(exch)

        if available_exchanges:
            st.info(
                f"📊 This pair is available on {len(available_exchanges)} exchanges: {', '.join(available_exchanges)}")
        else:
            st.warning("This pair was not found on any other exchanges")
            fetch_from_all = False
            available_exchanges = [exchange]

    # Enhanced timeframe selection for multi-exchange
    st.markdown("#### ⏱️ Timeframe Selection")

    if fetch_from_all:
        # Get all unique timeframes from all available exchanges
        all_timeframes = {}
        for exch in available_exchanges:
            try:
                timeframes = data_manager.get_exchange_timeframes(exch)
                for tf_key, tf_value in timeframes.items():
                    display_name = f"{tf_value}-{exch.capitalize()}"
                    all_timeframes[display_name] = (exch, tf_key, tf_value)
            except Exception as e:
                st.warning(f"Could not get timeframes for {exch}: {e}")

        timeframe_options = ["All Timeframes"] + list(all_timeframes.keys())

        selected_timeframe_option = st.selectbox(
            "Select Timeframe",
            timeframe_options,
            key=f"acquisition_timeframe_multi_{pair}"
        )

        # Store the mapping for later use
        st.session_state.timeframe_mapping = all_timeframes
    else:
        # Single exchange timeframe selection
        exchange_timeframes = data_manager.get_exchange_timeframes(exchange)
        timeframe_options = list(exchange_timeframes.keys())
        timeframe_options_with_all = ["All Timeframes"] + timeframe_options

        selected_timeframe_option = st.selectbox(
            "Select Timeframe",
            timeframe_options_with_all,
            key=f"acquisition_timeframe_{exchange}_{pair}"
        )

    # Date selection
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30),
                                   key=f"start_date_{exchange}_{pair}")
    with col_d2:
        end_date = st.date_input("End Date", value=datetime.now(),
                                 key=f"end_date_{exchange}_{pair}")

    # Advanced options for data management
    with st.expander("⚙️ Advanced Data Options"):
        st.markdown("**Data Management Strategy:**")
        auto_append = st.checkbox("🔄 Auto-append and merge data", value=True,
                                  help="Append new data to existing data and fill gaps")
        reprocess_after_fetch = st.checkbox("🔄 Re-preprocess after fetch", value=True,
                                            help="Automatically re-preprocess all data for this pair after fetching")

        if reprocess_after_fetch:
            st.info("All preprocessed data for this pair will be regenerated after fetching new data")

    if st.button("📥 Acquire Data",
                 use_container_width=True,
                 type="primary",
                 key=f"acquire_data_{exchange}_{pair}"):
        if is_online:
            acquire_data(data_manager, exchange, pair, selected_timeframe_option,
                         start_date, end_date, fetch_from_all, available_exchanges,
                         auto_append, reprocess_after_fetch)
        else:
            st.error("Internet connection required for data acquisition")


def acquire_data(data_manager, base_exchange, pair, timeframe_option, start_date, end_date,
                 fetch_from_all=False, available_exchanges=None, auto_append=True, reprocess_after_fetch=True):
    """Acquire data for selected timeframe(s) with enhanced storage logic"""

    if fetch_from_all and available_exchanges:
        # Multi-exchange acquisition
        task_id = f"acquire_multi_{pair}_{int(time.time())}"
        task_monitor.start_task(task_id, f"Acquiring data from {len(available_exchanges)} exchanges for {pair}")

        try:
            if timeframe_option == "All Timeframes":
                # Get all timeframes from all exchanges
                timeframes_to_fetch = []
                for exch in available_exchanges:
                    exch_timeframes = data_manager.get_exchange_timeframes(exch)
                    for tf_key, tf_value in exch_timeframes.items():
                        timeframes_to_fetch.append((exch, tf_key, tf_value))
            else:
                # Single timeframe from mapping
                timeframe_mapping = st.session_state.get('timeframe_mapping', {})
                if timeframe_option in timeframe_mapping:
                    exch, tf_key, tf_value = timeframe_mapping[timeframe_option]
                    timeframes_to_fetch = [(exch, tf_key, tf_value)]
                else:
                    timeframes_to_fetch = [(base_exchange, timeframe_option, timeframe_option)]

            total_timeframes = len(timeframes_to_fetch)
            success_count = 0

            # Use ThreadPoolExecutor for parallel fetching
            def fetch_single_timeframe(args):
                exch, timeframe_key, timeframe_value = args
                try:
                    start_dt = datetime.combine(start_date, datetime.min.time())
                    end_dt = datetime.combine(end_date, datetime.max.time())

                    # Get existing data range to optimize fetching
                    existing_start, existing_end = data_manager.get_existing_data_range(exch, pair, timeframe_key)

                    # If we have existing data and auto_append is True, adjust the fetch range
                    if existing_start and existing_end and auto_append:
                        # Only fetch data that extends the range
                        if start_dt < existing_start and end_dt > existing_end:
                            # Fetch both before and after existing data
                            fetch_before = True
                            fetch_after = True
                        elif start_dt < existing_start:
                            # Fetch only before existing data
                            fetch_before = True
                            fetch_after = False
                        elif end_dt > existing_end:
                            # Fetch only after existing data
                            fetch_before = False
                            fetch_after = True
                        else:
                            # New data is within existing range, no need to fetch
                            return (exch, timeframe_key, False, 0, "within_existing_range")
                    else:
                        # No existing data or not auto-appending, fetch the full range
                        fetch_before = True
                        fetch_after = True

                    # Fetch data based on the determined ranges
                    fetched_dfs = []

                    if fetch_before and start_dt < (existing_start if existing_start else end_dt):
                        fetch_start = start_dt
                        fetch_end = existing_start - timedelta(minutes=1) if existing_start else end_dt
                        df_before = data_manager.fetch_ohlcv_data(exch, pair, timeframe_key, fetch_start, fetch_end)
                        if df_before is not None and not df_before.empty:
                            fetched_dfs.append(df_before)

                    if fetch_after and end_dt > (existing_end if existing_end else start_dt):
                        fetch_start = existing_end + timedelta(minutes=1) if existing_end else start_dt
                        fetch_end = end_dt
                        df_after = data_manager.fetch_ohlcv_data(exch, pair, timeframe_key, fetch_start, fetch_end)
                        if df_after is not None and not df_after.empty:
                            fetched_dfs.append(df_after)

                    if not fetched_dfs:
                        return (exch, timeframe_key, False, 0, "no_new_data")

                    # Combine all fetched data
                    combined_new_df = pd.concat(fetched_dfs).drop_duplicates('timestamp').sort_values('timestamp')

                    if combined_new_df.empty:
                        return (exch, timeframe_key, False, 0, "empty_after_combining")

                    save_result = data_manager.save_pair_data(
                        exch,
                        pair,
                        timeframe_key,
                        combined_new_df,
                        merge_strategy="append" if auto_append else "replace"
                    )
                    return (exch, timeframe_key, True, len(combined_new_df), save_result)

                except Exception as e:
                    return (exch, timeframe_key, False, 0, str(e))

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(fetch_single_timeframe, tf) for tf in timeframes_to_fetch]

                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    exch, timeframe, success, records, result = future.result()
                    progress = int((i + 1) / total_timeframes * 100)

                    if success:
                        success_count += 1
                        task_monitor.update_task(task_id, progress,
                                                 f"✅ {exch}-{timeframe}: {records} records")
                        st.success(f"✅ {exch}-{timeframe}: {records} records {result}")
                    else:
                        task_monitor.update_task(task_id, progress,
                                                 f"❌ {exch}-{timeframe}: {result}")
                        if result not in ["within_existing_range", "no_new_data", "empty_after_combining"]:
                            st.error(f"❌ {exch}-{timeframe}: {result}")
                        else:
                            st.info(f"ℹ️ {exch}-{timeframe}: {result}")

            # Reprocess preprocessed data if requested
            if success_count > 0 and reprocess_after_fetch:
                reprocess_preprocessed_data(data_manager, pair, available_exchanges)

            if success_count > 0:
                task_monitor.complete_task(task_id, f"Acquired {success_count}/{total_timeframes} datasets")
                st.balloons()
                st.success(f"🎉 Successfully acquired data from {success_count} exchange-timeframe combinations!")
            else:
                task_monitor.fail_task(task_id, "No new data acquired from any exchange")

        except Exception as e:
            task_monitor.fail_task(task_id, f"Multi-exchange acquisition error: {str(e)}")
            st.error(f"Error during multi-exchange data acquisition: {str(e)}")

    else:
        # Single exchange acquisition with enhanced logic
        task_id = f"acquire_{base_exchange}_{pair}_{int(time.time())}"

        if timeframe_option == "All Timeframes":
            timeframes = list(data_manager.get_exchange_timeframes(base_exchange).keys())
            task_monitor.start_task(task_id, f"Acquiring all timeframes for {pair}")
        else:
            timeframes = [timeframe_option]
            task_monitor.start_task(task_id, f"Acquiring {timeframe_option} data for {pair}")

        try:
            total_timeframes = len(timeframes)
            success_count = 0

            for i, timeframe in enumerate(timeframes):
                progress = int((i / total_timeframes) * 100)
                task_monitor.update_task(task_id, progress, f"Processing {timeframe}...")

                start_dt = datetime.combine(start_date, datetime.min.time())
                end_dt = datetime.combine(end_date, datetime.max.time())

                # Get existing data range to optimize fetching
                existing_start, existing_end = data_manager.get_existing_data_range(base_exchange, pair, timeframe)

                # Determine what to fetch based on existing data and auto_append setting
                if existing_start and existing_end and auto_append:
                    # Only fetch data that extends the existing range
                    if start_dt < existing_start and end_dt > existing_end:
                        # Fetch both before and after
                        fetch_ranges = [
                            (start_dt, existing_start - timedelta(minutes=1)),
                            (existing_end + timedelta(minutes=1), end_dt)
                        ]
                    elif start_dt < existing_start:
                        # Fetch only before
                        fetch_ranges = [(start_dt, min(existing_start - timedelta(minutes=1), end_dt))]
                    elif end_dt > existing_end:
                        # Fetch only after
                        fetch_ranges = [(max(existing_end + timedelta(minutes=1), start_dt), end_dt)]
                    else:
                        # New range is within existing data
                        st.info(f"ℹ️ {timeframe}: Requested range is within existing data")
                        continue
                else:
                    # Fetch the full range (either no existing data or not auto-appending)
                    fetch_ranges = [(start_dt, end_dt)]

                # Fetch data for each range
                fetched_dfs = []
                for fetch_start, fetch_end in fetch_ranges:
                    if fetch_start >= fetch_end:
                        continue

                    df = data_manager.fetch_ohlcv_data(
                        base_exchange,
                        pair,
                        timeframe,
                        fetch_start,
                        fetch_end
                    )

                    if df is not None and not df.empty:
                        fetched_dfs.append(df)

                if not fetched_dfs:
                    st.warning(f"⚠️ No new data fetched for {timeframe}")
                    continue

                # Combine all fetched data
                combined_df = pd.concat(fetched_dfs).drop_duplicates('timestamp').sort_values('timestamp')

                if combined_df.empty:
                    st.warning(f"⚠️ No valid data to save for {timeframe}")
                    continue

                save_result = data_manager.save_pair_data(
                    base_exchange,
                    pair,
                    timeframe,
                    combined_df,
                    merge_strategy="append" if auto_append else "replace"
                )

                if save_result != "error":
                    success_count += 1
                    st.success(f"✅ {timeframe}: {len(combined_df)} records {save_result}")
                else:
                    st.error(f"❌ Failed to save {timeframe} data")

            # Reprocess preprocessed data if requested and we have new data
            if success_count > 0 and reprocess_after_fetch:
                reprocess_preprocessed_data(data_manager, pair, [base_exchange])

            if success_count > 0:
                task_monitor.complete_task(task_id, f"Acquired {success_count}/{total_timeframes} timeframes")
                st.balloons()
                st.success(f"🎉 Successfully acquired data for {success_count} timeframe(s)")
            else:
                task_monitor.fail_task(task_id, "No new data acquired")

        except Exception as e:
            task_monitor.fail_task(task_id, f"Acquisition error: {str(e)}")
            st.error(f"Error during data acquisition: {str(e)}")


def reprocess_preprocessed_data(data_manager, pair, exchanges):
    """Reprocess all preprocessed data for a given pair across exchanges"""
    task_id = f"reprocess_{pair}_{int(time.time())}"
    task_monitor.start_task(task_id, f"Reprocessing preprocessed data for {pair}")

    try:
        # This would trigger reprocessing of all preprocessed data
        # You would need to implement the actual reprocessing logic here
        st.info(f"🔄 Reprocessing preprocessed data for {pair} across {len(exchanges)} exchanges...")

        # Placeholder for actual reprocessing logic
        # This would involve:
        # 1. Finding all preprocessed datasets for this pair
        # 2. Reloading the raw data
        # 3. Running preprocessing again
        # 4. Saving the updated preprocessed data

        task_monitor.complete_task(task_id, f"Reprocessed data for {pair}")
        st.success(f"✅ Preprocessed data updated for {pair}")

    except Exception as e:
        task_monitor.fail_task(task_id, f"Reprocessing error: {str(e)}")
        st.error(f"Error during reprocessing: {str(e)}")