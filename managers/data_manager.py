import os
import json
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta
import streamlit as st
import logging
from ..config.performance import cache_data
from ..components.header import check_internet_connection


class ExchangeDataManager:
    """Enhanced data manager with improved data merging and gap detection"""

    def __init__(self):
        self.db_path = r"C:\Users\DELL\Desktop\PABLO\PABLO\pablo\Memory\CryptoDatabase"
        self.preprocessed_db_path = r"C:\Users\DELL\Desktop\PABLO\PABLO\pablo\Memory\PreprocessedValidatedData"
        self.cache_path = os.path.join(self.db_path, "cache")
        os.makedirs(self.cache_path, exist_ok=True)
        os.makedirs(self.preprocessed_db_path, exist_ok=True)

    def get_exchange_instance(self, exchange_name):
        """Get CCXT exchange instance"""
        exchange_class = getattr(ccxt, exchange_name, None)
        if exchange_class:
            return exchange_class({
                'timeout': 30000,
                'enableRateLimit': True,
            })
        return None

    @cache_data(ttl=3600)
    def fetch_spot_pairs(self, exchange_name, use_cache=True):
        """Fetch all spot trading pairs from exchange using CCXT"""
        cache_file = os.path.join(self.cache_path, f"{exchange_name}_spot_pairs.json")

        if use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    if datetime.now().timestamp() - cached_data['timestamp'] < 3600:
                        return cached_data['pairs']
            except:
                pass

        if check_internet_connection():
            try:
                exchange = self.get_exchange_instance(exchange_name)
                if exchange:
                    markets = exchange.load_markets()
                    spot_pairs = []

                    for symbol, market in markets.items():
                        if market.get('spot', False) and market.get('active', False):
                            pair = symbol.replace('/', '_')
                            spot_pairs.append(pair)

                    cache_data_dict = {
                        'timestamp': datetime.now().timestamp(),
                        'pairs': sorted(spot_pairs)
                    }
                    with open(cache_file, 'w') as f:
                        json.dump(cache_data_dict, f)

                    logging.info(f"Fetched {len(spot_pairs)} spot pairs from {exchange_name}")
                    return spot_pairs

            except Exception as e:
                logging.error(f"Error fetching pairs from {exchange_name}: {e}")
                raise e

        return []

    def get_exchange_timeframes(self, exchange_name):
        """Get all possible timeframes for a specific exchange"""
        try:
            exchange = self.get_exchange_instance(exchange_name)
            if exchange:
                return exchange.timeframes
        except Exception as e:
            logging.error(f"Error getting timeframes for {exchange_name}: {e}")

        default_timeframes = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
            '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'
        }
        return default_timeframes

    @cache_data(ttl=3600)
    def get_available_exchanges(self):
        """Get list of available exchanges from database"""
        try:
            exchanges = [f for f in os.listdir(self.db_path)
                         if os.path.isdir(os.path.join(self.db_path, f)) and not f == "cache"]
            return exchanges
        except:
            return ['binance', 'bitmex', 'bybit', 'coinbase', 'cryptocom', 'gate', 'huobi', 'kucoin', 'okx']

    @cache_data(ttl=600)
    def get_trading_pairs(self, exchange):
        """Get available trading pairs for an exchange from database"""
        try:
            exchange_path = os.path.join(self.db_path, exchange)
            if not os.path.exists(exchange_path):
                return []
            pairs = [f for f in os.listdir(exchange_path)
                     if os.path.isdir(os.path.join(exchange_path, f))]
            return pairs
        except:
            return []

    @cache_data(ttl=600)
    def get_available_timeframes(self, exchange, pair):
        """Get available timeframes for a specific pair"""
        try:
            pair_path = os.path.join(self.db_path, exchange, pair)
            if not os.path.exists(pair_path):
                return []
            timeframes = [f.replace('.feather', '') for f in os.listdir(pair_path)
                          if f.endswith('.feather') and not f.endswith('_ALL.feather')]
            return timeframes
        except:
            return []

    def load_pair_data(self, exchange, pair, timeframe):
        """Load data for specific pair and timeframe"""
        try:
            file_path = os.path.join(self.db_path, exchange, pair, f"{timeframe}.feather")
            if os.path.exists(file_path):
                df = pd.read_feather(file_path)
                return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
        return None

    def get_existing_data_range(self, exchange, pair, timeframe):
        """Get the date range of existing data for a pair"""
        try:
            df = self.load_pair_data(exchange, pair, timeframe)
            if df is not None and not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df['timestamp'].min(), df['timestamp'].max()
        except Exception as e:
            logging.error(f"Error getting existing data range: {e}")
        return None, None

    def fetch_ohlcv_data(self, exchange_name, pair, timeframe, since, until):
        """Fetch OHLCV data from exchange with improved data validation"""
        if not check_internet_connection():
            st.error("Internet connection required for data fetching")
            return None

        try:
            exchange = self.get_exchange_instance(exchange_name)
            if not exchange:
                st.error(f"Exchange {exchange_name} not supported")
                return None

            symbol = pair.replace('_', '/')
            since_timestamp = exchange.parse8601(since.isoformat() + 'Z')
            until_timestamp = exchange.parse8601(until.isoformat() + 'Z')

            all_ohlcv = []
            current_since = since_timestamp

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                while current_since < until_timestamp:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
                    if not ohlcv:
                        break

                    all_ohlcv.extend(ohlcv)
                    current_since = ohlcv[-1][0] + 1

                    progress = min(1.0, (current_since - since_timestamp) / (until_timestamp - since_timestamp))
                    progress_bar.progress(progress)
                    status_text.text(f"Progress: {progress:.1%} - Fetched {len(all_ohlcv)} records")

            except Exception as e:
                logging.warning(f"Partial data fetched: {e}")

            progress_bar.empty()
            status_text.empty()

            if all_ohlcv:
                df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.drop_duplicates('timestamp').sort_values('timestamp')
                df = df[(df['timestamp'] >= since) & (df['timestamp'] <= until)]

                # Validate data quality
                df = self._validate_ohlcv_data(df)

                logging.info(f"Fetched {len(df)} records for {exchange_name} {pair} {timeframe}")
                return df
            else:
                st.warning("No data returned from exchange API")
                return None

        except Exception as e:
            logging.error(f"Error fetching OHLCV data for {pair}: {e}")
            st.error(f"Error fetching data: {str(e)}")
            return None

    def _validate_ohlcv_data(self, df):
        """Validate OHLCV data and fix common issues"""
        if df.empty:
            return df

        # Ensure proper data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove rows with null values in critical columns
        critical_columns = ['timestamp', 'open', 'high', 'low', 'close']
        df = df.dropna(subset=critical_columns)

        # Validate OHLC relationships
        mask = (
                (df['high'] >= df['low']) &
                (df['high'] >= df['open']) &
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) &
                (df['low'] <= df['close'])
        )

        if not mask.all():
            logging.warning(f"Found {len(df[~mask])} rows with invalid OHLC relationships")
            df = df[mask]

        # Ensure volume is non-negative
        if 'volume' in df.columns:
            df['volume'] = df['volume'].clip(lower=0)

        return df

    def save_pair_data(self, exchange, pair, timeframe, df, merge_strategy="append"):
        """Save data to database with improved merging logic"""
        try:
            pair_path = os.path.join(self.db_path, exchange, pair)
            os.makedirs(pair_path, exist_ok=True)

            file_path = os.path.join(pair_path, f"{timeframe}.feather")

            if os.path.exists(file_path) and merge_strategy == "append":
                # Load existing data and merge intelligently
                existing_df = pd.read_feather(file_path)

                # Ensure timestamp is datetime and set as index for merging
                if 'timestamp' in existing_df.columns and 'timestamp' in df.columns:
                    existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])

                    # Set timestamp as index for both dataframes
                    existing_df = existing_df.set_index('timestamp').sort_index()
                    df = df.set_index('timestamp').sort_index()

                    # Combine dataframes - use new data where available, otherwise keep existing
                    combined_df = self._smart_merge_data(existing_df, df)

                    # Reset index to get timestamp back as column
                    combined_df = combined_df.reset_index().rename(columns={'index': 'timestamp'})

                    combined_df.to_feather(file_path)
                    logging.info(
                        f"Smart merged data: {len(existing_df)} existing + {len(df)} new = {len(combined_df)} total for {exchange} {pair} {timeframe}")
                    return f"smart_merged ({len(combined_df)} total records)"
                else:
                    # Fallback: simple concat if timestamp column is missing
                    combined_df = pd.concat([existing_df, df]).drop_duplicates().sort_index()
                    combined_df.to_feather(file_path)
                    logging.info(f"Simple merged {len(df)} records for {exchange} {pair} {timeframe}")
                    return "merged_simple"
            else:
                # Replace or create new file
                df.to_feather(file_path)
                logging.info(f"Saved {len(df)} records for {exchange} {pair} {timeframe}")
                return "created"

        except Exception as e:
            logging.error(f"Error saving data for {pair}: {e}")
            st.error(f"Error saving data: {e}")
            return "error"

    def _smart_merge_data(self, existing_df, new_df):
        """Smart merge existing and new data without creating null values"""
        try:
            # Combine both dataframes
            combined_df = pd.concat([existing_df, new_df])

            # Remove exact duplicates (same timestamp and same values)
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]

            # Sort by timestamp
            combined_df = combined_df.sort_index()

            return combined_df

        except Exception as e:
            logging.error(f"Error in smart merge: {e}")
            # Fallback to simple concat
            return pd.concat([existing_df, new_df]).sort_index()

    def save_preprocessed_data(self, exchange, pair, algorithm, timeframe, processed_data, scaler, validation_results):
        """Save preprocessed and validated data to the preprocessed database"""
        try:
            algorithm_path = os.path.join(self.preprocessed_db_path, exchange, algorithm, pair)
            os.makedirs(algorithm_path, exist_ok=True)

            data_file_path = os.path.join(algorithm_path, f"{timeframe}_processed.feather")
            metadata_file_path = os.path.join(algorithm_path, f"{timeframe}_metadata.json")

            # Save processed data
            processed_data.to_feather(data_file_path)

            # Convert numpy types to Python native types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj

            # Prepare metadata with serializable types
            metadata = {
                'exchange': exchange,
                'pair': pair,
                'algorithm': algorithm,
                'timeframe': timeframe,
                'processed_timestamp': datetime.now().isoformat(),
                'records_count': int(len(processed_data)),  # Ensure integer
                'validation_results': convert_numpy_types(validation_results),
                'scaler_type': type(scaler).__name__ if scaler else 'None',
                'data_columns': list(processed_data.columns),
                'date_range': {
                    'start': processed_data[
                        'timestamp'].min().isoformat() if 'timestamp' in processed_data.columns else None,
                    'end': processed_data[
                        'timestamp'].max().isoformat() if 'timestamp' in processed_data.columns else None
                }
            }

            # Convert all numpy types in metadata
            metadata = convert_numpy_types(metadata)

            with open(metadata_file_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            logging.info(f"Saved preprocessed data for {exchange} {pair} {algorithm} {timeframe}")
            return True

        except Exception as e:
            logging.error(f"Error saving preprocessed data for {pair}: {e}")
            st.error(f"Error saving preprocessed data: {e}")
            return False

    def load_preprocessed_data(self, exchange, algorithm, pair, timeframe):
        """Load preprocessed data from the preprocessed database"""
        try:
            data_file_path = os.path.join(self.preprocessed_db_path, exchange, algorithm, pair,
                                          f"{timeframe}_processed.feather")
            metadata_file_path = os.path.join(self.preprocessed_db_path, exchange, algorithm, pair,
                                              f"{timeframe}_metadata.json")

            if os.path.exists(data_file_path):
                processed_data = pd.read_feather(data_file_path)
                metadata = {}
                if os.path.exists(metadata_file_path):
                    with open(metadata_file_path, 'r') as f:
                        metadata = json.load(f)
                return processed_data, metadata
            else:
                return None, None

        except Exception as e:
            logging.error(f"Error loading preprocessed data: {e}")
            return None, None

    def get_available_preprocessed_datasets(self):
        """Get list of all available preprocessed datasets"""
        datasets = []
        try:
            for exchange in os.listdir(self.preprocessed_db_path):
                exchange_path = os.path.join(self.preprocessed_db_path, exchange)
                if os.path.isdir(exchange_path):
                    for algorithm in os.listdir(exchange_path):
                        algorithm_path = os.path.join(exchange_path, algorithm)
                        if os.path.isdir(algorithm_path):
                            for pair in os.listdir(algorithm_path):
                                pair_path = os.path.join(algorithm_path, pair)
                                if os.path.isdir(pair_path):
                                    for file in os.listdir(pair_path):
                                        if file.endswith('_processed.feather'):
                                            timeframe = file.replace('_processed.feather', '')
                                            datasets.append({
                                                'exchange': exchange,
                                                'algorithm': algorithm,
                                                'pair': pair,
                                                'timeframe': timeframe
                                            })
            return datasets
        except Exception as e:
            logging.error(f"Error getting preprocessed datasets: {e}")
            return []

    def delete_preprocessed_data(self, exchange, algorithm, pair, timeframe):
        """Delete preprocessed data from database"""
        try:
            algorithm_path = os.path.join(self.preprocessed_db_path, exchange, algorithm, pair)
            data_file_path = os.path.join(algorithm_path, f"{timeframe}_processed.feather")
            metadata_file_path = os.path.join(algorithm_path, f"{timeframe}_metadata.json")

            if os.path.exists(data_file_path):
                os.remove(data_file_path)
                if os.path.exists(metadata_file_path):
                    os.remove(metadata_file_path)

                # Remove empty directories
                try:
                    if not os.listdir(algorithm_path):
                        os.rmdir(algorithm_path)
                    algorithm_parent = os.path.dirname(algorithm_path)
                    if not os.listdir(algorithm_parent):
                        os.rmdir(algorithm_parent)
                    exchange_parent = os.path.dirname(algorithm_parent)
                    if not os.listdir(exchange_parent):
                        os.rmdir(exchange_parent)
                except:
                    pass

                logging.info(f"Deleted preprocessed data for {exchange} {algorithm} {pair} {timeframe}")
                return True
            else:
                logging.warning(f"Preprocessed data not found for {exchange} {algorithm} {pair} {timeframe}")
                return False

        except Exception as e:
            logging.error(f"Error deleting preprocessed data: {e}")
            st.error(f"Error deleting preprocessed data: {e}")
            return False

    def check_pair_availability(self, pair):
        """Check which exchanges have the given trading pair"""
        available_exchanges = []
        exchange_options = ['binance', 'bitmex', 'bybit', 'coinbase', 'cryptocom', 'gate', 'huobi', 'kucoin', 'okx']

        for exchange in exchange_options:
            try:
                pairs = self.fetch_spot_pairs(exchange, use_cache=True)
                if pair in pairs:
                    available_exchanges.append(exchange)
            except Exception as e:
                print(f"Error checking {exchange} for {pair}: {e}")

        return available_exchanges

    def get_all_exchanges_timeframes(self):
        """Get all timeframes from all exchanges with exchange prefixes"""
        all_timeframes = {}
        exchange_options = ['binance', 'bitmex', 'bybit', 'coinbase', 'cryptocom', 'gate', 'huobi', 'kucoin', 'okx']

        for exchange in exchange_options:
            try:
                timeframes = self.get_exchange_timeframes(exchange)
                for tf_key, tf_value in timeframes.items():
                    display_name = f"{tf_value}-{exchange.capitalize()}"
                    all_timeframes[display_name] = {
                        'exchange': exchange,
                        'timeframe_key': tf_key,
                        'timeframe_value': tf_value
                    }
            except Exception as e:
                print(f"Error getting timeframes for {exchange}: {e}")

        return all_timeframes


