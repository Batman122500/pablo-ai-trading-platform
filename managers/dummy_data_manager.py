import streamlit as st
import pandas as pd
from datetime import datetime


class DummyDataManager:
    """Fallback data manager to prevent crashes when main data manager fails"""

    def __init__(self):
        self.db_path = r"C:\Users\DELL\Desktop\PABLO\PABLO\pablo\Memory\CryptoDatabase"
        self.preprocessed_db_path = r"C:\Users\DELL\Desktop\PABLO\PABLO\pablo\Memory\PreprocessedValidatedData"

    def get_available_exchanges(self):
        """Return default exchanges"""
        return ['binance', 'bitmex', 'bybit', 'coinbase', 'cryptocom', 'gate', 'huobi', 'kucoin', 'okx']

    def get_trading_pairs(self, exchange):
        """Return empty list for trading pairs"""
        return []

    def get_available_timeframes(self, exchange, pair):
        """Return empty list for timeframes"""
        return []

    def load_pair_data(self, exchange, pair, timeframe):
        """Return None for data"""
        return None

    def fetch_spot_pairs(self, exchange_name, use_cache=True):
        """Return empty list for spot pairs"""
        st.warning("Using fallback data manager - internet features disabled")
        return []

    def get_exchange_timeframes(self, exchange_name):
        """Return default timeframes"""
        return {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
            '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'
        }

    def fetch_ohlcv_data(self, exchange_name, pair, timeframe, since, until):
        """Return None for OHLCV data"""
        st.warning("Data fetching disabled in fallback mode")
        return None

    def save_pair_data(self, exchange, pair, timeframe, df):
        """Return error for save operations"""
        return "error"

    def save_preprocessed_data(self, exchange, pair, algorithm, timeframe, processed_data, scaler, validation_results):
        """Return False for save operations"""
        return False

    def load_preprocessed_data(self, exchange, algorithm, pair, timeframe):
        """Return None for preprocessed data"""
        return None, None

    def get_available_preprocessed_datasets(self):
        """Return empty list for datasets"""
        return []

    def delete_preprocessed_data(self, exchange, algorithm, pair, timeframe):
        """Return False for delete operations"""
        return False