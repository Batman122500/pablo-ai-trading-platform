import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class AlgorithmSpecificPreprocessor:
    """Enhanced preprocessor with multi-model support and trend classification"""

    def __init__(self):
        self.algorithm_configs = {
            'LSTM': {
                'description': 'Long Short-Term Memory - Sequential data processing',
                'required_columns': ['open', 'high', 'low', 'close', 'volume'],
                'normalization': 'minmax',
                'sequence_length': 60,
                'handle_missing': 'interpolate',
                'feature_engineering': True,
                'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15},
                'type': 'regression'
            },
            'GRU': {
                'description': 'Gated Recurrent Unit - Efficient sequential processing',
                'required_columns': ['open', 'high', 'low', 'close', 'volume'],
                'normalization': 'minmax',
                'sequence_length': 50,
                'handle_missing': 'interpolate',
                'feature_engineering': True,
                'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15},
                'type': 'regression'
            },
            'Gradient Boosting': {
                'description': 'Gradient Boosting Machines - Tree-based ensemble',
                'required_columns': ['open', 'high', 'low', 'close', 'volume'],
                'normalization': 'standard',
                'handle_missing': 'drop',
                'feature_engineering': True,
                'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15},
                'type': 'regression'
            },
            'Random Forest': {
                'description': 'Random Forest - Robust ensemble method',
                'required_columns': ['open', 'high', 'low', 'close', 'volume'],
                'normalization': 'standard',
                'handle_missing': 'drop',
                'feature_engineering': True,
                'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15},
                'type': 'regression'
            },
            'CNN': {
                'description': 'Convolutional Neural Network - Pattern recognition',
                'required_columns': ['open', 'high', 'low', 'close', 'volume'],
                'normalization': 'minmax',
                'handle_missing': 'interpolate',
                'feature_engineering': True,
                'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15},
                'type': 'regression'
            },
            'Transformer': {
                'description': 'Transformer Architecture - Attention mechanisms',
                'required_columns': ['open', 'high', 'low', 'close', 'volume'],
                'normalization': 'minmax',
                'handle_missing': 'interpolate',
                'feature_engineering': True,
                'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15},
                'type': 'regression'
            },
            'Classification': {
                'description': 'Classification algorithms for trend identification',
                'required_columns': ['open', 'high', 'low', 'close', 'volume'],
                'normalization': 'standard',
                'handle_missing': 'interpolate',
                'feature_engineering': True,
                'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15},
                'type': 'classification',
                'classification_labels': True
            },
            'Deep Reinforcement Learning': {
                'description': 'Deep Reinforcement Learning - Reward-based learning',
                'required_columns': ['open', 'high', 'low', 'close', 'volume'],
                'normalization': 'standard',
                'handle_missing': 'interpolate',
                'feature_engineering': True,
                'split_ratios': {'train': 0.8, 'val': 0.1, 'test': 0.1},
                'type': 'reinforcement'
            },
            'XGBoost': {
                'description': 'Extreme Gradient Boosting - Optimized gradient boosting',
                'required_columns': ['open', 'high', 'low', 'close', 'volume'],
                'normalization': 'standard',
                'handle_missing': 'drop',
                'feature_engineering': True,
                'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15},
                'type': 'regression'
            },
            'LightGBM': {
                'description': 'Light Gradient Boosting - Fast gradient boosting',
                'required_columns': ['open', 'high', 'low', 'close', 'volume'],
                'normalization': 'standard',
                'handle_missing': 'drop',
                'feature_engineering': True,
                'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15},
                'type': 'regression'
            },
            'Ensemble': {
                'description': 'Ensemble Methods - Combined model predictions',
                'required_columns': ['open', 'high', 'low', 'close', 'volume'],
                'normalization': 'standard',
                'handle_missing': 'drop',
                'feature_engineering': True,
                'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15},
                'type': 'ensemble'
            },
            'ARIMA': {
                'description': 'AutoRegressive Integrated Moving Average - Statistical forecasting',
                'required_columns': ['close'],
                'normalization': 'none',
                'handle_missing': 'interpolate',
                'feature_engineering': False,
                'split_ratios': {'train': 0.8, 'val': 0.1, 'test': 0.1},
                'type': 'statistical'
            }
        }

        self.technical_indicators = {
            'Moving Averages': ['SMA', 'EMA', 'WMA'],
            'Momentum': ['RSI', 'MACD', 'Stochastic', 'Williams %R', 'CCI', 'ROC', 'MFI', 'TSI'],
            'Volatility': ['Bollinger_Bands', 'ATR', 'Keltner_Channel', 'Donchian_Channel', 'STD', 'VAR', 'CHV'],
            'Volume': ['Volume_SMA', 'OBV', 'Volume_Profile', 'ADL', 'CMF'],
            'Support_Resistance': ['Pivot_Points', 'Fibonacci', 'Support_Resistance_Levels']
        }

    def get_algorithm_config(self, algorithm_name):
        """Get configuration for specific algorithm"""
        return self.algorithm_configs.get(algorithm_name, {})

    def clean_data_algorithm_specific(self, df, algorithm_name):
        """Enhanced cleaning based on algorithm requirements"""
        config = self.get_algorithm_config(algorithm_name)

        if df is None or df.empty:
            return df

        df_clean = df.copy()

        handle_method = config.get('handle_missing', 'interpolate')
        if handle_method == 'interpolate':
            df_clean = df_clean.interpolate(method='linear', limit_direction='both')
        elif handle_method == 'drop':
            df_clean = df_clean.dropna()
        elif handle_method == 'fill':
            df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')

        df_clean = df_clean.drop_duplicates()

        return df_clean

    def normalize_data(self, df, algorithm_name):
        """Normalize data based on algorithm requirements"""
        config = self.get_algorithm_config(algorithm_name)
        normalization_method = config.get('normalization', 'standard')

        if normalization_method == 'none':
            return df, None

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_normalized = df.copy()

        if normalization_method == 'standard':
            scaler = StandardScaler()
            df_normalized[numeric_cols] = scaler.fit_transform(df_normalized[numeric_cols])
        elif normalization_method == 'minmax':
            scaler = MinMaxScaler()
            df_normalized[numeric_cols] = scaler.fit_transform(df_normalized[numeric_cols])

        return df_normalized, scaler

    def add_classification_labels(self, df, lookforward_period=5, threshold=0.02):
        """Add classification labels for trend prediction"""
        df_classified = df.copy()

        df_classified['future_close'] = df_classified['close'].shift(-lookforward_period)
        df_classified['price_change'] = (df_classified['future_close'] - df_classified['close']) / df_classified['close']

        df_classified['trend_label'] = 0
        df_classified.loc[df_classified['price_change'] > threshold, 'trend_label'] = 1
        df_classified.loc[df_classified['price_change'] < -threshold, 'trend_label'] = -1

        df_classified = df_classified.drop(['future_close', 'price_change'], axis=1)

        return df_classified

    def calculate_technical_indicators(self, df, selected_indicators):
        """Calculate comprehensive technical indicators with momentum and volatility"""
        df_with_indicators = df.copy()

        # Moving Averages
        if 'SMA' in selected_indicators:
            df_with_indicators['SMA_20'] = ta.trend.SMAIndicator(df_with_indicators['close'], window=20).sma_indicator()
            df_with_indicators['SMA_50'] = ta.trend.SMAIndicator(df_with_indicators['close'], window=50).sma_indicator()
            df_with_indicators['SMA_200'] = ta.trend.SMAIndicator(df_with_indicators['close'], window=200).sma_indicator()

        if 'EMA' in selected_indicators:
            df_with_indicators['EMA_20'] = ta.trend.EMAIndicator(df_with_indicators['close'], window=20).ema_indicator()
            df_with_indicators['EMA_50'] = ta.trend.EMAIndicator(df_with_indicators['close'], window=50).ema_indicator()

        if 'WMA' in selected_indicators:
            df_with_indicators['WMA_20'] = ta.trend.WMAIndicator(df_with_indicators['close'], window=20).wma()

        # Momentum Indicators
        if 'RSI' in selected_indicators:
            df_with_indicators['RSI_14'] = ta.momentum.RSIIndicator(df_with_indicators['close'], window=14).rsi()
            df_with_indicators['RSI_21'] = ta.momentum.RSIIndicator(df_with_indicators['close'], window=21).rsi()

        if 'MACD' in selected_indicators:
            macd = ta.trend.MACD(df_with_indicators['close'])
            df_with_indicators['MACD'] = macd.macd()
            df_with_indicators['MACD_signal'] = macd.macd_signal()
            df_with_indicators['MACD_hist'] = macd.macd_diff()

        if 'Stochastic' in selected_indicators:
            stoch = ta.momentum.StochasticOscillator(df_with_indicators['high'], df_with_indicators['low'], df_with_indicators['close'])
            df_with_indicators['Stoch_%K'] = stoch.stoch()
            df_with_indicators['Stoch_%D'] = stoch.stoch_signal()

        if 'Williams %R' in selected_indicators:
            df_with_indicators['Williams_%R'] = ta.momentum.WilliamsRIndicator(df_with_indicators['high'], df_with_indicators['low'], df_with_indicators['close']).williams_r()

        if 'CCI' in selected_indicators:
            df_with_indicators['CCI_20'] = ta.trend.CCIIndicator(df_with_indicators['high'], df_with_indicators['low'], df_with_indicators['close'], window=20).cci()

        # Additional Momentum Indicators
        if 'ROC' in selected_indicators:
            df_with_indicators['ROC_10'] = ta.momentum.ROCIndicator(df_with_indicators['close'], window=10).roc()
            df_with_indicators['ROC_21'] = ta.momentum.ROCIndicator(df_with_indicators['close'], window=21).roc()

        if 'MFI' in selected_indicators:
            df_with_indicators['MFI_14'] = ta.volume.MFIIndicator(df_with_indicators['high'], df_with_indicators['low'], df_with_indicators['close'], df_with_indicators['volume'], window=14).money_flow_index()

        if 'TSI' in selected_indicators:
            df_with_indicators['TSI'] = ta.momentum.TSIIndicator(df_with_indicators['close']).tsi()

        # Volatility Indicators
        if 'Bollinger_Bands' in selected_indicators:
            bb = ta.volatility.BollingerBands(df_with_indicators['close'])
            df_with_indicators['BB_upper'] = bb.bollinger_hband()
            df_with_indicators['BB_lower'] = bb.bollinger_lband()
            df_with_indicators['BB_middle'] = bb.bollinger_mavg()
            df_with_indicators['BB_width'] = (df_with_indicators['BB_upper'] - df_with_indicators['BB_lower']) / df_with_indicators['BB_middle']

        if 'ATR' in selected_indicators:
            df_with_indicators['ATR_14'] = ta.volatility.AverageTrueRange(df_with_indicators['high'], df_with_indicators['low'], df_with_indicators['close'], window=14).average_true_range()

        if 'Keltner_Channel' in selected_indicators:
            df_with_indicators['KC_upper'] = df_with_indicators['EMA_20'] + (2 * df_with_indicators['ATR_14'])
            df_with_indicators['KC_lower'] = df_with_indicators['EMA_20'] - (2 * df_with_indicators['ATR_14'])

        # Additional Volatility Indicators
        if 'STD' in selected_indicators:
            df_with_indicators['STD_20'] = df_with_indicators['close'].rolling(window=20).std()

        if 'VAR' in selected_indicators:
            df_with_indicators['VAR_20'] = df_with_indicators['close'].rolling(window=20).var()

        if 'CHV' in selected_indicators:
            hl_range = df_with_indicators['high'] - df_with_indicators['low']
            ema_hl = hl_range.ewm(span=10).mean()
            df_with_indicators['CHV'] = (ema_hl - ema_hl.shift(10)) / ema_hl.shift(10) * 100

        # Volume Indicators
        if 'Volume_SMA' in selected_indicators:
            df_with_indicators['Volume_SMA_20'] = df_with_indicators['volume'].rolling(window=20).mean()

        if 'OBV' in selected_indicators:
            df_with_indicators['OBV'] = ta.volume.OnBalanceVolumeIndicator(df_with_indicators['close'], df_with_indicators['volume']).on_balance_volume()

        if 'ADL' in selected_indicators:
            df_with_indicators['ADL'] = ta.volume.AccDistIndexIndicator(df_with_indicators['high'], df_with_indicators['low'], df_with_indicators['close'], df_with_indicators['volume']).acc_dist_index()

        if 'CMF' in selected_indicators:
            df_with_indicators['CMF_20'] = ta.volume.ChaikinMoneyFlowIndicator(df_with_indicators['high'], df_with_indicators['low'], df_with_indicators['close'], df_with_indicators['volume'], window=20).chaikin_money_flow()

        return df_with_indicators

    def validate_processed_data(self, df, algorithm_name):
        """Validate processed data quality"""
        config = self.get_algorithm_config(algorithm_name)
        validation_results = {}

        required_cols = config.get('required_columns', [])
        missing_cols = [col for col in required_cols if col not in df.columns]
        validation_results['missing_columns'] = missing_cols
        validation_results['has_required_columns'] = len(missing_cols) == 0

        validation_results['total_records'] = len(df)
        validation_results['null_count'] = df.isnull().sum().sum()
        validation_results['completeness'] = (1 - validation_results['null_count'] / (len(df) * len(df.columns))) * 100

        if config.get('sequence_length'):
            min_required = config['sequence_length'] * 10
            validation_results['sufficient_for_sequences'] = len(df) >= min_required
            validation_results['min_required_records'] = min_required

        if config.get('type') == 'classification':
            validation_results['has_classification_labels'] = 'trend_label' in df.columns

        return validation_results

    def split_data_for_training(self, df, algorithm_name):
        """Split data according to algorithm-specific ratios"""
        config = self.get_algorithm_config(algorithm_name)
        split_ratios = config.get('split_ratios', {'train': 0.7, 'val': 0.15, 'test': 0.15})

        train_ratio = split_ratios['train']
        val_ratio = split_ratios['val']
        test_ratio = split_ratios['test']

        from ..managers.preprocessor import DataPreprocessor
        return DataPreprocessor.split_data(df, train_ratio, val_ratio, test_ratio)