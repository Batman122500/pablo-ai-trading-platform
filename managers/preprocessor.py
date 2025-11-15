import pandas as pd
import numpy as np

class DataPreprocessor:
    """Enhanced data preprocessing and validation"""

    @staticmethod
    def clean_data(df):
        """Enhanced data cleaning that preserves candle structure"""
        if df is None:
            return df

        result_df = df.copy()

        # First, ensure timestamp is proper datetime and sort
        if 'timestamp' in result_df.columns:
            result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
            result_df = result_df.sort_values('timestamp').reset_index(drop=True)

        # Remove exact duplicates
        result_df = result_df.drop_duplicates()

        # Forward fill then backward fill missing values to preserve continuity
        result_df = result_df.fillna(method='ffill')
        result_df = result_df.fillna(method='bfill')

        # For OHLCV data, ensure logical consistency
        if all(col in result_df.columns for col in ['open', 'high', 'low', 'close']):
            # Fix high/low inconsistencies
            result_df['high'] = result_df[['open', 'high', 'low', 'close']].max(axis=1)
            result_df['low'] = result_df[['open', 'high', 'low', 'close']].min(axis=1)

            # Remove rows where essential data is still missing
            result_df = result_df.dropna(subset=['open', 'high', 'low', 'close'])

            # Remove extreme outliers using more conservative approach
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in result_df.columns:
                    Q1 = result_df[col].quantile(0.05)  # More conservative 5% quartile
                    Q3 = result_df[col].quantile(0.95)  # More conservative 95% quartile
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3 * IQR  # More tolerant bounds
                    upper_bound = Q3 + 3 * IQR
                    result_df = result_df[(result_df[col] >= lower_bound) & (result_df[col] <= upper_bound)]

        return result_df

    @staticmethod
    def validate_data_quality(df):
        """Validate data quality and return metrics"""
        if df is None:
            return {}

        quality_metrics = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'null_count': df.isnull().sum().sum(),
            'completeness_score': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_count': df.duplicated().sum(),
            'date_range': f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}" if 'timestamp' in df.columns else 'N/A',
            'data_span_days': (df['timestamp'].max() - df['timestamp'].min()).days if 'timestamp' in df.columns else 0,
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 ** 2
        }

        return quality_metrics

    @staticmethod
    def split_data(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split data into train, validation, and test sets"""
        if df is None or df.empty:
            return None, None, None

        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            train_ratio = 0.7
            val_ratio = 0.15
            test_ratio = 0.15

        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')

        total_size = len(df)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)

        train_data = df.iloc[:train_size]
        val_data = df.iloc[train_size:train_size + val_size]
        test_data = df.iloc[train_size + val_size:]

        return train_data, val_data, test_data