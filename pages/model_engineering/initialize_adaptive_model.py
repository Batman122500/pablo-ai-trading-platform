import streamlit as st
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam


def initialize_adaptive_model(model_type, learning_rate, forgetting_factor, update_frequency):
    """Initialize adaptive learning models"""

    model_config = {
        'model_type': model_type,
        'learning_rate': learning_rate,
        'forgetting_factor': forgetting_factor,
        'update_frequency': update_frequency,
        'initialized_at': __import__('datetime').datetime.now().isoformat(),
        'samples_processed': 0,
        'model_updates': 0
    }

    # Initialize appropriate model based on type
    if model_type == "Online Price Direction Classifier":
        model = SGDClassifier(
            loss='log_loss',
            learning_rate='adaptive',
            eta0=learning_rate,
            random_state=42
        )
    elif model_type == "Streaming Movement Regressor":
        model = xgb.XGBRegressor(
            learning_rate=learning_rate,
            n_estimators=100,
            random_state=42
        )
    elif model_type == "Real-time Volatility Forecaster":
        # Simple neural network for volatility
        model = Sequential([
            Dense(64, activation='relu', input_shape=(10,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    else:
        # Default to random forest
        model = RandomForestClassifier(
            n_estimators=50,
            random_state=42
        )

    # Store in session state
    if 'adaptive_models' not in st.session_state:
        st.session_state.adaptive_models = {}

    model_id = f"{model_type}_{__import__('time').time()}"
    st.session_state.adaptive_models[model_id] = {
        'model': model,
        'config': model_config,
        'performance_history': [],
        'last_update': __import__('datetime').datetime.now()
    }

    return model_id