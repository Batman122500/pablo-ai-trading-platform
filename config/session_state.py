import streamlit as st
from ..managers.data_manager import ExchangeDataManager
from ..managers.algo_preprocessor import AlgorithmSpecificPreprocessor
from ..managers.preprocessor import DataPreprocessor
from ..managers.dummy_data_manager import DummyDataManager


def initialize_session_state():
    """Comprehensive session state initialization with better error handling"""

    # Base session state variables
    default_states = {
        'initialized': True,
        'data_manager': None,
        'algo_preprocessor': None,
        'preprocessor': None,

        # Exchange / Pair data
        'available_exchanges': None,
        'spot_pairs': None,
        'selected_management_exchange': None,

        # Preprocessing
        'processing_active': False,
        'processing_progress': 0,
        'processing_status': "",
        'preprocessing_results': {},

        # UI navigation
        'current_tab': "dashboard",
        'button_counter': 0,
        'current_page': "Main Dashboard",
        'main_navigation': "Main Dashboard",

        # Current selections
        'current_exchange': None,
        'current_pair': None,
        'current_algorithm': None,

        # Chart settings
        'selected_timeframe_for_chart': None,
        'selected_indicators_for_chart': [],

        # Background + multi-exchange management
        'background_tasks': {},
        'exchange_pairs': {},
        'all_exchanges_pairs': {},
        'timeframe_mapping': {},
        'multi_exchange_fetch': False,

        # Recent saved datasets
        'last_saved_datasets': [],

        # =============================
        # 🔥 MODEL ENGINEERING STATE
        # =============================

        # All trained or loaded models
        'trained_models': {},

        # Active training session
        'current_training': None,

        # Validation & Test
        'validation_results': {},
        'test_results': {},

        # Hyperparameter optimization
        'optimization_results': {},

        # Adaptive / online models
        'adaptive_models': {},

        # Model registry
        'model_registry': {},

        # Training tracking
        'training_in_progress': False,
        'training_progress': 0,
        'training_metrics': [],

        # Performance history
        'model_performance_history': {},
        'best_models': {},
    }

    # Initialize missing state variables only
    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # 🔥 CRITICAL: Force initialize data_manager if it's still None
    if st.session_state.data_manager is None:
        try:
            from ..managers.data_manager import ExchangeDataManager
            st.session_state.data_manager = ExchangeDataManager()
        except Exception as e:
            st.error(f"Failed to initialize data manager: {e}")
            from ..managers.dummy_data_manager import DummyDataManager
            st.session_state.data_manager = DummyDataManager()


def get_data_manager():
    """Lazy initialization of data manager with better error handling"""
    # 🔥 CRITICAL: Check if data_manager exists and is valid
    if (st.session_state.data_manager is None or
            not hasattr(st.session_state.data_manager, 'get_available_exchanges')):
        initialize_session_state()  # Re-initialize if corrupted

    return st.session_state.data_manager


def cleanup_on_navigation():
    """Clean up heavy data when navigating away"""
    current_page = st.session_state.get('current_page', '')
    new_page = st.session_state.get('main_navigation', 'Main Dashboard')

    if current_page != new_page:

        # Heavy cleanup when leaving Data Management
        if current_page == "Data Management":
            st.session_state.preprocessing_results = {}
            st.session_state.spot_pairs = None

        # Update page state
        st.session_state.current_page = new_page


def get_data_manager():
    """Lazy initialization of data manager with better error handling"""
    if st.session_state.data_manager is None:
        try:
            st.session_state.data_manager = ExchangeDataManager()

            if st.session_state.data_manager is None:
                st.error("Failed to create data manager instance")
                st.session_state.data_manager = DummyDataManager()

        except Exception as e:
            st.error(f"Failed to initialize data manager: {e}")
            st.session_state.data_manager = DummyDataManager()

    return st.session_state.data_manager


def get_algo_preprocessor():
    """Lazy load algorithm-specific preprocessor"""
    if st.session_state.algo_preprocessor is None:
        st.session_state.algo_preprocessor = AlgorithmSpecificPreprocessor()
    return st.session_state.algo_preprocessor


def get_preprocessor():
    """Lazy load general data preprocessor"""
    if st.session_state.preprocessor is None:
        st.session_state.preprocessor = DataPreprocessor()
    return st.session_state.preprocessor


def update_progress(progress, status=""):
    """Update global progress bar + status"""
    st.session_state.processing_progress = progress
    if status:
        st.session_state.processing_status = status


def get_unique_key(prefix):
    """Generate a unique key for Streamlit widgets"""
    st.session_state.button_counter += 1
    return f"{prefix}_{st.session_state.button_counter}"


def mark_data_saved(dataset_info):
    """Track recently saved datasets"""
    if 'last_saved_datasets' not in st.session_state:
        st.session_state.last_saved_datasets = []

    st.session_state.last_saved_datasets.append({
        'timestamp': __import__('datetime').datetime.now(),
        'dataset': dataset_info
    })

    # Keep only last 10 entries
    if len(st.session_state.last_saved_datasets) > 10:
        st.session_state.last_saved_datasets = st.session_state.last_saved_datasets[-10:]

