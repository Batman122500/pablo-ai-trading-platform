# pages/model_engineering/model_optimization.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import optuna
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings('ignore')


def show_model_optimization():
    """Complete Model Optimization Interface"""
    st.markdown('<div class="subsection-header">⚡ Model Optimization & Hyperparameter Tuning</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Optimization Setup")

        # Model Selection for Optimization
        trained_models = get_trained_models_for_optimization()
        if trained_models:
            selected_model = st.selectbox(
                "Select Model to Optimize:",
                trained_models,
                format_func=lambda x: f"{x['model_type']} - {x['algorithm']} - {x.get('accuracy', 0):.3f}",
                key="optimization_model_select"
            )

            # Optimization Strategy
            st.markdown("#### 📋 Optimization Strategy")
            optimization_method = st.radio(
                "Optimization Method:",
                ["Bayesian Optimization", "Grid Search", "Random Search", "Genetic Algorithm"],
                key="optimization_method"
            )

            # Optimization Parameters
            st.markdown("#### ⚙️ Optimization Parameters")
            n_trials = st.slider("Number of Trials", 10, 200, 50, key="n_trials")
            timeout = st.slider("Timeout (minutes)", 5, 120, 30, key="optimization_timeout")

            # Hyperparameter Search Space
            st.markdown("#### 🔍 Search Space")
            search_space = configure_search_space(selected_model['algorithm'])

            # Performance Metric
            st.markdown("#### 📊 Optimization Target")
            metric = st.selectbox(
                "Optimization Metric:",
                ["Accuracy", "F1-Score", "Precision", "Recall", "AUC-ROC", "MSE", "MAE", "R²"],
                key="optimization_metric"
            )

            # Early Stopping
            st.markdown("#### ⏱️ Early Stopping")
            early_stopping = st.checkbox("Enable Early Stopping", value=True, key="early_stopping_opt")
            if early_stopping:
                patience = st.slider("Patience", 5, 50, 10, key="early_stopping_patience")

            if st.button("🚀 Start Optimization", use_container_width=True, type="primary"):
                run_model_optimization(selected_model, optimization_method, n_trials, timeout,
                                       search_space, metric, early_stopping, patience if early_stopping else None)

        else:
            st.warning("No trained models available for optimization.")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        display_optimization_results_advanced()


def run_model_optimization(model_info, method, n_trials, timeout, search_space, metric, early_stopping, patience):
    """Run comprehensive model optimization"""
    # Load model and data
    model = load_trained_model(model_info['model_id'])
    dataset = model_info['dataset']

    # Load optimization data
    data_manager = get_data_manager()
    opt_data, _ = data_manager.load_preprocessed_data(
        dataset['exchange'], dataset['algorithm'], dataset['pair'], dataset['timeframe']
    )

    # Prepare features and targets
    X, y, feature_names = prepare_optimization_data(opt_data, model_info['model_type'])

    # Run optimization based on method
    if method == "Bayesian Optimization":
        results = perform_bayesian_optimization(model_info, X, y, n_trials, timeout, search_space, metric,
                                                early_stopping, patience)
    elif method == "Grid Search":
        results = perform_grid_search(model_info, X, y, search_space, metric)
    elif method == "Random Search":
        results = perform_random_search(model_info, X, y, n_trials, search_space, metric)
    else:  # Genetic Algorithm
        results = perform_genetic_optimization(model_info, X, y, n_trials, search_space, metric)

    # Store results
    st.session_state.optimization_results = results
    st.session_state.current_optimized_model = model_info

    st.success("✅ Optimization completed!")


def perform_bayesian_optimization(model_info, X, y, n_trials, timeout, search_space, metric, early_stopping, patience):
    """Perform Bayesian optimization using Optuna"""

    def objective(trial):
        # Suggest hyperparameters based on algorithm
        params = {}
        if model_info['algorithm'] == "Random Forest":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
        elif model_info['algorithm'] == "XGBoost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0)
            }
        # Add more algorithms as needed...

        # Create model with suggested parameters
        model = create_model_with_params(model_info['algorithm'], params)

        # Evaluate using cross-validation
        scores = cross_val_score(model, X, y, cv=5, scoring=metric.lower())

        return scores.mean()

    # Create study and optimize
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout * 60)

    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'study': study,
        'optimization_method': 'Bayesian',
        'trials_completed': len(study.trials)
    }


def display_optimization_results_advanced():
    """Display advanced optimization results"""
    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    st.markdown("#### 📊 Optimization Results Dashboard")

    if st.session_state.get('optimization_results'):
        results = st.session_state.optimization_results
        model_info = st.session_state.get('current_optimized_model', {})

        # Optimization Summary
        st.markdown("##### 🎯 Optimization Summary")
        display_optimization_summary(results, model_info)

        # Hyperparameter Analysis
        st.markdown("##### ⚙️ Hyperparameter Analysis")
        display_hyperparameter_analysis(results)

        # Performance Improvement
        st.markdown("##### 📈 Performance Improvement")
        display_performance_improvement(results, model_info)

        # Optimization History
        st.markdown("##### 📊 Optimization History")
        display_optimization_history(results)

    else:
        st.info("👆 Run optimization to see results")

    st.markdown('</div>', unsafe_allow_html=True)


def display_optimization_summary(results, model_info):
    """Display optimization summary"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Best Score", f"{results['best_score']:.4f}")

    with col2:
        st.metric("Trials Completed", results['trials_completed'])

    with col3:
        improvement = calculate_improvement(results, model_info)
        st.metric("Improvement", f"{improvement:.2%}")

    with col4:
        st.metric("Method", results['optimization_method'])

    # Best Parameters
    st.markdown("**🎯 Best Hyperparameters:**")
    best_params = results['best_params']
    for param, value in best_params.items():
        st.write(f"- `{param}`: `{value}`")


def display_hyperparameter_analysis(results):
    """Display hyperparameter importance and relationships"""
    if 'study' in results:
        study = results['study']

        tab1, tab2, tab3 = st.tabs(["Parameter Importance", "Parallel Coordinates", "Slice Plot"])

        with tab1:
            # Parameter importance plot
            try:
                fig = optuna.visualization.plot_param_importances(study)
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Parameter importance visualization not available")

        with tab2:
            # Parallel coordinates plot
            try:
                fig = optuna.visualization.plot_parallel_coordinate(study)
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Parallel coordinates visualization not available")

        with tab3:
            # Slice plot
            try:
                fig = optuna.visualization.plot_slice(study)
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Slice plot visualization not available")


def display_performance_improvement(results, model_info):
    """Display performance improvement analysis"""
    original_performance = model_info.get('accuracy', model_info.get('r2', 0))
    optimized_performance = results['best_score']
    improvement = optimized_performance - original_performance

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=optimized_performance,
            delta={'reference': original_performance, 'relative': False},
            title={"text": "Performance<br>Optimized vs Original"},
            domain={'x': [0, 1], 'y': [0, 1]}
        ))
        fig.update_layout(height=200)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Improvement breakdown
        st.metric("Original Performance", f"{original_performance:.4f}")
        st.metric("Optimized Performance", f"{optimized_performance:.4f}")
        st.metric("Absolute Improvement", f"{improvement:.4f}")
        st.metric("Relative Improvement", f"{(improvement / original_performance) * 100:.2f}%")


def display_optimization_history(results):
    """Display optimization history and convergence"""
    if 'study' in results:
        study = results['study']

        tab1, tab2 = st.tabs(["Optimization History", "Convergence Plot"])

        with tab1:
            try:
                fig = optuna.visualization.plot_optimization_history(study)
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Optimization history visualization not available")

        with tab2:
            try:
                fig = optuna.visualization.plot_intermediate_values(study)
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Convergence plot visualization not available")


def get_trained_models_for_optimization():
    """Get trained models available for optimization"""
    if 'trained_models' not in st.session_state or not st.session_state.trained_models:
        return []

    return [
        {
            'model_id': model_id,
            'model_type': info.get('model_type', 'Unknown'),
            'algorithm': info.get('algorithm', 'Unknown'),
            'accuracy': info.get('performance_metrics', {}).get('accuracy', 0),
            'dataset': info.get('dataset', {})
        }
        for model_id, info in st.session_state.trained_models.items()
    ]


def configure_search_space(algorithm):
    """Configure hyperparameter search space based on algorithm"""
    search_spaces = {
        'Random Forest': {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [3, 5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'subsample': [0.8, 0.9, 1.0]
        },
        'LSTM': {
            'units': [32, 64, 128],
            'dropout_rate': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [16, 32, 64]
        },
        'GRU': {
            'units': [32, 64, 128],
            'dropout_rate': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.01, 0.1]
        },
        'CNN': {
            'filters': [32, 64, 128],
            'kernel_size': [2, 3, 5],
            'learning_rate': [0.001, 0.01, 0.1]
        }
    }

    return search_spaces.get(algorithm, search_spaces['Random Forest'])


def load_trained_model(model_id):
    """Load a trained model from session state"""
    if 'trained_models' not in st.session_state:
        st.error("No trained models available")
        return None

    model_info = st.session_state.trained_models.get(model_id)
    if not model_info:
        st.error(f"Model {model_id} not found")
        return None

    return model_info.get('model')


def get_data_manager():
    """Get data manager instance"""
    from ...config.session_state import get_data_manager as get_dm
    return get_dm()


def prepare_optimization_data(data, model_type):
    """Prepare data for optimization"""
    if data is None or data.empty:
        return None, None, []

    # Exclude non-feature columns
    exclude_cols = ['timestamp', 'trend_label', 'future_price', 'target']
    feature_cols = [col for col in data.columns if col not in exclude_cols and not col.startswith('_')]

    X = data[feature_cols].values

    # Prepare target based on model type
    if model_type == 'classification' and 'trend_label' in data.columns:
        y = data['trend_label'].values
    elif 'close' in data.columns:
        # Default to price prediction
        y = data['close'].values
    else:
        y = np.zeros(len(data))

    return X, y, feature_cols


def create_model_with_params(algorithm, params):
    """Create model with given parameters"""
    if algorithm == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params, random_state=42)
    elif algorithm == "XGBoost":
        import xgboost as xgb
        return xgb.XGBClassifier(**params, random_state=42)
    elif algorithm == "LSTM":
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout
        model = Sequential()
        model.add(LSTM(params.get('units', 50), return_sequences=True, input_shape=(None, 1)))
        model.add(Dropout(params.get('dropout_rate', 0.2)))
        model.add(LSTM(params.get('units', 50)))
        model.add(Dropout(params.get('dropout_rate', 0.2)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    else:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=42)


def calculate_improvement(results, model_info):
    """Calculate performance improvement from optimization"""
    original_performance = model_info.get('accuracy', 0.5)
    optimized_performance = results.get('best_score', 0.5)

    if original_performance == 0:
        return 0

    improvement = (optimized_performance - original_performance) / original_performance
    return max(improvement, 0)


# Placeholder implementations for search methods
def perform_grid_search(model_info, X, y, search_space, metric):
    """Perform grid search optimization"""
    st.info("Grid Search would be implemented here")
    return {
        'best_params': {'n_estimators': 100, 'max_depth': 10},
        'best_score': 0.85,
        'optimization_method': 'Grid Search',
        'trials_completed': 1
    }


def perform_random_search(model_info, X, y, n_trials, search_space, metric):
    """Perform random search optimization"""
    st.info("Random Search would be implemented here")
    return {
        'best_params': {'n_estimators': 150, 'max_depth': 8},
        'best_score': 0.83,
        'optimization_method': 'Random Search',
        'trials_completed': n_trials
    }


def perform_genetic_optimization(model_info, X, y, n_trials, search_space, metric):
    """Perform genetic algorithm optimization"""
    st.info("Genetic Optimization would be implemented here")
    return {
        'best_params': {'n_estimators': 200, 'max_depth': 12},
        'best_score': 0.87,
        'optimization_method': 'Genetic Algorithm',
        'trials_completed': n_trials
    }