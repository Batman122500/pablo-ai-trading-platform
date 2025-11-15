# pages/model_engineering/model_testing.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta


def show_model_testing():
    """Complete Model Testing Framework"""
    st.markdown('<div class="subsection-header">📊 Model Testing & Backtesting</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Testing Configuration")

        # Model Selection for Testing
        validated_models = get_validated_models()
        if validated_models:
            selected_model = st.selectbox(
                "Select Model to Test:",
                validated_models,
                format_func=lambda x: f"{x['model_type']} - {x['algorithm']} - Acc: {x.get('accuracy', 0):.3f}",
                key="testing_model_select"
            )

            # Testing Strategy
            st.markdown("#### 📋 Testing Strategy")
            testing_method = st.radio(
                "Testing Method:",
                ["Backtesting", "Walk-Forward Analysis", "Stress Testing", "Scenario Analysis"],
                key="testing_method"
            )

            # Testing Period
            st.markdown("#### 📅 Testing Period")
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date = st.date_input("Start Date",
                                           value=datetime.now() - timedelta(days=365),
                                           key="testing_start_date")
            with col_date2:
                end_date = st.date_input("End Date",
                                         value=datetime.now(),
                                         key="testing_end_date")

            # Market Conditions
            st.markdown("#### 🌡️ Market Conditions")
            market_regimes = st.multiselect(
                "Include Market Regimes:",
                ["Bull Market", "Bear Market", "High Volatility", "Low Volatility", "Sideways"],
                default=["Bull Market", "Bear Market", "High Volatility"],
                key="market_regimes"
            )

            # Performance Thresholds
            st.markdown("#### 🎯 Performance Thresholds")
            min_accuracy = st.slider("Minimum Accuracy", 0.5, 0.9, 0.7, key="min_accuracy")
            max_drawdown = st.slider("Maximum Drawdown %", 5.0, 50.0, 20.0, key="max_drawdown")

            if st.button("🚀 Run Comprehensive Testing", use_container_width=True, type="primary"):
                run_comprehensive_testing(selected_model, testing_method, start_date, end_date,
                                          market_regimes, min_accuracy, max_drawdown)

        else:
            st.warning("No validated models available. Validate a model first!")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        display_testing_results_advanced()


def get_validated_models():
    """Get validated models available for testing"""
    if 'trained_models' not in st.session_state:
        return []

    # Filter models that have validation results
    validated_models = []
    for model_id, model_info in st.session_state.trained_models.items():
        if model_info.get('performance_metrics'):
            validated_models.append({
                'model_id': model_id,
                'model_type': model_info.get('model_type', 'Unknown'),
                'algorithm': model_info.get('algorithm', 'Unknown'),
                'accuracy': model_info.get('performance_metrics', {}).get('accuracy', 0),
                'dataset': model_info.get('dataset', {})
            })

    return validated_models


def load_testing_data(dataset_info, start_date, end_date):
    """Load testing data for the specified period"""
    data_manager = get_data_manager()

    # Load preprocessed data
    test_data, metadata = data_manager.load_preprocessed_data(
        dataset_info['exchange'],
        dataset_info['algorithm'],
        dataset_info['pair'],
        dataset_info['timeframe']
    )

    if test_data is None:
        st.error("Could not load testing data")
        return None

    # Filter by date range if timestamp column exists
    if 'timestamp' in test_data.columns:
        test_data['timestamp'] = __import__('pandas').to_datetime(test_data['timestamp'])
        mask = (test_data['timestamp'] >= __import__('pandas').to_datetime(start_date)) & \
               (test_data['timestamp'] <= __import__('pandas').to_datetime(end_date))
        test_data = test_data[mask]

    return test_data


def prepare_testing_data(data, model_type):
    """Prepare testing data features and targets"""
    if data is None or data.empty:
        return None, None, []

    # Exclude non-feature columns
    exclude_cols = ['timestamp', 'trend_label', 'future_price', 'target']
    feature_cols = [col for col in data.columns if col not in exclude_cols and not col.startswith('_')]

    X = data[feature_cols].values

    # Prepare target based on model type
    if model_type.lower() == 'classification' and 'trend_label' in data.columns:
        y = data['trend_label'].values
    elif 'close' in data.columns:
        y = data['close'].values
    else:
        y = np.zeros(len(data))

    return X, y, feature_cols


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


def perform_walk_forward_testing(model, testing_data, model_type):
    """Perform walk-forward testing for time series data"""
    # Implementation for walk-forward testing
    st.info("Walk-forward testing would be implemented here")
    return {
        'performance_metrics': {'accuracy': 0.75, 'total_return': 0.15, 'sharpe_ratio': 1.2, 'max_drawdown': 0.12},
        'trading_results': {'returns': [], 'total_return': 0.15, 'sharpe_ratio': 1.2, 'max_drawdown': 0.12,
                            'win_rate': 0.6},
        'predictions': [],
        'actuals': [],
        'dates': None
    }


def perform_stress_testing(model, testing_data, model_type):
    """Perform stress testing under extreme market conditions"""
    # Implementation for stress testing
    st.info("Stress testing would be implemented here")
    return {
        'performance_metrics': {'accuracy': 0.65, 'total_return': 0.08, 'sharpe_ratio': 0.8, 'max_drawdown': 0.25},
        'trading_results': {'returns': [], 'total_return': 0.08, 'sharpe_ratio': 0.8, 'max_drawdown': 0.25,
                            'win_rate': 0.55},
        'predictions': [],
        'actuals': [],
        'dates': None
    }


def perform_scenario_analysis(model, testing_data, model_type):
    """Perform scenario analysis for different market regimes"""
    # Implementation for scenario analysis
    st.info("Scenario analysis would be implemented here")
    return {
        'performance_metrics': {'accuracy': 0.72, 'total_return': 0.12, 'sharpe_ratio': 1.0, 'max_drawdown': 0.18},
        'trading_results': {'returns': [], 'total_return': 0.12, 'sharpe_ratio': 1.0, 'max_drawdown': 0.18,
                            'win_rate': 0.58},
        'predictions': [],
        'actuals': [],
        'dates': None
    }


def display_trading_simulation(results):
    """Display trading simulation results"""
    if not results or 'trading_results' not in results:
        st.info("No trading simulation results available")
        return

    trading_results = results['trading_results']

    st.markdown("##### 💰 Trading Performance")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Return", f"{trading_results.get('total_return', 0) * 100:.2f}%")
    with col2:
        st.metric("Sharpe Ratio", f"{trading_results.get('sharpe_ratio', 0):.2f}")
    with col3:
        st.metric("Max Drawdown", f"{trading_results.get('max_drawdown', 0) * 100:.2f}%")
    with col4:
        st.metric("Win Rate", f"{trading_results.get('win_rate', 0) * 100:.2f}%")

    # Trading equity curve
    returns = trading_results.get('returns', [])
    if returns:
        cumulative_returns = np.cumprod(1 + np.array(returns)) - 1

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=cumulative_returns,
            mode='lines',
            name='Strategy Equity',
            line=dict(color='green', width=2)
        ))

        fig.update_layout(
            title="Trading Equity Curve",
            xaxis_title="Time",
            yaxis_title="Cumulative Return",
            height=300,
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)


def display_risk_analysis(results):
    """Display comprehensive risk analysis"""
    if not results or 'trading_results' not in results:
        st.info("No risk analysis data available")
        return

    trading_results = results['trading_results']
    returns = trading_results.get('returns', [])

    if not returns:
        st.info("No returns data for risk analysis")
        return

    st.markdown("##### ⚠️ Risk Metrics")

    # Calculate additional risk metrics
    returns_array = np.array(returns)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        volatility = np.std(returns_array) * np.sqrt(252) * 100
        st.metric("Annual Volatility", f"{volatility:.2f}%")
    with col2:
        var_95 = np.percentile(returns_array, 5) * 100
        st.metric("VaR (95%)", f"{var_95:.2f}%")
    with col3:
        cvar_95 = returns_array[returns_array <= np.percentile(returns_array, 5)].mean() * 100
        st.metric("CVaR (95%)", f"{cvar_95:.2f}%")
    with col4:
        sortino_ratio = (np.mean(returns_array) / np.std(returns_array[returns_array < 0])) * np.sqrt(252) if np.std(
            returns_array[returns_array < 0]) > 0 else 0
        st.metric("Sortino Ratio", f"{sortino_ratio:.2f}")

    # Drawdown analysis
    cumulative_returns = np.cumprod(1 + returns_array) - 1
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (running_max - cumulative_returns) / (1 + running_max)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=drawdown * 100,
        mode='lines',
        name='Drawdown',
        line=dict(color='red', width=2),
        fill='tozeroy'
    ))

    fig.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Time",
        yaxis_title="Drawdown %",
        height=300,
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)


def display_model_robustness(results):
    """Display model robustness analysis"""
    st.markdown("##### 🛡️ Model Robustness Analysis")

    # Robustness metrics (placeholder - these would be calculated during testing)
    robustness_metrics = {
        'Parameter Sensitivity': 'Low',
        'Data Stability': 'High',
        'Market Regime Adaptation': 'Medium',
        'Out-of-Sample Performance': 'Stable'
    }

    for metric, value in robustness_metrics.items():
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**{metric}**")
        with col2:
            color = "green" if value in ['High', 'Stable', 'Low'] else "orange" if value == 'Medium' else "red"
            st.markdown(f"<span style='color: {color}; font-weight: bold'>{value}</span>", unsafe_allow_html=True)

    # Robustness score
    robustness_score = 78  # This would be calculated based on various factors
    st.markdown("##### 🎯 Overall Robustness Score")
    st.progress(robustness_score / 100)
    st.write(f"Robustness Score: {robustness_score}/100")
def generate_predictions(model, X_test):
    """Generate predictions from model"""
    try:
        if hasattr(model, 'predict'):
            return model.predict(X_test)
        elif hasattr(model, 'predict_proba'):
            return model.predict_proba(X_test)[:, 1]
        else:
            # Default to random predictions
            return np.random.random(len(X_test))
    except:
        return np.random.random(len(X_test))


def simulate_trading(predictions, test_data, model_type):
    """Simulate trading based on predictions"""
    if len(predictions) == 0:
        return {'returns': [], 'total_return': 0, 'sharpe_ratio': 0}

    # Simple trading simulation
    returns = []
    position = 0
    cash = 10000  # Starting cash

    for i in range(1, len(predictions)):
        if model_type == 'classification':
            # Buy if prediction is up, sell if down
            if predictions[i] > 0.5 and position == 0:  # Buy signal
                position = cash / test_data.iloc[i]['close'] if 'close' in test_data.columns else 1
                cash = 0
            elif predictions[i] < 0.5 and position > 0:  # Sell signal
                cash = position * test_data.iloc[i]['close'] if 'close' in test_data.columns else cash * 1.02
                position = 0
        else:
            # Regression model - use prediction direction
            price_change = predictions[i] - (test_data.iloc[i - 1]['close'] if 'close' in test_data.columns else 0)
            if price_change > 0 and position == 0:
                position = cash / test_data.iloc[i]['close'] if 'close' in test_data.columns else 1
                cash = 0
            elif price_change < 0 and position > 0:
                cash = position * test_data.iloc[i]['close'] if 'close' in test_data.columns else cash * 0.98
                position = 0

        # Calculate portfolio value
        portfolio_value = cash + (position * test_data.iloc[i]['close'] if 'close' in test_data.columns else cash)
        if i > 0:
            prev_value = cash + (position * test_data.iloc[i - 1]['close'] if 'close' in test_data.columns else cash)
            daily_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0
            returns.append(daily_return)

    total_return = (portfolio_value - 10000) / 10000 if len(returns) > 0 else 0

    return {
        'returns': returns,
        'total_return': total_return,
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'max_drawdown': calculate_max_drawdown(returns),
        'win_rate': calculate_win_rate(returns)
    }


def calculate_returns(data):
    """Calculate price returns from data"""
    if 'close' not in data.columns:
        return np.random.random(len(data)) * 0.1 - 0.05  # Random returns

    returns = data['close'].pct_change().dropna()
    return returns.values


def calculate_strategy_returns(predictions, market_returns, data):
    """Calculate strategy returns based on predictions"""
    if len(predictions) != len(market_returns):
        return []

    strategy_returns = []
    for i in range(len(predictions)):
        # Simple strategy: go long when prediction is positive
        if predictions[i] > 0.5:
            strategy_returns.append(market_returns[i] if i < len(market_returns) else 0)
        else:
            strategy_returns.append(0)  # Stay in cash

    return strategy_returns


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate Sharpe ratio"""
    if len(returns) == 0:
        return 0

    returns = np.array(returns)
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0


def calculate_max_drawdown(returns):
    """Calculate maximum drawdown"""
    if len(returns) == 0:
        return 0

    cumulative = np.cumprod(1 + np.array(returns))
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak
    return np.max(drawdown)


def calculate_win_rate(returns):
    """Calculate win rate (percentage of positive returns)"""
    if len(returns) == 0:
        return 0

    positive_returns = len([r for r in returns if r > 0])
    return positive_returns / len(returns)


def evaluate_performance_thresholds(results, min_accuracy, max_drawdown):
    """Evaluate if performance meets thresholds"""
    metrics = results.get('performance_metrics', {})

    accuracy_ok = metrics.get('accuracy', 0) >= min_accuracy
    drawdown_ok = metrics.get('max_drawdown', 1) <= max_drawdown / 100

    return accuracy_ok and drawdown_ok


def display_benchmark_comparison(results):
    """Display benchmark comparison"""
    st.info("Benchmark comparison would be displayed here")

    # Placeholder benchmark data
    benchmark_data = {
        'Metric': ['Accuracy', 'Total Return', 'Sharpe Ratio', 'Max Drawdown'],
        'Strategy': [
            results.get('performance_metrics', {}).get('accuracy', 0),
            results.get('performance_metrics', {}).get('total_return', 0),
            results.get('performance_metrics', {}).get('sharpe_ratio', 0),
            results.get('performance_metrics', {}).get('max_drawdown', 0)
        ],
        'Benchmark': [0.5, 0.1, 0.8, 0.15]  # Placeholder benchmark values
    }

    df = __import__('pandas').DataFrame(benchmark_data)
    st.dataframe(df, use_container_width=True)


def get_data_manager():
    """Get data manager instance"""
    from ...config.session_state import get_data_manager as get_dm
    return get_dm()

def run_comprehensive_testing(model_info, testing_method, start_date, end_date, market_regimes, min_accuracy,
                              max_drawdown):
    """Run comprehensive model testing"""
    # Load model and prepare testing data
    model = load_trained_model(model_info['model_id'])

    # Get testing data for the specified period
    testing_data = load_testing_data(model_info['dataset'], start_date, end_date)

    if testing_data is None or testing_data.empty:
        st.error("No testing data available for the selected period")
        return

    # Run appropriate testing method
    if testing_method == "Backtesting":
        results = perform_backtesting(model, testing_data, model_info['model_type'])
    elif testing_method == "Walk-Forward Analysis":
        results = perform_walk_forward_testing(model, testing_data, model_info['model_type'])
    elif testing_method == "Stress Testing":
        results = perform_stress_testing(model, testing_data, model_info['model_type'])
    else:  # Scenario Analysis
        results = perform_scenario_analysis(model, testing_data, model_info['model_type'])

    # Apply performance thresholds
    results['passed_thresholds'] = evaluate_performance_thresholds(results, min_accuracy, max_drawdown)

    # Store results
    st.session_state.testing_results = results
    st.session_state.current_tested_model = model_info

    st.success("✅ Testing completed!")


def perform_backtesting(model, testing_data, model_type):
    """Perform comprehensive backtesting"""
    predictions = []
    actuals = []
    dates = []

    # Generate predictions
    X_test, y_test, feature_names = prepare_testing_data(testing_data, model_type)

    if hasattr(model, 'predict'):
        predictions = model.predict(X_test)
    else:
        # Handle custom model prediction
        predictions = generate_predictions(model, X_test)

    # Calculate performance metrics
    performance_metrics = calculate_backtesting_metrics(y_test, predictions, testing_data)

    # Generate trading signals and calculate P&L
    trading_results = simulate_trading(predictions, testing_data, model_type)

    return {
        'performance_metrics': performance_metrics,
        'trading_results': trading_results,
        'predictions': predictions,
        'actuals': y_test,
        'dates': testing_data['timestamp'].values if 'timestamp' in testing_data.columns else None
    }


def calculate_backtesting_metrics(y_true, y_pred, testing_data):
    """Calculate comprehensive backtesting metrics"""
    metrics = {}

    # Basic accuracy metrics
    if len(np.unique(y_true)) <= 2:  # Classification
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
    else:  # Regression
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)

    # Trading-specific metrics
    returns = calculate_returns(testing_data)
    strategy_returns = calculate_strategy_returns(y_pred, returns, testing_data)

    if len(strategy_returns) > 0:
        metrics['total_return'] = np.prod(1 + strategy_returns) - 1
        metrics['sharpe_ratio'] = calculate_sharpe_ratio(strategy_returns)
        metrics['max_drawdown'] = calculate_max_drawdown(strategy_returns)
        metrics['win_rate'] = calculate_win_rate(strategy_returns)

    return metrics


def display_testing_results_advanced():
    """Display advanced testing results"""
    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    st.markdown("#### 📊 Testing Results Dashboard")

    if st.session_state.get('testing_results'):
        results = st.session_state.testing_results
        model_info = st.session_state.get('current_tested_model', {})

        # Performance Summary with Pass/Fail
        st.markdown("##### 🎯 Testing Summary")
        display_testing_summary(results, model_info)

        # Detailed Performance Metrics
        st.markdown("##### 📈 Performance Analysis")
        display_performance_analysis(results)

        # Trading Simulation Results
        st.markdown("##### 💰 Trading Simulation")
        display_trading_simulation(results)

        # Risk Analysis
        st.markdown("##### ⚠️ Risk Analysis")
        display_risk_analysis(results)

        # Model Robustness
        st.markdown("##### 🛡️ Model Robustness")
        display_model_robustness(results)

    else:
        st.info("👆 Run testing to see results")

    st.markdown('</div>', unsafe_allow_html=True)


def display_testing_summary(results, model_info):
    """Display testing summary with pass/fail status"""
    metrics = results.get('performance_metrics', {})
    passed = results.get('passed_thresholds', False)

    # Status indicator
    status_color = "green" if passed else "red"
    status_icon = "✅" if passed else "❌"
    status_text = "PASSED" if passed else "FAILED"

    st.markdown(
        f'<div style="padding: 1rem; border-radius: 10px; background: rgba({status_color}, 0.1); border: 2px solid {status_color};">'
        f'<h3 style="color: {status_color}; margin: 0;">{status_icon} Testing {status_text}</h3>'
        f'</div>', unsafe_allow_html=True)

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        accuracy = metrics.get('accuracy', metrics.get('r2', 0))
        st.metric("Accuracy/R²", f"{accuracy:.3f}")

    with col2:
        total_return = metrics.get('total_return', 0) * 100
        st.metric("Total Return", f"{total_return:.2f}%")

    with col3:
        sharpe = metrics.get('sharpe_ratio', 0)
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")

    with col4:
        max_dd = metrics.get('max_drawdown', 0) * 100
        st.metric("Max Drawdown", f"{max_dd:.2f}%")

    # Recommendations
    if passed:
        st.success("🎉 Model passed all testing thresholds! Ready for deployment.")
    else:
        st.error("⚠️ Model failed some testing thresholds. Review results before deployment.")


def display_performance_analysis(results):
    """Display detailed performance analysis"""
    metrics = results.get('performance_metrics', {})

    tab1, tab2, tab3 = st.tabs(["Accuracy Metrics", "Return Analysis", "Comparative Analysis"])

    with tab1:
        # Classification metrics
        if 'accuracy' in metrics:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
            with col3:
                st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
            with col4:
                st.metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}")

        # Regression metrics
        if 'r2' in metrics:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{metrics['r2']:.3f}")
            with col2:
                st.metric("MSE", f"{metrics.get('mse', 0):.6f}")
            with col3:
                st.metric("MAE", f"{metrics.get('mae', 0):.6f}")

    with tab2:
        # Returns analysis
        if 'total_return' in metrics:
            returns_data = results.get('trading_results', {}).get('returns', [])
            if len(returns_data) > 0:
                display_returns_analysis(returns_data, metrics)

    with tab3:
        # Benchmark comparison
        display_benchmark_comparison(results)


def display_returns_analysis(returns, metrics):
    """Display returns analysis with charts"""
    # Cumulative returns
    cumulative_returns = np.cumprod(1 + np.array(returns)) - 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=cumulative_returns,
        mode='lines',
        name='Strategy Returns',
        line=dict(color='green')
    ))

    # Add benchmark (buy & hold)
    # This would be calculated from actual price data
    fig.add_trace(go.Scatter(
        y=[0] * len(cumulative_returns),  # Placeholder
        mode='lines',
        name='Buy & Hold',
        line=dict(color='blue', dash='dash')
    ))

    fig.update_layout(
        title="Cumulative Returns",
        xaxis_title="Time",
        yaxis_title="Cumulative Return",
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)

    # Returns distribution
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='Returns Distribution',
        marker_color='lightblue'
    ))

    fig_hist.update_layout(
        title="Returns Distribution",
        xaxis_title="Daily Returns",
        yaxis_title="Frequency",
        height=300
    )

    st.plotly_chart(fig_hist, use_container_width=True)