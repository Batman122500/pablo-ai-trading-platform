import streamlit as st
import pandas as pd
from ..config.performance import performance_monitor
from ..components.header import show_connection_status

@performance_monitor
def show_model_deployment():
    """Model deployment view"""
    st.markdown('<div class="section-header">🤖 Model Deployment</div>', unsafe_allow_html=True)
    is_online = show_connection_status()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.markdown("#### Model Performance")
        performance_data = {
            'Model': ['LSTM_v1', 'XGBoost_v2', 'Ensemble_v1', 'Classification_v1', 'DRL_v1'],
            'Accuracy': [0.78, 0.82, 0.85, 0.76, 0.71],
            'Precision': [0.75, 0.80, 0.83, 0.74, 0.69]
        }
        st.dataframe(pd.DataFrame(performance_data), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.markdown("#### Deployment Status")
        status_data = {
            'Environment': ['Production', 'Staging', 'Development'],
            'Status': ['🟢 Active', '🟡 Testing', '🔴 Inactive'],
            'Version': ['v1.2.3', 'v1.3.0', 'v2.0.0-beta']
        }
        st.dataframe(pd.DataFrame(status_data), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)