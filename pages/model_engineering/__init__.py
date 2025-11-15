# pages/model_engineering/__init__.py
import streamlit as st
from .model_training import show_model_training
from .model_validation import show_model_validation
from .model_testing import show_model_testing
from .model_optimization import show_model_optimization
from .adaptive_models import show_adaptive_models


def show_model_engineering():
    """Main Model Engineering interface with tabs"""
    st.markdown('<div class="section-header">🤖 Model Engineering Center</div>', unsafe_allow_html=True)

    # Tab selection
    tab_options = [
        "🧠 Model Training",
        "✅ Model Validation",
        "📊 Model Testing",
        "⚡ Model Optimization",
        "🔄 Adaptive Models"
    ]

    selected_tab = st.selectbox(
        "Choose Section:",
        tab_options,
        key="model_engineering_tabs",
        label_visibility="collapsed"
    )

    if selected_tab == "🧠 Model Training":
        show_model_training()
    elif selected_tab == "✅ Model Validation":
        show_model_validation()
    elif selected_tab == "📊 Model Testing":
        show_model_testing()
    elif selected_tab == "⚡ Model Optimization":
        show_model_optimization()
    elif selected_tab == "🔄 Adaptive Models":
        show_adaptive_models()