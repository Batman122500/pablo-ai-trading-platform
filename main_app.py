import sys
import os
import streamlit as st

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# Now use absolute imports
from pablo.ui_base.config.session_state import initialize_session_state, cleanup_on_navigation
from pablo.ui_base.config.performance import performance_monitor
from pablo.ui_base.config.theme import apply_custom_theme

from pablo.ui_base.components.sidebar import render_sidebar
from pablo.ui_base.components.header import show_connection_status
from pablo.ui_base.components.task_monitor import show_task_monitor

from pablo.ui_base.pages.dashboard import show_dashboard
from pablo.ui_base.pages.data_management.historical_data import show_historical_data_tab
from pablo.ui_base.pages.data_management.pair_management import show_trading_pair_management_tab
from pablo.ui_base.pages.data_management.preprocessing import show_data_preprocessing_tab
from pablo.ui_base.pages.data_management.preprocessed_data import show_preprocessed_data_management
from pablo.ui_base.pages.model_deployment import show_model_deployment
from pablo.ui_base.pages.model_engineering import show_model_engineering

def safe_page_render(page_func, page_name):
    """Safely render pages with error handling."""
    try:
        page_func()
    except Exception as e:
        st.error(f"❌ Error loading {page_name}: {str(e)}")
        st.info("🔄 Try refreshing the page or navigating away and back.")


@performance_monitor
def main():
    """Main application entry point."""

    # Apply theme first
    apply_custom_theme()

    # Initialize session state - 🔥 CRITICAL: This must happen first
    initialize_session_state()

    # Global task monitor
    show_task_monitor()

    # Sidebar navigation
    page = render_sidebar()

    # Handle cleanup on navigation
    cleanup_on_navigation()

    # Dynamic page header
    st.markdown(f'<div class="section-header">🚀 {page}</div>', unsafe_allow_html=True)

    # Online/offline indicator
    is_online = show_connection_status()

    # Main routing
    if page == "Main Dashboard":
        safe_page_render(show_dashboard, "Dashboard")

    elif page == "Data Management":
        safe_page_render(show_enhanced_data_management, "Data Management")

    elif page == "Model Engineering":
        # 🔥 CRITICAL: Re-initialize for model engineering
        initialize_session_state()
        safe_page_render(show_model_engineering, "Model Engineering")

    elif page == "Model Deployment":
        safe_page_render(show_model_deployment, "Model Deployment")

    elif page == "Trading Monitor":
        if not is_online:
            st.error("🔴 Trading monitor requires internet connection.")
        else:
            st.info("📡 Trading monitor coming soon!")

    elif page == "Analytics":
        st.info("📊 Analytics dashboard coming soon!")

    elif page == "Configuration":
        st.info("⚙️ Configuration panel coming soon!")


def show_enhanced_data_management():
    """Data Management center with all sub-sections."""
    st.markdown('<div class="section-header">📊 Data Management Center</div>', unsafe_allow_html=True)

    tab_options = [
        "📈 Historical Data",
        "🔄 Trading Pair Management",
        "🧹 Data Preprocessing",
        "💾 Preprocessed Data"
    ]

    selected_tab = st.selectbox(
        "Choose Section:",
        tab_options,
        key="data_management_tabs",
        label_visibility="collapsed"
    )

    if selected_tab == "📈 Historical Data":
        show_historical_data_tab()

    elif selected_tab == "🔄 Trading Pair Management":
        show_trading_pair_management_tab()

    elif selected_tab == "🧹 Data Preprocessing":
        show_data_preprocessing_tab()

    elif selected_tab == "💾 Preprocessed Data":
        show_preprocessed_data_management()

