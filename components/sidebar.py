import streamlit as st
from ..config.session_state import get_unique_key
from .header import check_internet_connection


def render_sidebar():
    """Render the navigation sidebar"""
    with st.sidebar:
        st.markdown("## 🎮 Navigation")
        st.markdown("---")

        # Updated navigation with "Model Engineering"
        page = st.selectbox(
            "Go to:",
            [
                "Main Dashboard",
                "Data Management",
                "Model Engineering",
                "Model Deployment",
                "Trading Monitor",
                "Analytics",
                "Configuration"
            ],
            key="main_navigation",
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### 🔐 User Info")
        st.write(f"User: **pablo_trader**")

        # Check internet status
        is_online = check_internet_connection()
        status_icon = "🟢" if is_online else "🔴"
        status_text = "Online" if is_online else "Offline"
        st.write(f"Status: {status_icon} {status_text}")

        # Logout button
        if st.button("🚪 Logout", use_container_width=True, key=get_unique_key("logout_btn")):
            st.session_state.authenticated = False
            st.rerun()

        return page
