import streamlit as st
import socket
import requests

def check_internet_connection():
    """Check if the system is connected to the internet"""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        pass

    try:
        response = requests.get("https://www.google.com", timeout=5)
        return True
    except:
        return False

def show_connection_status():
    """Display internet connection status at the top of each section"""
    is_online = check_internet_connection()

    if is_online:
        st.markdown(
            '<div class="connection-status">'
            '🟢 ONLINE - Connected to Internet</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="connection-status" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);">'
            '🔴 OFFLINE - No Internet Connection</div>',
            unsafe_allow_html=True
        )

    return is_online