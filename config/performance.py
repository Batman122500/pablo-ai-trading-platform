import time
import logging
import functools
import streamlit as st

def performance_monitor(func):
    """Decorator to monitor function performance"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        if duration > 1.0:
            logging.info(f"⏱️ {func.__name__} took {duration:.2f} seconds")
        return result
    return wrapper

def cache_data(ttl=3600):
    """Cache decorator with TTL"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
            if cache_key in st.session_state:
                cached_data = st.session_state[cache_key]
                if time.time() - cached_data['timestamp'] < ttl:
                    return cached_data['data']

            result = func(*args, **kwargs)
            st.session_state[cache_key] = {
                'data': result,
                'timestamp': time.time()
            }
            return result
        return wrapper
    return decorator

def async_operation(operation_name):
    """Decorator for async operations with progress tracking"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            st.session_state.processing_active = True
            st.session_state.processing_progress = 0
            st.session_state.processing_status = f"Starting {operation_name}..."

            try:
                result = func(*args, **kwargs)
                st.session_state.processing_progress = 100
                st.session_state.processing_status = f"{operation_name} completed successfully!"
                time.sleep(1)
                return result
            except Exception as e:
                st.session_state.processing_status = f"Error in {operation_name}: {str(e)}"
                raise e
            finally:
                st.session_state.processing_active = False
        return wrapper
    return decorator

def lazy_load_component(component_func, *args, **kwargs):
    """Lazy load heavy components only when needed"""
    try:
        return component_func(*args, **kwargs)
    except Exception as e:
        st.error(f"Error loading component: {e}")
        return None