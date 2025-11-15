import streamlit as st


def apply_custom_theme():
    """Apply a modern gradient theme with natural green and light blue colors"""
    st.markdown("""
    <style>
        /* Modern Natural Gradient Theme */
        :root {
            --primary-dark: #0a1929;
            --secondary-dark: #102a43;
            --accent-blue: #2d87c8;
            --accent-teal: #38b2ac;
            --accent-green: #48bb78;
            --accent-emerald: #0d9488;
            --gradient-primary: linear-gradient(135deg, #0d9488 0%, #2d87c8 100%);
            --gradient-secondary: linear-gradient(135deg, #38b2ac 0%, #48bb78 100%);
            --gradient-background: linear-gradient(135deg, #0a1929 0%, #1e3a5f 50%, #2d5a80 100%);
            --gradient-card: linear-gradient(135deg, rgba(16, 42, 67, 0.9) 0%, rgba(45, 135, 200, 0.1) 100%);
            --gradient-success: linear-gradient(135deg, #48bb78 0%, #38b2ac 100%);
            --text-primary: #f7fafc;
            --text-secondary: #e2e8f0;
            --text-accent: #90cdf4;
            --success: #48bb78;
            --warning: #ed8936;
            --danger: #f56565;
            --shadow-light: 0 4px 6px rgba(13, 148, 136, 0.1);
            --shadow-medium: 0 10px 15px rgba(13, 148, 136, 0.1);
            --shadow-heavy: 0 20px 25px rgba(13, 148, 136, 0.1);
        }

        /* Main background with animated gradient */
        .main {
            background: var(--gradient-background);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            color: var(--text-primary);
            min-height: 100vh;
        }

        @keyframes gradientShift {
            0% { background-position: 0% 50% }
            50% { background-position: 100% 50% }
            100% { background-position: 0% 50% }
        }

        .stApp {
            background: var(--gradient-background);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            background-attachment: fixed;
        }

        /* Enhanced Headers with glowing effects */
        .main-header {
            font-size: 3rem;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 2px;
            text-shadow: 0 4px 8px rgba(13, 148, 136, 0.3);
            animation: glow 2s ease-in-out infinite alternate;
        }

        @keyframes glow {
            from { text-shadow: 0 0 10px rgba(13, 148, 136, 0.5), 0 0 20px rgba(13, 148, 136, 0.3); }
            to { text-shadow: 0 0 15px rgba(13, 148, 136, 0.8), 0 0 30px rgba(13, 148, 136, 0.5); }
        }

        .section-header {
            font-size: 1.8rem;
            background: var(--gradient-secondary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            font-weight: 700;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid;
            border-image: var(--gradient-primary) 1;
            position: relative;
            overflow: hidden;
        }

        .section-header::after {
            content: '';
            position: absolute;
            bottom: -3px;
            left: 0;
            width: 100%;
            height: 3px;
            background: var(--gradient-primary);
            transform: scaleX(0);
            transform-origin: left;
            transition: transform 0.3s ease;
        }

        .section-header:hover::after {
            transform: scaleX(1);
        }

        .subsection-header {
            font-size: 1.4rem;
            color: var(--accent-teal);
            margin-top: 1rem;
            margin-bottom: 0.8rem;
            font-weight: 600;
            padding-left: 0.5rem;
            border-left: 4px solid var(--accent-teal);
            transition: all 0.3s ease;
            position: relative;
        }

        .subsection-header:hover {
            color: var(--accent-green);
            border-left-color: var(--accent-green);
            transform: translateX(5px);
        }

        /* Enhanced Cards with glassmorphism and hover effects */
        .data-card {
            background: var(--gradient-card);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
            border: 1px solid rgba(45, 135, 200, 0.2);
            box-shadow: var(--shadow-medium);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .data-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(56, 178, 172, 0.1), transparent);
            transition: left 0.5s ease;
        }

        .data-card:hover::before {
            left: 100%;
        }

        .data-card:hover {
            transform: translateY(-8px);
            box-shadow: var(--shadow-heavy);
            border-color: rgba(56, 178, 172, 0.4);
        }

        .metric-card {
            background: var(--gradient-primary);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 0.5rem;
            text-align: center;
            color: white;
            box-shadow: var(--shadow-light);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .metric-card::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
            transform: rotate(45deg);
            transition: all 0.6s ease;
        }

        .metric-card:hover::before {
            transform: rotate(45deg) translate(50%, 50%);
        }

        .metric-card:hover {
            transform: scale(1.05);
            box-shadow: var(--shadow-medium);
        }

        /* SCROLLABLE PAIRS CONTAINER */
        .scrollable-pairs-container {
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid rgba(56, 178, 172, 0.3);
            border-radius: 10px;
            padding: 15px;
            background: rgba(16, 42, 67, 0.6);
            margin: 10px 0;
        }

        .pair-item {
            padding: 8px 12px;
            background: rgba(30, 58, 95, 0.6);
            border-radius: 6px;
            border: 1px solid rgba(56, 178, 172, 0.2);
            text-align: center;
            transition: all 0.3s ease;
            font-family: 'Courier New', monospace;
            cursor: pointer;
        }

        .pair-item:hover {
            background: rgba(56, 178, 172, 0.2);
            border-color: rgba(56, 178, 172, 0.5);
            transform: translateY(-2px);
        }

        /* CONSISTENT BUTTON DIMENSIONS */
        .stButton button {
            min-height: 38px !important;
            height: 38px !important;
            font-size: 14px !important;
            padding: 8px 16px !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }

        .stButton button:hover {
            transform: translateY(-2px) !important;
            box-shadow: var(--shadow-medium) !important;
        }

        /* Grid layout for pairs */
        .pairs-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 8px;
            padding: 10px;
        }

        .selected-pair {
            background: var(--gradient-primary);
            border-color: var(--accent-blue);
        }

        /* Enhanced Status indicators */
        .status-online {
            color: var(--accent-green);
            font-weight: bold;
            text-shadow: 0 0 10px rgba(72, 187, 120, 0.5);
            animation: pulse 2s infinite;
        }

        .status-offline {
            color: var(--danger);
            font-weight: bold;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }

        .connection-status {
            padding: 1rem 2rem;
            border-radius: 25px;
            margin-bottom: 1rem;
            text-align: center;
            font-weight: bold;
            background: var(--gradient-success);
            border: none;
            color: white;
            box-shadow: var(--shadow-light);
            transition: all 0.3s ease;
        }

        .connection-status:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-medium);
        }

        /* Enhanced Technical indicators */
        .indicator-positive {
            color: var(--accent-green);
            font-weight: bold;
            text-shadow: 0 0 8px rgba(72, 187, 120, 0.3);
        }

        .indicator-negative {
            color: var(--danger);
            font-weight: bold;
        }

        .indicator-neutral {
            color: var(--warning);
            font-weight: bold;
        }

        /* Enhanced Streamlit component overrides */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: rgba(16, 42, 67, 0.8);
            padding: 0.5rem;
            border-radius: 16px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(56, 178, 172, 0.2);
        }

        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            color: var(--text-secondary) !important;
            border-radius: 12px !important;
            padding: 0.8rem 1.5rem !important;
            border: 1px solid rgba(56, 178, 172, 0.2) !important;
            height: auto !important;
            transition: all 0.3s ease !important;
        }

        .stTabs [aria-selected="true"] {
            background: var(--gradient-primary) !important;
            color: white !important;
            border-color: transparent !important;
            box-shadow: var(--shadow-light) !important;
            transform: scale(1.05);
        }

        /* Enhanced Button styling */
        .stButton button {
            background: var(--gradient-primary) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.8rem 1.8rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: var(--shadow-light) !important;
            position: relative;
            overflow: hidden;
        }

        .stButton button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s ease;
        }

        .stButton button:hover::before {
            left: 100%;
        }

        .stButton button:hover {
            transform: translateY(-3px) !important;
            box-shadow: var(--shadow-medium) !important;
        }

        /* Enhanced Selectbox and input */
        .stSelectbox div div {
            background: rgba(16, 42, 67, 0.8) !important;
            color: var(--text-primary) !important;
            border: 1px solid rgba(56, 178, 172, 0.3) !important;
            border-radius: 10px !important;
            transition: all 0.3s ease !important;
        }

        .stSelectbox div div:hover {
            border-color: var(--accent-teal) !important;
            box-shadow: 0 0 10px rgba(56, 178, 172, 0.2) !important;
        }

        .stTextInput input {
            background: rgba(16, 42, 67, 0.8) !important;
            color: var(--text-primary) !important;
            border: 1px solid rgba(56, 178, 172, 0.3) !important;
            border-radius: 10px !important;
            transition: all 0.3s ease !important;
        }

        .stTextInput input:focus {
            border-color: var(--accent-teal) !important;
            box-shadow: 0 0 15px rgba(56, 178, 172, 0.3) !important;
        }

        /* Enhanced Dataframe styling */
        .dataframe {
            background: rgba(16, 42, 67, 0.8) !important;
            color: var(--text-primary) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(56, 178, 172, 0.2) !important;
        }

        /* Enhanced Metric styling */
        [data-testid="metric-container"] {
            background: rgba(16, 42, 67, 0.8) !important;
            border: 1px solid rgba(56, 178, 172, 0.2) !important;
            border-radius: 16px !important;
            padding: 1.5rem !important;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }

        [data-testid="metric-container"]:hover {
            border-color: var(--accent-teal) !important;
            transform: translateY(-2px);
            box-shadow: var(--shadow-light);
        }

        /* Better vertical spacing */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: var(--secondary-dark);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb {
            background: var(--gradient-primary);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent-teal);
        }

        /* Loading animation */
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .loading {
            animation: pulse 2s infinite;
        }

        /* Badge styles */
        .badge {
            display: inline-block;
            padding: 0.3rem 0.7rem;
            border-radius: 15px;
            font-size: 0.75rem;
            font-weight: 600;
            margin: 0 0.2rem;
            background: var(--gradient-secondary);
            color: white;
            box-shadow: var(--shadow-light);
        }

        .badge-success {
            background: var(--gradient-success);
        }

        .badge-warning {
            background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        }

        .badge-info {
            background: var(--gradient-primary);
        }

        /* Progress bar styling */
        .stProgress > div > div > div > div {
            background: var(--gradient-success);
            border-radius: 10px;
        }

        /* Floating particles background effect */
        .particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
        }

        .particle {
            position: absolute;
            width: 4px;
            height: 4px;
            background: var(--accent-teal);
            border-radius: 50%;
            animation: float 6s infinite ease-in-out;
        }

        @keyframes float {
            0%, 100% { transform: translateY(0) rotate(0deg); opacity: 0; }
            50% { transform: translateY(-20px) rotate(180deg); opacity: 0.5; }
        }

        /* Task monitor enhancements */
        .task-item {
            background: rgba(16, 42, 67, 0.8);
            border: 1px solid rgba(56, 178, 172, 0.3);
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
            transition: all 0.3s ease;
        }

        .task-item:hover {
            border-color: var(--accent-teal);
            transform: translateX(5px);
        }

        /* Chart container enhancements */
        .js-plotly-plot .plotly {
            border-radius: 16px;
            border: 1px solid rgba(56, 178, 172, 0.2);
        }

        /* Sidebar enhancements */
        .css-1d391kg {
            background: var(--gradient-background) !important;
        }

        /* Expander enhancements */
        .streamlit-expanderHeader {
            background: rgba(16, 42, 67, 0.8) !important;
            border: 1px solid rgba(56, 178, 172, 0.2) !important;
            border-radius: 10px !important;
            color: var(--text-primary) !important;
        }

        .streamlit-expanderHeader:hover {
            border-color: var(--accent-teal) !important;
        }

        /* Consistent button dimensions for View/Delete */
        .consistent-button {
            min-height: 38px !important;
            height: 38px !important;
            font-size: 14px !important;
            padding: 8px 16px !important;
            border-radius: 8px !important;
        }

        /* Tab content styling */
        .tab-content {
            padding: 1rem;
            background: rgba(16, 42, 67, 0.6);
            border-radius: 10px;
            margin: 0.5rem 0;
        }
    </style>

    <div class="particles" id="particles-js"></div>

    <script>
        // Create floating particles
        document.addEventListener('DOMContentLoaded', function() {
            const particlesContainer = document.getElementById('particles-js');
            const particleCount = 30;

            for (let i = 0; i < particleCount; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = Math.random() * 100 + 'vw';
                particle.style.top = Math.random() * 100 + 'vh';
                particle.style.animationDelay = Math.random() * 6 + 's';
                particle.style.animationDuration = (3 + Math.random() * 4) + 's';
                particlesContainer.appendChild(particle);
            }
        });
    </script>
    """, unsafe_allow_html=True)
