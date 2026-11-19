"""
Student Performance Prediction and Learning Pattern Analysis
Using Machine Learning with Applied Mathematics.

Main Streamlit Application Entrypoint.
Comprehensive M.Tech Academic Project covering CO-1 through CO-6.
Includes Authentication (Home, Register, Login with Hide/Unhide Password toggle).
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.utils.config import (
    RAW_DATA_PATH,
    REGRESSION_MODEL_PATH,
    CLASSIFICATION_MODEL_PATH,
    CATEGORY_LABELS,
    FEATURE_NAMES,
)
from src.data.loader import prepare_dataset
from src.utils.auth_views import render_login_page, render_register_page

# Streamlit Page Configuration
st.set_page_config(
    page_title="Student Performance AI/ML Lab",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Academic Theme CSS
st.markdown(
    """
    <style>
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .user-pill {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 12px;
        font-size: 0.85rem;
        color: #1E40AF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session Authentication State
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None

# Ensure baseline data exists
if not RAW_DATA_PATH.exists():
    with st.spinner("Generating initial reproducible academic dataset..."):
        prepare_dataset(n_samples=1500, seed=42)

st.sidebar.title("🎓 Navigation Portal")
st.sidebar.markdown("**M.Tech Academic Project**")
st.sidebar.markdown("*Machine Learning with Applied Mathematics*")

# Sidebar User Status
if st.session_state["authenticated"] and st.session_state["user"]:
    user_info = st.session_state["user"]
    st.sidebar.markdown(
        f"""
        <div class="user-pill">
            👤 <b>{user_info['name']}</b><br>
            <span style="font-size:0.75rem; color:#475569;">{user_info['email']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["user"] = None
        st.rerun()

st.sidebar.markdown("---")

# Navigation Options Based on Auth State
if not st.session_state["authenticated"]:
    NAVIGATION_PAGES = {
        "🏠 Home Page": "pages.1_Home_Dashboard",
        "🔑 Login Page": "AUTH_LOGIN",
        "📝 Register Page": "AUTH_REGISTER",
    }
    selected_page = st.sidebar.radio("Authentication & Public Pages:", list(NAVIGATION_PAGES.keys()))
else:
    NAVIGATION_PAGES = {
        "🏠 Home Dashboard": "pages.1_Home_Dashboard",
        "1. Student Performance Prediction": "pages.2_Student_Performance_Prediction",
        "2. Dataset Explorer": "pages.3_Dataset_Explorer",
        "3. Regression Lab (CO-2)": "pages.4_Regression_Lab",
        "4. Classification Lab (CO-2)": "pages.5_Classification_Lab",
        "5. Student Clustering Lab (CO-3)": "pages.6_Student_Clustering_Lab",
        "6. PCA, t-SNE & UMAP (CO-3)": "pages.7_PCA_tSNE_UMAP_Visualization",
        "7. Anomaly Detection Lab (CO-3)": "pages.8_Anomaly_Detection_Lab",
        "8. Applied Mathematics Lab (CO-1)": "pages.9_Applied_Mathematics_Lab",
        "9. Optimization & Training Lab (CO-4)": "pages.10_Optimization_and_Training_Lab",
        "10. Model Evaluation Lab (CO-5)": "pages.11_Model_Evaluation_Lab",
        "11. Fairness & Statistical Testing (CO-5)": "pages.12_Fairness_and_Statistical_Testing_Lab",
        "12. MLOps & Model Monitoring (CO-6)": "pages.13_MLOps_and_Model_Monitoring",
        "13. About Project & CO Mapping": "pages.14_About_Project_and_CO_Mapping",
    }
    selected_page = st.sidebar.radio("Academic Laboratory Modules:", list(NAVIGATION_PAGES.keys()))

st.sidebar.markdown("---")
st.sidebar.caption("System Status: **Active** | Python 3.11+ | PyTorch | Scikit-Learn | FastAPI")

# Page Routing
page_target = NAVIGATION_PAGES[selected_page]

if page_target == "AUTH_LOGIN":
    render_login_page()
elif page_target == "AUTH_REGISTER":
    render_register_page()
else:
    import importlib
    try:
        page_module = importlib.import_module(page_target)
        page_module.render_page()
    except Exception as e:
        st.error(f"Error loading module {selected_page}: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
