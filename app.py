"""
Student Performance Prediction and Learning Pattern Analysis.
Main Streamlit Application Entrypoint.
Clean interface with automated redirection:
- Initial startup -> Home
- Registration success -> Login
- Login success -> Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.utils.config import (
    RAW_DATA_PATH,
    REGRESSION_MODEL_PATH,
    CLASSIFICATION_MODEL_PATH,
    FEATURE_NAMES,
)
from src.data.loader import prepare_dataset
from src.utils.auth_views import render_login_page, render_register_page

# Streamlit Page Configuration
st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Clean Theme CSS
st.markdown(
    """
    <style>
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 2.1rem;
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

# Initialize Session State
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None
if "nav_selection" not in st.session_state:
    # Always default to Home on launch
    st.session_state["nav_selection"] = "Home"

# Ensure baseline data exists
if not RAW_DATA_PATH.exists():
    with st.spinner("Initializing system dataset..."):
        prepare_dataset(n_samples=1500, seed=42)

# Sidebar Branding
st.sidebar.title("Student Analytics AI")

# Sidebar User Status / Logout
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
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["user"] = None
        st.session_state["nav_selection"] = "Home"
        st.rerun()

st.sidebar.markdown("---")

# Navigation Menu Definition
if not st.session_state["authenticated"]:
    # Unauthenticated menu: Home, Login, Register
    nav_options = ["Home", "Login", "Register"]
    if st.session_state["nav_selection"] not in nav_options:
        st.session_state["nav_selection"] = "Home"

    current_idx = nav_options.index(st.session_state["nav_selection"])
    selected_page = st.sidebar.radio("Navigation:", nav_options, index=current_idx)
    st.session_state["nav_selection"] = selected_page
else:
    # Authenticated menu: Dashboard and analytical modules
    nav_map = {
        "Dashboard": "pages.1_Home_Dashboard",
        "Student Performance Prediction": "pages.2_Student_Performance_Prediction",
        "Dataset Explorer": "pages.3_Dataset_Explorer",
        "Regression Lab": "pages.4_Regression_Lab",
        "Classification Lab": "pages.5_Classification_Lab",
        "Student Clustering Lab": "pages.6_Student_Clustering_Lab",
        "PCA, t-SNE & UMAP": "pages.7_PCA_tSNE_UMAP_Visualization",
        "Anomaly Detection Lab": "pages.8_Anomaly_Detection_Lab",
        "Applied Mathematics Lab": "pages.9_Applied_Mathematics_Lab",
        "Optimization & Training Lab": "pages.10_Optimization_and_Training_Lab",
        "Model Evaluation Lab": "pages.11_Model_Evaluation_Lab",
        "Fairness & Statistical Testing": "pages.12_Fairness_and_Statistical_Testing_Lab",
        "Model Monitoring & Drift": "pages.13_MLOps_and_Model_Monitoring",
        "About System": "pages.14_About_Project_and_CO_Mapping",
    }
    nav_options = list(nav_map.keys())

    if st.session_state["nav_selection"] not in nav_options:
        st.session_state["nav_selection"] = "Dashboard"

    current_idx = nav_options.index(st.session_state["nav_selection"])
    selected_page = st.sidebar.radio("Analytics Modules:", nav_options, index=current_idx)
    st.session_state["nav_selection"] = selected_page

st.sidebar.markdown("---")

# Dynamic Page Rendering
if not st.session_state["authenticated"]:
    if st.session_state["nav_selection"] == "Login":
        render_login_page()
    elif st.session_state["nav_selection"] == "Register":
        render_register_page()
    else:
        # Home Page
        import pages.1_Home_Dashboard as home_module
        home_module.render_page()
else:
    target_module_name = nav_map[st.session_state["nav_selection"]]
    import importlib
    try:
        mod = importlib.import_module(target_module_name)
        mod.render_page()
    except Exception as e:
        st.error(f"Error loading {selected_page}: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
