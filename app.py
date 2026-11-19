"""
Student Performance Prediction and Learning Pattern Analysis
Using Machine Learning with Applied Mathematics.

Main Streamlit Application Entrypoint.
Comprehensive M.Tech Academic Project covering CO-1 through CO-6.
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
    .co-badge {
        display: inline-block;
        background: #EEF2FF;
        color: #4338CA;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Ensure baseline data exists
if not RAW_DATA_PATH.exists():
    with st.spinner("Generating initial reproducible academic dataset..."):
        prepare_dataset(n_samples=1500, seed=42)

st.sidebar.title("🎓 Navigation Portal")
st.sidebar.markdown("**M.Tech Academic Project**")
st.sidebar.markdown("*Applied Mathematics & ML (CO-1 to CO-6)*")

# Pages Dictionary
PAGES = {
    "1. Home Dashboard": "pages.1_Home_Dashboard",
    "2. Student Performance Prediction": "pages.2_Student_Performance_Prediction",
    "3. Dataset Explorer": "pages.3_Dataset_Explorer",
    "4. Regression Lab (CO-2)": "pages.4_Regression_Lab",
    "5. Classification Lab (CO-2)": "pages.5_Classification_Lab",
    "6. Student Clustering Lab (CO-3)": "pages.6_Student_Clustering_Lab",
    "7. PCA, t-SNE & UMAP (CO-3)": "pages.7_PCA_tSNE_UMAP_Visualization",
    "8. Anomaly Detection Lab (CO-3)": "pages.8_Anomaly_Detection_Lab",
    "9. Applied Mathematics Lab (CO-1)": "pages.9_Applied_Mathematics_Lab",
    "10. Optimization & Training Lab (CO-4)": "pages.10_Optimization_and_Training_Lab",
    "11. Model Evaluation Lab (CO-5)": "pages.11_Model_Evaluation_Lab",
    "12. Fairness & Statistical Testing (CO-5)": "pages.12_Fairness_and_Statistical_Testing_Lab",
    "13. MLOps & Model Monitoring (CO-6)": "pages.13_MLOps_and_Model_Monitoring",
    "14. About Project & CO Mapping": "pages.14_About_Project_and_CO_Mapping",
}

selected_page = st.sidebar.radio("Select Module:", list(PAGES.keys()))

st.sidebar.markdown("---")
st.sidebar.caption("System Status: **Active** | Python 3.11+ | PyTorch | Scikit-Learn | FastAPI")

# Dynamically import and render selected page
import importlib

try:
    module_name = PAGES[selected_page]
    page_module = importlib.import_module(module_name)
    page_module.render_page()
except Exception as e:
    st.error(f"Error loading module {selected_page}: {str(e)}")
    import traceback
    st.code(traceback.format_exc())
