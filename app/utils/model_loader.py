"""
Model Loading Utilities
========================
Cached loaders for all CyberTwin models.
Uses @st.cache_resource so models are loaded once per session.
"""
import os
import json
import joblib
import torch
import streamlit as st
import sys

# Ensure project root is on path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.config import MODEL_PATHS, TEMPORAL_CONFIG
from src.models.temporal_bilstm import TemporalBiLSTM


@st.cache_resource
def load_preprocessor():
    """Load the fitted RobustScaler preprocessing pipeline."""
    path = MODEL_PATHS['preprocessor']
    if not os.path.exists(path):
        st.error(f"Preprocessor not found: `{path}`")
        return None
    return joblib.load(path)


@st.cache_resource
def load_random_forest():
    """Load the trained Random Forest classifier."""
    path = MODEL_PATHS['random_forest']
    if not os.path.exists(path):
        st.error(f"Random Forest model not found: `{path}`")
        return None
    return joblib.load(path)


@st.cache_resource
def load_logistic_regression():
    """Load the trained Logistic Regression classifier."""
    path = MODEL_PATHS['logistic_regression']
    if not os.path.exists(path):
        st.error(f"Logistic Regression model not found: `{path}`")
        return None
    return joblib.load(path)


@st.cache_resource
def load_temporal_bilstm():
    """
    Load the corrected M4.1 Temporal BiLSTM model.
    Returns (model, device) tuple or None on failure.
    """
    path = MODEL_PATHS['temporal_bilstm']
    if not os.path.exists(path):
        st.error(f"Temporal BiLSTM checkpoint not found: `{path}`")
        return None

    device = torch.device('cpu')  # CPU-only on this system
    model = TemporalBiLSTM(
        input_dim=TEMPORAL_CONFIG['input_dim'],
        hidden_dim=TEMPORAL_CONFIG['hidden_dim'],
        num_layers=TEMPORAL_CONFIG['num_layers'],
        dropout=TEMPORAL_CONFIG['dropout'],
    )
    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    return model, device


@st.cache_resource
def load_metadata():
    """Load model metadata JSON."""
    path = MODEL_PATHS['metadata']
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)
