"""
CSV Validation Utilities
========================
Validates uploaded CSV files against the expected feature set.
"""
import pandas as pd
import json
import streamlit as st
import os

from app.config import MODEL_PATHS

def validate_uploaded_csv(uploaded_file):
    """
    Validates the uploaded CSV file.
    Checks if all required features are present based on model metadata.
    
    Returns
    -------
    tuple: (DataFrame or None, list of missing features or error message)
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, f"Error reading CSV: {e}"

    if not os.path.exists(MODEL_PATHS['metadata']):
        return None, "Model metadata not found. Cannot validate features."

    try:
        with open(MODEL_PATHS['metadata'], 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        return None, f"Error reading model metadata: {e}"
        
    expected_dim = metadata.get("feature_dimension", 70)
    
    # We can't strictly validate exact column names if the metadata only has dimension,
    # but let's assume the user knows they need to provide exactly 70 features 
    # (after dropping non-features) or matching the original pipeline.
    # In a real app, we'd check `metadata['feature_names']`.
    
    # For this prototype, we check dimension:
    # If the user uploaded something with fewer columns than expected, fail.
    # If they uploaded exactly or more, we assume the first 70 are the features.
    
    if df.shape[1] < expected_dim:
        return None, f"Missing features. Expected {expected_dim} columns, but found {df.shape[1]}."
        
    return df, None
