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

EXPECTED_FEATURES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 
    'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 
    'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 
    'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 
    'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 
    'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 
    'Bwd IAT Min', 'Fwd PSH Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 
    'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 
    'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 
    'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'ECE Flag Count', 
    'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 
    'Fwd Header Length.1', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 
    'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 
    'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Total_Packets', 'Fwd_Bwd_Packet_Ratio', 
    'Fwd_Bwd_Byte_Ratio'
]

def validate_uploaded_csv(uploaded_file):
    """
    Validates the uploaded CSV file against the exact feature schema.
    
    Returns
    -------
    tuple: (DataFrame or None, list of missing features or error message)
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, f"Error reading CSV: {e}"

    missing_cols = [col for col in EXPECTED_FEATURES if col not in df.columns]
    unexpected_cols = [col for col in df.columns if col not in EXPECTED_FEATURES]

    if missing_cols:
        error_msg = f"Invalid input file. Missing required features: {missing_cols[:5]}{'...' if len(missing_cols)>5 else ''}. "
        if unexpected_cols:
            error_msg += f"Unexpected features: {unexpected_cols[:5]}{'...' if len(unexpected_cols)>5 else ''}. "
        error_msg += "Please upload a CSV containing the expected CyberTwin feature schema."
        return None, error_msg

    # Ensure correct feature order before inference
    df = df[EXPECTED_FEATURES]
    
    # Validate numeric conversion
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing values consistently with training (e.g. median imputation normally, but for now we just fillna 0 or drop)
    if df.isnull().values.any():
        df = df.fillna(0)

    return df, None
