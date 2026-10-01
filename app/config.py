"""
CyberTwin Application Configuration
====================================
Centralizes all model paths, result paths, and application parameters.
All paths are resolved relative to the project root.
"""

import os

# ─────────────────────────────────────────────
# Project Root (one level up from app/)
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
# Model Paths
# ─────────────────────────────────────────────
MODEL_PATHS = {
    'random_forest':     os.path.join(BASE_DIR, 'models', 'random_forest.joblib'),
    'logistic_regression': os.path.join(BASE_DIR, 'models', 'logistic_regression.joblib'),
    'temporal_bilstm':   os.path.join(BASE_DIR, 'results', 'milestone_4_1', 'temporal_model_best.pth'),
    'cnn_bilstm':        os.path.join(BASE_DIR, 'models', 'cybertwin_cnn_bilstm_best.pth'),
    'preprocessor':      os.path.join(BASE_DIR, 'models', 'preprocessor.joblib'),
    'metadata':          os.path.join(BASE_DIR, 'models', 'cybertwin_model_metadata.json'),
}

# ─────────────────────────────────────────────
# Processed Data Paths
# ─────────────────────────────────────────────
DATA_PATHS = {
    'X_train': os.path.join(BASE_DIR, 'data', 'processed', 'X_train.csv'),
    'y_train': os.path.join(BASE_DIR, 'data', 'processed', 'y_train.csv'),
    'X_val':   os.path.join(BASE_DIR, 'data', 'processed', 'X_val.csv'),
    'y_val':   os.path.join(BASE_DIR, 'data', 'processed', 'y_val.csv'),
    'X_test':  os.path.join(BASE_DIR, 'data', 'processed', 'X_test.csv'),
    'y_test':  os.path.join(BASE_DIR, 'data', 'processed', 'y_test.csv'),
}

# ─────────────────────────────────────────────
# Result CSV Paths
# ─────────────────────────────────────────────
RESULT_PATHS = {
    # Milestone 3 model comparison (flow-level)
    'flow_model_comparison':   os.path.join(BASE_DIR, 'results', 'tables', 'final_model_comparison.csv'),

    # Milestone 4.1 corrected temporal comparison (sequence-level)
    'temporal_model_comparison': os.path.join(BASE_DIR, 'results', 'milestone_4_1', 'tables', 'corrected_model_comparison.csv'),

    # Temporal order ablation
    'ablation_metrics':  os.path.join(BASE_DIR, 'results', 'milestone_4_1', 'tables', 'temporal_order_ablation.csv'),
    'ablation_delta':    os.path.join(BASE_DIR, 'results', 'milestone_4_1', 'tables', 'temporal_order_ablation_delta.csv'),
    'scenario_ablation': os.path.join(BASE_DIR, 'results', 'milestone_4_1', 'tables', 'scenario_order_ablation.csv'),

    # SHAP
    'shap_importance':   os.path.join(BASE_DIR, 'results', 'milestone_4_1', 'tables', 'shap_rf_importance.csv'),

    # Sequence audit
    'sequence_audit':    os.path.join(BASE_DIR, 'results', 'milestone_4_1', 'tables', 'sequence_source_audit.csv'),
}

# ─────────────────────────────────────────────
# Figure Paths
# ─────────────────────────────────────────────
FIGURE_PATHS = {
    'shap_bar':            os.path.join(BASE_DIR, 'results', 'milestone_4_1', 'figures', 'shap_rf_bar.png'),
    'shap_summary':        os.path.join(BASE_DIR, 'results', 'milestone_4_1', 'figures', 'shap_rf_summary.png'),
    'confusion_rf':        os.path.join(BASE_DIR, 'results', 'figures', 'confusion_matrix_random_forest.png'),
    'confusion_lr':        os.path.join(BASE_DIR, 'results', 'figures', 'confusion_matrix_logistic_regression.png'),
    'roc_curve':           os.path.join(BASE_DIR, 'results', 'figures', 'roc_curve_comparison.png'),
    'pr_curve':            os.path.join(BASE_DIR, 'results', 'figures', 'pr_curve_comparison.png'),
    'ablation_metrics':    os.path.join(BASE_DIR, 'results', 'milestone_4_1', 'figures', 'temporal_order_ablation_metrics.png'),
    'ablation_confusion':  os.path.join(BASE_DIR, 'results', 'milestone_4_1', 'figures', 'confusion_ordered_vs_shuffled.png'),
}

# ─────────────────────────────────────────────
# Temporal Model Parameters
# ─────────────────────────────────────────────
TEMPORAL_CONFIG = {
    'input_dim': 70,
    'hidden_dim': 128,
    'num_layers': 2,
    'dropout': 0.3,
    'sequence_length': 20,
}

# ─────────────────────────────────────────────
# Digital Twin State Thresholds
# ─────────────────────────────────────────────
# These are manually configured academic-prototype thresholds,
# matching the values in src/digital_twin.py.
DT_THRESHOLDS = {
    'suspicious_prob':     0.5,
    'suspicious_risk':     0.4,
    'under_attack_prob':   0.8,
    'under_attack_risk':   0.7,
    'recovery_prob':       0.2,
    'recovery_risk':       0.5,
    'normal_risk':         0.3,
    'normal_prob':         0.2,
}

# ─────────────────────────────────────────────
# Risk Score Interpretation (0–100 scale)
# ─────────────────────────────────────────────
RISK_BANDS = {
    'low':      (0, 30),
    'moderate': (31, 60),
    'high':     (61, 80),
    'critical': (81, 100),
}

# ─────────────────────────────────────────────
# Class Mapping
# ─────────────────────────────────────────────
CLASS_MAP = {0: 'BENIGN', 1: 'PortScan'}

# ─────────────────────────────────────────────
# Scenario Sequence Counts (test partition)
# ─────────────────────────────────────────────
SCENARIO_COUNTS = {
    'NORMAL':           200,
    'ATTACK_ONSET':     100,
    'SUSTAINED_ATTACK': 200,
    'ATTACK_RECOVERY':  100,
}
