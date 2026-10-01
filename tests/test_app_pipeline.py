import pytest
import numpy as np
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.model_loader import load_random_forest, load_temporal_bilstm, load_preprocessor
from app.utils.prediction import predict_rf_flows, predict_bilstm_sequence
from src.digital_twin import DigitalTwinNode, NodeState

def test_model_loading():
    rf = load_random_forest()
    assert rf is not None, "Random Forest model failed to load"
    
    tb_result = load_temporal_bilstm()
    assert tb_result is not None, "Temporal BiLSTM model failed to load"
    tb_model, device = tb_result
    
    prep = load_preprocessor()
    assert prep is not None, "Preprocessor failed to load"

def test_predictions():
    rf = load_random_forest()
    tb_result = load_temporal_bilstm()
    tb_model, device = tb_result
    
    # Dummy feature vector (70 features)
    dummy_flow = np.random.randn(1, 70)
    
    # Test RF
    rf_pred, rf_prob = predict_rf_flows(rf, dummy_flow)
    assert len(rf_pred) == 1
    assert len(rf_prob) == 1
    assert 0 <= rf_prob[0] <= 1
    
    # Test BiLSTM
    # Sequence length 20
    dummy_seq = np.random.randn(20, 70)
    tb_pred, tb_prob = predict_bilstm_sequence(tb_model, device, dummy_seq)
    assert tb_pred in [0, 1]
    assert 0 <= tb_prob <= 1

def test_digital_twin_state():
    node = DigitalTwinNode("test_node")
    assert node.state == NodeState.NORMAL
    
    # NORMAL -> SUSPICIOUS
    dummy_flow_low = np.zeros(70)
    node.update_metrics(dummy_flow_low, 0.6) # Risk = 0.5*0.6 + 0.3*(1.0) + 0 = 0.6 (> 0.4 -> SUSPICIOUS)
    assert node.state == NodeState.SUSPICIOUS
    
    # SUSPICIOUS -> UNDER_ATTACK
    dummy_flow_high = np.ones(70)
    node.update_metrics(dummy_flow_high, 0.9)
    assert node.state == NodeState.UNDER_ATTACK
    
    # UNDER_ATTACK -> RECOVERING
    for _ in range(50):
        node.update_metrics(np.zeros(70), 0.1)
        
    assert node.state == NodeState.NORMAL
