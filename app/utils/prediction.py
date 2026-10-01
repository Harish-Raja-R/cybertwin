"""
Prediction Utilities
=====================
Wrappers for flow-level (RF) and sequence-level (Temporal BiLSTM) inference.
"""
import numpy as np
import torch


def predict_rf_flows(model, X):
    """
    Flow-level prediction using Random Forest.

    Parameters
    ----------
    model : sklearn RandomForestClassifier
    X : array-like of shape (n_samples, 70)

    Returns
    -------
    predictions : ndarray of shape (n_samples,)  — 0 or 1
    probabilities : ndarray of shape (n_samples,) — P(PortScan)
    """
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs


def predict_bilstm_sequence(model, device, sequence):
    """
    Sequence-level prediction using Temporal BiLSTM.

    Parameters
    ----------
    model : TemporalBiLSTM
    device : torch.device
    sequence : array-like of shape (seq_len, 70)

    Returns
    -------
    prediction : int — 0 or 1
    probability : float — P(Attack)
    """
    seq_tensor = torch.tensor(
        np.asarray(sequence, dtype=np.float32)
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(seq_tensor)
        prob = torch.sigmoid(logits).item()

    return int(prob >= 0.5), prob
