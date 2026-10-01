import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import os
import sys

# Ensure the parent directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import (
    RESULT_PATHS, 
    FIGURE_PATHS, 
    SCENARIO_COUNTS, 
    DT_THRESHOLDS,
    CLASS_MAP,
    TEMPORAL_CONFIG
)
from app.utils.model_loader import load_temporal_bilstm, load_random_forest, load_logistic_regression, load_preprocessor
from app.utils.prediction import predict_bilstm_sequence, predict_rf_flows
from app.utils.digital_twin_state import DigitalTwinNode, state_color, state_emoji, risk_band, risk_band_color
from app.utils.validation import validate_uploaded_csv

st.set_page_config(page_title="CyberTwin Dashboard", page_icon="🛡️", layout="wide")

# Custom CSS for KPI cards and layout
st.markdown("""
<style>
.kpi-card {
    background-color: #1E1E1E;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    text-align: center;
    border: 1px solid #333;
}
.kpi-title {
    font-size: 1.1rem;
    color: #AAAAAA;
    margin-bottom: 10px;
}
.kpi-value {
    font-size: 2rem;
    font-weight: bold;
    color: #FFFFFF;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.title("CYBERTWIN")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "Dashboard",
    "Attack Detection",
    "Digital Twin",
    "Explainability",
    "Model Performance",
    "Simulation",
    "About"
])
st.sidebar.markdown("---")
st.sidebar.markdown("### System")
st.sidebar.markdown("Dataset: CIC-IDS2017")
st.sidebar.markdown("Models: RF + Temporal BiLSTM")
st.sidebar.markdown("Mode: Academic Prototype")

# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
@st.cache_data
def get_scenarios():
    """Load the synthetic temporal sequences for the Digital Twin simulation."""
    # Try loading from the corrected simulation output first
    from src.corrected_temporal_simulator import load_and_simulate_corrected
    # Redirect stdout to avoid messing up streamlit console if we can
    try:
        # Avoid reloading everything if already saved, but we'll use the function since it handles the generation
        # Since it takes time, we should ideally load the saved npz or run it once.
        # But for this prototype, we'll run it on the fly if needed, cached by Streamlit.
        train_data, val_data, test_data, manifest = load_and_simulate_corrected(window_size=TEMPORAL_CONFIG['sequence_length'])
        
        audit_test = pd.read_csv(RESULT_PATHS['sequence_audit'])
        audit_test = audit_test[audit_test['source_partition'] == 'TEST'].reset_index(drop=True)
        return test_data, audit_test
    except Exception as e:
        st.error(f"Error loading scenarios: {e}")
        return None, None

def render_kpi_card(title, value, color="#FFFFFF"):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value" style="color: {color};">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Global State for Dashboard (Default placeholders)
# ---------------------------------------------------------
if 'dashboard_state' not in st.session_state:
    st.session_state.dashboard_state = {
        'state': 'NORMAL',
        'prob': 0.0,
        'risk': 0.0,
        'model': 'Temporal BiLSTM'
    }

# ---------------------------------------------------------
# Pages
# ---------------------------------------------------------

if page == "Dashboard":
    st.title("CyberTwin")
    st.markdown("> **Digital Twin-Based Cyberattack Detection Using Deep Learning**")
    
    # KPIs
    ds = st.session_state.dashboard_state
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Network State", f"{state_emoji(ds['state'])} {ds['state']}", state_color(ds['state']))
    with col2:
        render_kpi_card("Attack Probability", f"{ds['prob']*100:.1f}%", state_color(ds['state']))
    with col3:
        render_kpi_card("Risk Score", f"{ds['risk']:.0f} / 100", risk_band_color(ds['risk']))
    with col4:
        render_kpi_card("Detection Model", ds['model'])
        
    st.markdown("---")
    st.markdown("### Welcome to CyberTwin")
    st.markdown("Please navigate to **Simulation** to run the Digital Twin cyberattack scenarios, or explore the **Explainability** and **Model Performance** pages for detailed project outputs.")

elif page == "Simulation":
    st.title("Scenario Simulation")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### Select Scenario")
        scenario = st.radio("", ["Normal", "Attack Onset", "Sustained Attack", "Attack Recovery"])
        run_btn = st.button("▶ Run Simulation", type="primary")
        
        with st.expander("How is risk calculated?"):
            st.markdown("""
            **Risk Score Formula:**
            `0.5 * Attack_Prob + 0.3 * Recent_Suspicious_Ratio + 0.2 * Norm_Traffic_Intensity`
            
            *(Scaled to 0-100 for display)*
            """)
    
    with col2:
        if not run_btn:
            st.info("Select a scenario and click 'Run Simulation' to observe the Digital Twin state transitions over time.")
            
        if run_btn:
            with st.spinner("Loading scenarios and models..."):
                test_data, audit_test = get_scenarios()
                model_info = load_temporal_bilstm()
            
            if model_info is None or test_data is None:
                st.error("Simulation dependencies could not be loaded.")
            else:
                model, device = model_info
                
                # Map selected option to scenario key
                scen_map = {
                    "Normal": "NORMAL",
                    "Attack Onset": "ATTACK_ONSET",
                    "Sustained Attack": "SUSTAINED_ATTACK",
                    "Attack Recovery": "ATTACK_RECOVERY"
                }
                scen_key = scen_map[scenario]
                scen_indices = audit_test[audit_test['scenario'] == scen_key].index.tolist()
                
                if not scen_indices:
                    st.warning(f"No sequences found for {scenario}.")
                else:
                    # To show a timeline, we take a subset of sequences from this scenario
                    # A typical simulation runs over multiple steps. The sequences in test_data are independent windows.
                    # For a demonstration, we will evaluate 20 consecutive windows to build a timeline.
                    selected_indices = scen_indices[:20]
                    
                    dt_node = DigitalTwinNode("node_001")
                    
                    progress_bar = st.progress(0)
                    
                    results = []
                    
                    for idx, i in enumerate(selected_indices):
                        seq = test_data[0][i]
                        pred, prob = predict_bilstm_sequence(model, device, seq)
                        
                        # Use the last flow in sequence for intensity proxy
                        dt_node.update_metrics(seq[-1], prob)
                        
                        current_state = dt_node.state.value
                        
                        results.append({
                            'Simulation Step': idx + 1,
                            'Probability': prob,
                            'State': current_state,
                            'Risk': dt_node.risk_score * 100
                        })
                        
                        # Simulate animation delay
                        time.sleep(0.1)
                        progress_bar.progress((idx + 1) / len(selected_indices))
                    
                    progress_bar.empty()
                    
                    res_df = pd.DataFrame(results)
                    final_state = res_df.iloc[-1]['State']
                    final_prob = res_df.iloc[-1]['Probability']
                    final_risk = res_df.iloc[-1]['Risk']
                    
                    # Update global dashboard state
                    st.session_state.dashboard_state = {
                        'state': final_state,
                        'prob': final_prob,
                        'risk': final_risk,
                        'model': 'Temporal BiLSTM'
                    }
                    
                    # Update KPIs locally for immediate feedback
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    with kpi1:
                        render_kpi_card("Network State", f"{state_emoji(final_state)} {final_state}", state_color(final_state))
                    with kpi2:
                        render_kpi_card("Attack Probability", f"{final_prob*100:.1f}%", state_color(final_state))
                    with kpi3:
                        render_kpi_card("Risk Score", f"{final_risk:.0f} / 100", risk_band_color(final_risk))
                    with kpi4:
                        render_kpi_card("Detection Model", "Temporal BiLSTM")
                        
                    # Alerts
                    st.markdown("### System Alert")
                    if final_state == "NORMAL":
                        st.success("No significant malicious activity detected.")
                    elif final_state == "SUSPICIOUS":
                        st.warning("Suspicious traffic pattern detected. Continue monitoring.")
                    elif final_state == "UNDER_ATTACK":
                        st.error("Potential cyberattack detected. Immediate investigation recommended.")
                    elif final_state == "RECOVERING":
                        st.info("Attack indicators are decreasing. Network remains under observation.")
                        
                    # Timeline Plotly
                    st.markdown("### Digital Twin Timeline")
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=res_df['Simulation Step'], y=res_df['Probability']*100,
                        mode='lines+markers', name='Attack Prob (%)', line=dict(color='red')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=res_df['Simulation Step'], y=res_df['Risk'],
                        mode='lines+markers', name='Risk Score', line=dict(color='orange')
                    ))
                    
                    fig.add_hline(y=DT_THRESHOLDS['under_attack_prob']*100, line_dash="dash", line_color="red", annotation_text="Attack Threshold")
                    fig.add_hline(y=DT_THRESHOLDS['suspicious_prob']*100, line_dash="dash", line_color="yellow", annotation_text="Suspicious Threshold")
                    
                    fig.update_layout(title="Attack Probability & Risk Score vs Simulation Step",
                                      xaxis_title="Simulation Step",
                                      yaxis_title="Value",
                                      yaxis=dict(range=[0, 105]))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### Prediction Log")
                    st.dataframe(res_df.style.format({'Probability': '{:.4f}', 'Risk': '{:.1f}'}))

elif page == "Digital Twin":
    st.title("Digital Twin Architecture")
    st.markdown("The Digital Twin state machine monitors sequences of network traffic and calculates a risk score based on the model's attack probability and historical activity.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### State Definitions")
        st.markdown(f"**{state_emoji('NORMAL')} NORMAL** - Low attack probability")
        st.markdown(f"**{state_emoji('SUSPICIOUS')} SUSPICIOUS** - Increasing attack probability")
        st.markdown(f"**{state_emoji('UNDER_ATTACK')} UNDER ATTACK** - High attack probability")
        st.markdown(f"**{state_emoji('RECOVERING')} RECOVERING** - Attack probability decreasing after detected attack")
    with col2:
        st.markdown("### Transitions")
        st.code("""
NORMAL -> SUSPICIOUS (Prob > 0.5 or Risk > 0.4)
SUSPICIOUS -> UNDER ATTACK (Prob > 0.8 or Risk > 0.7)
UNDER ATTACK -> RECOVERING (Prob < 0.2 and Risk < 0.5)
RECOVERING -> NORMAL (Risk < 0.3)
        """)

elif page == "Attack Detection":
    st.title("Static vs Temporal Attack Detection")
    st.markdown("Select an input mode and compare the Flow-level (Random Forest) vs Sequence-level (Temporal BiLSTM) detectors.")
    
    input_mode = st.radio("Input Mode", ["Sample CIC-IDS2017 data", "Demo temporal scenario", "Upload CSV"])
    
    if input_mode == "Demo temporal scenario":
        test_data, audit_test = get_scenarios()
        if test_data is not None:
            sample_idx = st.number_input("Select Sequence Index (0 to 599)", min_value=0, max_value=len(audit_test)-1, value=0)
            
            if st.button("Detect"):
                seq = test_data[0][sample_idx]
                label = test_data[1][sample_idx]
                scen = audit_test.iloc[sample_idx]['scenario']
                
                st.write(f"**True Label:** {CLASS_MAP[int(label)]} ({scen})")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Random Forest (Flow-level)")
                    rf_model = load_random_forest()
                    if rf_model:
                        # Evaluate on the last flow of the sequence
                        last_flow = seq[-1].reshape(1, -1)
                        rf_preds, rf_probs = predict_rf_flows(rf_model, last_flow)
                        rf_pred, rf_prob = rf_preds[0], rf_probs[0]
                        
                        st.metric("Attack Probability", f"{rf_prob*100:.1f}%")
                        st.metric("Predicted Class", CLASS_MAP[rf_pred])
                        
                with col2:
                    st.markdown("### Temporal BiLSTM (Sequence-level)")
                    tb_model = load_temporal_bilstm()
                    if tb_model:
                        model, device = tb_model
                        tb_pred, tb_prob = predict_bilstm_sequence(model, device, seq)
                        
                        st.metric("Attack Probability", f"{tb_prob*100:.1f}%")
                        st.metric("Predicted State", CLASS_MAP[tb_pred])
    
    elif input_mode == "Sample CIC-IDS2017 data":
        st.info("Select a single flow from the pre-processed validation set.")
        # Load a small snippet of X_val
        try:
            x_val = pd.read_csv(RESULT_PATHS['flow_model_comparison'].replace('results/tables/final_model_comparison.csv', 'data/processed/X_val.csv'), nrows=100)
            y_val = pd.read_csv(RESULT_PATHS['flow_model_comparison'].replace('results/tables/final_model_comparison.csv', 'data/processed/y_val.csv'), nrows=100)
            
            idx = st.slider("Flow Index", 0, 99, 0)
            if st.button("Detect Flow"):
                flow = x_val.iloc[idx].values.reshape(1, -1)
                true_label = int(y_val.iloc[idx].values[0])
                st.write(f"**True Label:** {CLASS_MAP[true_label]}")
                
                rf_model = load_random_forest()
                if rf_model:
                    rf_preds, rf_probs = predict_rf_flows(rf_model, flow)
                    rf_pred, rf_prob = rf_preds[0], rf_probs[0]
                    
                    st.metric("Random Forest Probability", f"{rf_prob*100:.1f}%")
                    st.metric("Random Forest Prediction", CLASS_MAP[rf_pred])
                    st.info("Note: Temporal BiLSTM requires a sequence of 20 flows, not a single flow.")
        except Exception as e:
            st.error(f"Error loading sample data: {e}")

    elif input_mode == "Upload CSV":
        st.info("CSV Upload Prototype")
        uploaded_file = st.file_uploader("Upload CSV containing exactly 70 processed flow features", type="csv")
        if uploaded_file is not None:
            df, error = validate_uploaded_csv(uploaded_file)
            if error:
                st.error(error)
            else:
                st.success("CSV validated.")
                st.write(df.head())
                st.warning("Ground-truth labels are unavailable; displaying predictions only.")
                
                if st.button("Detect on Uploaded Data"):
                    rf_model = load_random_forest()
                    if rf_model:
                        preds, probs = predict_rf_flows(rf_model, df.values[:, :70])
                        df['Predicted_Prob'] = probs
                        df['Predicted_Class'] = [CLASS_MAP[p] for p in preds]
                        st.dataframe(df[['Predicted_Prob', 'Predicted_Class']])

elif page == "Model Performance":
    st.title("Model Performance")
    
    st.markdown("### Evaluation Metrics")
    st.markdown("Comparing different models evaluated natively on their respective validation structures.")
    
    try:
        m3_df = pd.read_csv(RESULT_PATHS['flow_model_comparison'])
        m4_df = pd.read_csv(RESULT_PATHS['temporal_model_comparison'])
        
        m3_df['Evaluation'] = "Flow-level"
        m4_df['Evaluation'] = "Sequence-level"
        
        # Combine
        combined = pd.concat([m3_df, m4_df], ignore_index=True)
        # Select key metrics
        cols = ['model', 'Evaluation', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'specificity', 'false_positive_rate', 'false_negative_rate']
        available_cols = [c for c in cols if c in combined.columns]
        
        st.dataframe(combined[available_cols].style.format({
            c: "{:.4f}" for c in available_cols if c not in ['model', 'Evaluation']
        }))
    except Exception as e:
        st.error(f"Could not load performance metrics: {e}")
        
    st.markdown("---")
    st.markdown("## Temporal Order Ablation")
    try:
        ablation = pd.read_csv(RESULT_PATHS['ablation_metrics'])
        st.dataframe(ablation[['experiment', 'ordering', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'specificity']].style.format({
            'accuracy': '{:.4f}', 'precision': '{:.4f}', 'recall': '{:.4f}', 'f1': '{:.4f}', 'roc_auc': '{:.4f}', 'specificity': '{:.4f}'
        }))
        
        st.markdown("### Differences")
        delta = pd.read_csv(RESULT_PATHS['ablation_delta'])
        st.write(f"**Δ Accuracy:** {delta['delta_accuracy'].iloc[0]:+.4f}")
        st.write(f"**Δ F1:** {delta['delta_f1'].iloc[0]:+.4f}")
        st.write(f"**Δ ROC-AUC:** {delta['delta_roc_auc'].iloc[0]:+.4f}")
        
        st.info("> Destroying temporal ordering reduced performance, particularly for attack-recovery scenarios, providing evidence that sequence ordering contributes useful information under the controlled simulation.")
        st.warning("**Limitation:** These sequences are synthetically constructed from CIC-IDS2017 observations and therefore do not represent reconstructed historical network chronology.")
    except Exception as e:
        st.error(f"Could not load ablation metrics: {e}")

elif page == "Explainability":
    st.title("Explainability (XAI)")
    st.markdown("> **SHAP explanations shown here correspond to the Random Forest flow-level detector. They should not be interpreted as direct feature attributions for the recurrent Temporal BiLSTM.**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Global Feature Importance")
        if os.path.exists(FIGURE_PATHS['shap_bar']):
            st.image(FIGURE_PATHS['shap_bar'], use_container_width=True)
        else:
            st.warning("SHAP bar plot not found.")
            
    with col2:
        st.markdown("### Feature Impact (Summary)")
        if os.path.exists(FIGURE_PATHS['shap_summary']):
            st.image(FIGURE_PATHS['shap_summary'], use_container_width=True)
        else:
            st.warning("SHAP summary plot not found.")
            
    st.markdown("### Top Features")
    if os.path.exists(RESULT_PATHS['shap_importance']):
        imp_df = pd.read_csv(RESULT_PATHS['shap_importance'])
        st.dataframe(imp_df.head(10))
    else:
        st.warning("SHAP importance CSV not found.")

elif page == "About":
    st.title("About CyberTwin")
    st.markdown("""
    ### Problem
    Traditional IDS systems often classify individual traffic flows without representing evolving network state.

    ### Proposed Solution
    CyberTwin combines:
    * Machine learning
    * Deep learning
    * Temporal sequence modeling
    * Digital twin state representation
    * Risk scoring
    * Explainable AI
    
    ### Architecture
    """)
    st.code("""
              ┌────────────────────┐
              │   CIC-IDS2017      │
              └─────────┬──────────┘
                        ↓
              ┌────────────────────┐
              │ Preprocessing      │
              │ RobustScaler       │
              │ Feature Engineering│
              └─────────┬──────────┘
                        ↓
              ┌────────────────────┐
              │ Flow-Level RF      │
              └─────────┬──────────┘
                        ↓
              ┌────────────────────┐
              │ Temporal BiLSTM    │
              └─────────┬──────────┘
                        ↓
              ┌────────────────────┐
              │ Attack Probability │
              └─────────┬──────────┘
                        ↓
              ┌────────────────────┐
              │ Digital Twin       │
              │ State + Risk       │
              └─────────┬──────────┘
                        ↓
              ┌────────────────────┐
              │ Explainability     │
              │ SHAP / Dashboard   │
              └────────────────────┘
    """)
    
    st.markdown("""
    ### Important Limitation
    The current prototype uses controlled synthetic temporal sequences because the selected CIC-IDS2017 CSV slice does not retain reliable chronological metadata such as Timestamp/Flow ID/Source IP.
    """)
