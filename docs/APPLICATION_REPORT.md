# CyberTwin: Application Architecture Report

## 1. Overview
The CyberTwin Digital Twin Cyberattack Detection system transforms static, flow-level classification into a dynamic, state-aware cyber monitoring system. The `CyberTwin Dashboard`, built with Streamlit, integrates the experimental deep learning models developed throughout the milestones into a working interactive application.

## 2. Architecture
The application leverages a unified backend to serve inferences using two specialized models:
- **Flow-Level Detector:** A Random Forest algorithm representing traditional, stateless network security.
- **Sequence-Level Temporal Detector:** A Temporal BiLSTM representing the digital twin's predictive capability by analyzing chronologically ordered sub-sequences (windows of 20 flows) to determine cyberattack presence over time.

### Component Breakdown
1. **Model Loader (`app/utils/model_loader.py`):** Utilizes `st.cache_resource` for efficient in-memory loading of the Random Forest model (`.joblib`), the PyTorch Temporal BiLSTM (`.pth`), and the Preprocessor pipeline.
2. **Prediction Engine (`app/utils/prediction.py`):** Handles individual prediction inferences, correctly matching dimensions for tabular arrays and recurrent tensors.
3. **Digital Twin Simulator (`src/digital_twin.py`):** The state-transition system maintaining network entity health (NORMAL, SUSPICIOUS, UNDER_ATTACK, RECOVERING) influenced by an aggregated risk score formula combining recent attack probabilities, attack persistence, and proxy traffic volume.

## 3. UI/UX Workflow
The application is segmented into several logical views via a sidebar:
- **Dashboard:** Provides high-level KPIs representing the current network status (State, Attack Probability, Risk Score).
- **Simulation:** A dynamic sandbox allowing users to observe digital twin behavior under various cyberattack scenarios (Normal, Attack Onset, Sustained Attack, Attack Recovery).
- **Attack Detection:** Side-by-side comparative detection using both the Random Forest and Temporal BiLSTM algorithms on single network sequences.
- **Model Performance:** Retrospective on rigorous testing (M3 and M4) and the crucial Temporal Order Ablation study demonstrating the benefit of chronological ordering for deep sequence learners.
- **Explainability:** Showcasing SHAP interpretations isolated strictly to the Random Forest model to identify primary contributors to attack indications.

## 4. Constraints and Ethical Prototype Usage
As an academic prototype built strictly from an offline evaluation dataset (CIC-IDS2017), the system incorporates synthetic chronological simulations rather than operating over a live `pcap` stream. All timestamps, IP addresses, packet contents, and metrics shown in the dashboard represent true observed historical data properties or strictly defined experimental parameters resulting from the pipeline; none of this data is fabricated for aesthetic purposes.
