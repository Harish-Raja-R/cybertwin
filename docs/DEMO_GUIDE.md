# CyberTwin: Demonstration Guide

## Welcome to CyberTwin
CyberTwin is an academic prototype of a Digital Twin-based cyberattack detection system utilizing deep learning. This guide outlines how to effectively demonstrate the prototype to an audience, showcasing the project's progression from static tabular machine learning to state-aware sequence modeling.

## Pre-requisites
1. Ensure all Python dependencies from `requirements.txt` are installed.
2. Verify that the pre-trained models (`random_forest.joblib`, `temporal_model_best.pth`, `preprocessor.joblib`) are located in their respective directories.
3. Start the application:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Demonstration Walkthrough

### 1. Dashboard Overview
- **Action:** Open the application; the default landing page is the Dashboard.
- **Talking Points:** 
  - Introduce CyberTwin as an advanced ML academic project tracking cyber threats using Digital Twin architecture.
  - Explain the purpose of the KPIs, establishing that the network is currently at a `NORMAL` state.

### 2. Simulation (Core Digital Twin Feature)
- **Action:** Navigate to the `Simulation` page via the sidebar.
- **Action:** Select **"Attack Onset"** and click **Run Simulation**.
- **Talking Points:**
  - The simulation demonstrates how chronological windows (sequences of 20 flows) are evaluated by the Temporal BiLSTM.
  - Explain the **Risk Score formula** (expand the info section).
  - Observe how continuous malicious flows steadily raise the risk score, pushing the Digital Twin state from `NORMAL` to `SUSPICIOUS` and ultimately `UNDER_ATTACK`.

### 3. Attack Detection (Static vs Temporal)
- **Action:** Navigate to the `Attack Detection` page.
- **Action:** Use the **Demo Temporal Scenario** to evaluate a single sequence.
- **Talking Points:**
  - Highlight the side-by-side comparison between the Random Forest (which evaluates only the most recent flow in the sequence) and the Temporal BiLSTM (which evaluates the entire sequential context).
  - This contrasts stateless, traditional tabular detection against state-aware temporal evaluation.

### 4. Model Performance and Scientific Rigor
- **Action:** Navigate to the `Model Performance` page.
- **Talking Points:**
  - Show the rigorous metric comparisons (Accuracy, Precision, Recall, F1).
  - **Crucial Point:** Discuss the **Temporal Order Ablation** experiment at the bottom of the page. Reiterate that the prototype tested whether the Temporal BiLSTM was genuinely utilizing chronology, proving that randomizing the order caused a drop in performance, thus verifying the sequence-learning hypothesis.

### 5. Explainability (XAI)
- **Action:** Navigate to the `Explainability` page.
- **Talking Points:**
  - Showcase the SHAP global feature importance and summary plots.
  - State clearly the academic constraint: *These SHAP values explain the Random Forest (flow-level detector) behavior.* Explaining the Recurrent Neural Network directly using SHAP is out of scope due to the complexity of sequential gradients.

### 6. Conclusion / About
- **Action:** End on the `About` page.
- **Talking Points:** Summarize the pipeline from the CIC-IDS2017 dataset through preprocessing, modeling, and digital twin simulation. Reiterate the academic limitation regarding chronological synthesis of the dataset.
