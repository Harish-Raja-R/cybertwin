# CyberTwin: Digital Twin-Based Cyberattack Detection Using Deep Learning

## Project Objective
The goal of this project is to simulate a Digital Twin of a network environment and use deep learning to detect cyberattacks from network traffic. This repository implements the data pipeline, EDA, modeling, and evaluation of an advanced machine learning intrusion detection system.

## Current Milestone: Milestone 1
**Milestone 1 completed:** Dataset acquisition, verification, audit, and exploratory data analysis.

## Folder Structure
```
CyberTwin/
├── data/
│   ├── raw/          # Raw dataset files
│   ├── processed/    # Cleaned data
│   ├── splits/       # Train/test/validation splits
│   └── README.md
├── notebooks/        # Jupyter notebooks for EDA and experimentation
├── src/              # Python source code for data loading and preprocessing
├── results/          # Generated figures, tables, and reports
├── docs/             # Documentation and reports
├── requirements.txt  # Python dependencies
└── README.md         # Project readme
```

## Dataset Requirements
The project uses the **CIC-IDS2017** dataset.
Please refer to `docs/DATASET_DOWNLOAD.md` for manual download instructions if automated downloading is not used. 
To replicate the environment for Milestone 1, the dataset or a sample should be placed in `data/raw/`.

## Environment Setup
Ensure Python 3.10+ is installed.
```bash
pip install -r requirements.txt
```

## How to Run Dataset Audit
```bash
python src/dataset_audit.py
```
This script audits the raw dataset, checks data quality, class distribution, and potential data leakage. The output is reported in the terminal.

## How to Run EDA Notebook
```bash
jupyter notebook notebooks/01_dataset_eda.ipynb
```
The notebook contains comprehensive exploratory data analysis, visualizations, and findings. Important plots are saved in `results/figures/`.

## Current Status FINAL — AML Assignment II Submission
- ✅ Project Scaffolding
- ✅ Dataset Acquisition
- ✅ Dataset Audit
- ✅ Exploratory Data Analysis (EDA)
- ✅ Preprocessing & Feature Engineering
- ✅ Baselines & Deep Learning
- ✅ Leakage Stress Test & Generalization
- ✅ Digital Twin Simulation & Explainability
- ✅ Corrected Temporal Simulation & Temporal Order Ablation
- ✅ CyberTwin Application Prototype (Streamlit)

## Streamlit Application
Run locally from the repository root:
```bash
streamlit run app/streamlit_app.py
```
The application provides:
- Flow-level cyberattack detection
- Temporal BiLSTM detection
- Digital Twin simulation
  - Normal scenario
  - Attack Onset scenario
  - Sustained Attack scenario
  - Attack Recovery scenario
- Random Forest SHAP explainability
- Model performance comparison
- Temporal-order ablation results

> **Note:** The temporal scenarios are controlled/synthetic simulations based on CIC-IDS2017 observations. They are not actual real-world temporal traffic captures, as the necessary chronology metadata was not present in the original subset.

## Streamlit Cloud Deployment
**Repository:** Harish-Raja-R/cybertwin
**Branch:** main
**Main file:** app/streamlit_app.py
The application is designed to run using repository-tracked model and deployment artifacts. No raw CIC-IDS2017 dataset is required for application startup.

## Technology Stack
- **Language:** Python 3
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Machine Learning:** Scikit-learn (Random Forest, Logistic Regression)
- **Deep Learning:** PyTorch (BiLSTM)
- **Explainable AI:** SHAP
- **Application Framework:** Streamlit
- **Visualization:** Matplotlib, Seaborn, Plotly

## Project Status
**FINAL — AML Assignment II Submission**
**Application:** Streamlit
**Entry Point:** app/streamlit_app.py
**Dataset:** CIC-IDS2017
**Final Feature Count:** 70
**Temporal Sequence Length:** 20
