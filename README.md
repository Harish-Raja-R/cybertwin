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

## Current Status
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

## Technology Stack
- **Language:** Python 3
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Machine Learning:** Scikit-learn (Random Forest, Logistic Regression)
- **Deep Learning:** PyTorch (BiLSTM)
- **Explainable AI:** SHAP
- **Application Framework:** Streamlit
- **Visualization:** Matplotlib, Seaborn, Plotly

## Milestone 2 Status: COMPLETED
- Designed and executed robust data cleaning pipeline.
- Handled data leakage via strict feature selection.
- Implemented standard Train/Val/Test splitting (70/15/15).
- Stored fitted preprocessing components for reproducibility.



## Experimental Validation (Milestone 3.5)
To ensure the robustness of the highly performant machine learning models (scoring F1=0.9998), rigorous generalization audits were performed:
* **Random stratified benchmark** over a completely untouched 15% test set.
* **5-Fold Cross-validation** exclusively on the training subset.
* **Feature ablation** simulating corrupted/dropped data segments (which proved model resilience).
* **Distribution analysis** confirming zero dataset-shift between splits.
* **Generalization audit** detailing the data properties (including the limitation that a chronological validation was not possible due to omitted timestamps in the raw CSV).

## Temporal Validation (Milestone 4.1)
The final milestone implements the Digital Twin simulation. Since the original CIC-IDS2017 historical chronology was unavailable in the provided CSV slice, a **controlled synthetic temporal sequence generator** was created to simulate real-world cyberattack phases: Normal, Attack Onset, Sustained Attack, and Recovery.

**Temporal Order Ablation:**
To rigorously validate that the trained Temporal BiLSTM is actively learning sequential dynamics and not just memorizing the proportion of malicious flows in a window, a temporal-order ablation experiment was performed.
- The model achieved perfect metrics (F1 = 1.000) on structured temporal sequences.
- When the sequence order of the exact same flows was randomly shuffled, the model's performance on transition states (Attack Onset and Recovery) degraded significantly (e.g., Recovery accuracy dropped to 14%). 
- **Conclusion:** This provides strong evidence that the temporal model explicitly utilizes sequential ordering to accurately detect network state transitions, reinforcing its applicability as a stateful Digital Twin detector.

**Limitations:** 
The scenarios are synthetically generated from strictly partitioned independent tabular flows to overcome dataset limitations. While they represent valid mathematical transition sequences, they are not historical timelines.

## Milestone 5: CyberTwin Streamlit Application Prototype
An interactive dashboard built with Streamlit has been implemented in `app/streamlit_app.py`.
It serves as a working prototype demonstrating:
1. Side-by-side detection (Static vs Temporal).
2. A Digital Twin simulation showing network state transitions.
3. Model evaluation and temporal ablation results.
4. Explainable AI outputs (SHAP).

To run the application locally:
```bash
streamlit run app/streamlit_app.py
```
See `docs/APPLICATION_REPORT.md` and `docs/DEMO_GUIDE.md` for more details.
