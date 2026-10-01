# CIC-IDS2017 Dataset Download Instructions

## Official Source
The official dataset is provided by the Canadian Institute for Cybersecurity (CIC) at the University of New Brunswick (UNB).

**URL/Reference**: [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)

**Dataset Name/Version**: Intrusion Detection Evaluation Dataset (CIC-IDS2017)

## Manual Download Instructions
1. Navigate to the official URL provided above.
2. Click on the "Download Dataset" link (may require filling out a form to accept terms of use for academic/research purposes).
3. Download the `MachineLearningCSV.zip` file, which contains the pre-processed flow features extracted by CICFlowMeter.
4. Extract the ZIP archive.

## Expected Files (MachineLearningCSV)
You should expect the following files representing different days of the week and attack types:
- `Monday-WorkingHours.pcap_ISCX.csv` (Normal traffic)
- `Tuesday-WorkingHours.pcap_ISCX.csv` (FTP-Patator, SSH-Patator)
- `Wednesday-workingHours.pcap_ISCX.csv` (DoS attacks, Heartbleed)
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` (Web attacks)
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` (Infiltration)
- `Friday-WorkingHours-Morning.pcap_ISCX.csv` (Bot)
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` (PortScan)
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` (DDoS)

## Expected Directory Structure
Place the extracted CSV files into the `data/raw/` directory of the CyberTwin project.

```
CyberTwin/
├── data/
│   ├── raw/
│   │   ├── Monday-WorkingHours.pcap_ISCX.csv
│   │   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   │   └── ... (other CSVs)
```

## Note for Automation
Due to authentication/form requirements on the official site, direct automated downloads are not officially supported. Mirrors exist on Kaggle and Hugging Face (e.g., `rdpahalavan/CIC-IDS2017`), which can be used for automated sampling or testing.
