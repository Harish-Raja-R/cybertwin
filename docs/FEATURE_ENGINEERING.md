# Feature Engineering Log

## 1. Total Packets
- **Formula**: `Total Fwd Packets + Total Backward Packets`
- **Purpose**: Gives the overall packet size of the flow instead of just directional counts.
- **Original features used**: `Total Fwd Packets`, `Total Backward Packets`
- **Leakage Risk**: LOW. Based purely on observed flow behavior.

## 2. Forward to Backward Packet Ratio
- **Formula**: `Total Fwd Packets / (Total Backward Packets + 1e-6)`
- **Purpose**: Helps distinguish highly asymmetrical flows (e.g., small request leading to large download, typical in normal traffic but often inverted or highly skewed in attacks).
- **Original features used**: `Total Fwd Packets`, `Total Backward Packets`
- **Leakage Risk**: LOW. Based purely on observed flow behavior.

## 3. Forward to Backward Byte Ratio
- **Formula**: `Total Length of Fwd Packets / (Total Length of Bwd Packets + 1e-6)`
- **Purpose**: Represents data transfer asymmetry. Malicious payloads often show distinct asymmetry compared to benign video streaming or web browsing.
- **Original features used**: `Total Length of Fwd Packets`, `Total Length of Bwd Packets`
- **Leakage Risk**: LOW. Based purely on observed flow behavior.

## Additional Notes
We avoided engineering dozens of arbitrary features and relied primarily on CICFlowMeter's robust extracted network features. The target for this milestone was establishing the pipeline; we can expand this list heavily later when tuning the CyberTwin model if needed.
