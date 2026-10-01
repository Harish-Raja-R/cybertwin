# CyberTwin Digital Twin Design Document

## Virtual Entities
The system simulates network nodes representing abstract monitored systems (e.g., `node_001`, `node_002`). Because the raw dataset (`Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`) omits IP addresses, these nodes act as synthetic aggregators for temporal traffic segments rather than distinct physical hosts.

## Network State
Each virtual node manages a local state that transitions dynamically. A higher-level `DigitalTwinNetwork` observes these nodes and computes a holistic global risk metric.

## Traffic State
Nodes maintain a bounded historical queue (window length = 50) tracking instantaneous `traffic_intensity` (approximated via total feature magnitude in the absence of explicit byte rates) and the predicted `attack_probability` derived from the Temporal BiLSTM model.

## Attack State Transitions
Nodes progress through four states based on hard thresholds mapped to probability and risk:
1. `NORMAL`: The baseline operational state.
2. `SUSPICIOUS`: Triggered if the current attack probability exceeds 50% or if the cumulative risk score surpasses 0.4.
3. `UNDER_ATTACK`: Triggered if the attack probability spikes past 80% or risk exceeds 0.7.
4. `RECOVERING`: A cool-down state entered from `UNDER_ATTACK` when the attack probability drops below 20% and the risk normalizes below 0.5. Requires further risk reduction (<0.3) to return to `NORMAL`.

## Risk Score
The composite risk score translates instantaneous model outputs and historical context into an actionable continuous metric (0.0 to 1.0):
```python
Risk Score = (0.5 * Attack Probability) + 
             (0.3 * Suspicious Ratio) + 
             (0.2 * Normalized Traffic Intensity)
```
- **Attack Probability**: Directly supplied by the Temporal BiLSTM for the current sequence.
- **Suspicious Ratio**: The proportion of recent predictions (last 50 windows) that exceeded the 0.5 probability threshold.
- **Normalized Traffic Intensity**: A heuristic metric representing recent volume.

*Note: The weights are simulation parameters rather than learned physical-system parameters.*

## Limitations
The Digital Twin operates in a strictly simulated chronological regime. Real-world physical properties such as network topology, routing delays, and causally-linked concurrent flows from identical physical endpoints are inherently stripped due to the limitations of the anonymized CIC-IDS2017 feature set slice.
