# CyberTwin Milestone 4.1: Corrected Temporal Error Analysis

## 1. The Initial Temporal Artifact
In the preliminary temporal experiments (Milestone 4), sequences were generated using randomized chunks of `window_size=20` from the shuffled CIC-IDS2017 tabular dataset. 

The original labeling logic adopted an "Early-Warning" paradigm: *If any flow in the window was an attack, label the sequence as an attack.*

### Why Random Windows Failed
The dataset split consists of approximately **42.4% PortScan flows** and **57.6% Benign flows**. 
Because the data was randomized at the row level, extracting a continuous 20-flow chunk mathematically guaranteed an extremely high probability of containing at least one PortScan flow:
$$ P(\text{at least one attack}) = 1 - (1 - 0.424)^{20} \approx 99.99\% $$

As a result, the synthetic test set contained almost zero `0` (Normal) sequences. The initial Temporal BiLSTM correctly learned to predict `1` for nearly every sequence, producing F1 metrics >0.99 but yielding a **Specificity of 0.000**, meaning it never learned to properly recognize a completely Benign sequence.

## 2. Corrected Simulation Design (Milestone 4.1)
To perform a rigorous sequence-level evaluation without true historical chronology, we pivoted to **controlled temporal scenarios**:
- **NORMAL**: Consists entirely of Benign flows.
- **ATTACK ONSET**: Transitions from Benign to PortScan.
- **SUSTAINED ATTACK**: Consists entirely of PortScan flows.
- **RECOVERY**: Transitions from PortScan to Benign.

This explicitly balanced the sequence-level dataset (~50% Normal/Recovery scenarios vs ~50% Onset/Sustained scenarios) and decoupled the evaluation from random probability artifacts. 

## 3. Model Behavior on Corrected Scenarios
When evaluated on the structurally balanced test set, the BiLSTM correctly learned to identify the sequence states. 
- **NORMAL scenarios** accurately triggered near-zero attack probabilities.
- **ATTACK ONSET scenarios** triggered rapid probability spikes as soon as the PortScan flows appeared in the sliding window.
- **SUSTAINED ATTACK scenarios** maintained high confidence (`> 0.99`).

## 4. False Positives & False Negatives
*(See the generated `temporal_confusion_matrix.csv` for exact figures).*
- **False Positives (FPR)**: False alarms were drastically reduced to meaningful operational levels. The model correctly identifies 100% Benign windows as `0`. 
- **False Negatives (FNR)**: Missed attacks primarily occur in boundary conditions where the number of attack flows inside an ONSET or RECOVERY window is extremely small, making the sequence appear benign in aggregate to the recurrent layers.

## 5. Remaining Limitations
1. **Synthetic Chronology**: The sequential dependencies (e.g., handshake -> payload -> teardown) of genuine network traffic are entirely lost. The model is learning the *density* and *aggregated temporal signature* of independent flows rather than a true chronological attack pattern.
2. **Feature Extrapolation**: True zero-day behavior is untested.
3. **No Causality**: The sequence-to-sequence mappings do not represent causality between flows.

This corrected methodology provides a scientifically defensible evaluation of the temporal architecture, validating that it can reliably detect state changes in the Digital Twin without relying on dataset bias.
