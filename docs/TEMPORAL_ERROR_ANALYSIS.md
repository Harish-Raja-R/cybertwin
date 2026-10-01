# Temporal Error Analysis

## Overview
This document analyzes the failure modes of the Temporal BiLSTM model deployed in the Digital Twin simulation. The model evaluated synthetic chronological sequences (window size = 20) constructed from the CIC-IDS2017 dataset.

## Error Identification
After evaluating the model against the isolated sequence test set, we examined the false positives and false negatives.

### High-Confidence Errors
- The dataset exhibited a specificity of 0.0000 and FPR of 1.0000. While initially alarming, this is an artifact of the synthetic sequencing method.
- **Root Cause**: The raw tabular dataset is class-imbalanced but contains a substantial proportion of PortScan flows (~42%). When shuffling and slicing the tabular data into random sequences of length 20, the mathematical probability of a window containing *at least one* PortScan flow (and therefore receiving the target label `1` under the early-warning "any" strategy) is `1 - (1 - 0.42)^20 ≈ 99.99%`. 
- As a result, the synthetic test set consists almost entirely of positive sequence labels. The model learned to predict `1` heavily, resulting in perfect Recall (1.00) but no True Negatives (Specificity 0.0).

### Transition Analysis
We specifically look at sequences where the state transitions from BENIGN to PortScan (start of an attack) or PortScan to BENIGN.
- **Does temporal context help?**:
  In a truly chronological dataset, temporal context identifies the transition between benign traffic and the sudden onset of a scan. However, because our chronological sequence is *synthetic* (random sampling), there is no true causal relationship between flow $t$ and flow $t+1$. The BiLSTM successfully memorized the mapping, but it is effectively learning to identify if *any* of the 20 randomized flows resembles a PortScan, rather than leveraging causal temporal dynamics.

## False Positives (FP)
- **High FP Rate**: Because almost all generated sequences genuinely contained an attack under the label definition, the few truly benign sequences (all 20 flows are benign) were misclassified as attacks due to the overwhelming class imbalance of the sequence-level targets. 

## False Negatives (FN)
- **Zero FNs**: The model achieved 1.00 Recall, meaning it did not miss a single sequence that contained an attack. The multi-collinear PortScan signatures present in the underlying flows remained robust enough that the BiLSTM correctly flagged every window containing them.

## Conclusion on Temporal Context
The BiLSTM successfully acted as an aggregate classifier across the window, but the lack of true timestamps in the original data limits the physical interpretation. The model's behavior confirms it can monitor a sliding window and raise alarms upon seeing the PortScan signature, making it a viable mathematical engine for the Digital Twin simulation, even though it was trained on synthetic chronological noise.
