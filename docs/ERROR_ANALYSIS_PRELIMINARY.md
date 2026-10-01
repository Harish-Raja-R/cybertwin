# Preliminary Error Analysis (Milestone 3)

## Overview
This document analyzes the misclassifications made by the final 1D-CNN + BiLSTM deep learning model on the test set.

## Error Types
- **False Positives (FP)**: The model predicted `PortScan` (1) when the true label was `BENIGN` (0).
- **False Negatives (FN)**: The model predicted `BENIGN` (0) when the true label was `PortScan` (1).

## Error Statistics (CNN + BiLSTM)
- **False Positives**: ~16,300
- **False Negatives**: ~10
- **Overall Pattern**: The model exhibits extremely high recall (99.9%) but suffers from very low precision (45%).

## Discussion
1. **What types of flows are misclassified?**
   - The deep learning model, given only 6 epochs to train from scratch on CPU, struggled to differentiate effectively and resorted to predicting a soft probability (around `0.52`) for a large portion of the `BENIGN` dataset. Because the threshold is `0.5`, this resulted in a massive surge of False Positives.
   
2. **Are false positives concentrated in BENIGN traffic?**
   - Yes, almost entirely. Over 16,000 benign traffic samples were flagged as PortScan by the deep model.
   - Conversely, Random Forest correctly filtered out almost all BENIGN traffic without issue, suggesting that deep models on tabular data require extensive tuning (or embedding architectures) to match tree-based splits.

3. **Are false negatives concentrated in PortScan?**
   - No, the model accurately detected almost every PortScan (only ~10 false negatives). The high class weight used in the `BCEWithLogitsLoss` penalized FN so heavily that the network preferred to guess `1` when unsure.

4. **Are errors associated with unusual feature patterns?**
   - The error distributions indicate that deep learning models are sensitive to scaling. While robust scaling was applied, tabular network flow features lack the spatial or temporal sequential relationships that CNNs and RNNs typically leverage. The BiLSTM treats the 70 features as a sequence, which is conceptually arbitrary compared to actual packet payloads.

5. **Limitations**
   - The dataset currently contains only `BENIGN` and `PortScan` traffic. The PortScan signatures are heavily rigid, making Random Forest artificially perfect.
   - The advanced model's architecture (1D-CNN + BiLSTM) is currently applied to isolated *flows*. In the Digital Twin implementation (Milestone 4+), providing sequential flows (e.g., passing a time window of flows instead of single tabular vectors) would allow the BiLSTM to capture actual temporal attacks, drastically reducing False Positives.

## Conclusion
The advanced deep learning model is highly sensitive (zero-day detection potential) but suffers from high false alarms in its current flow-level state. While Random Forest achieves an F1 score of 0.9999, the CNN+BiLSTM is foundational for the upcoming Digital Twin integration where sequence modeling over multiple flows will provide actual defensive context.
