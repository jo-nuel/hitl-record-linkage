# Evaluation Summary

The final report-facing evaluation uses three methods:

- Human-only Clerical Review Baseline
- AI-only ML Matcher
- AI + HITL Active Learning Matcher

The human-only baseline estimates the workload of reviewing all blocked candidate pairs. The ML methods are evaluated on a frozen test set. The AI + HITL method selects uncertain pairs using model uncertainty, simulates reviewer labels with FEBRL4 ground truth, adds those labels to the training data, and retrains in batches.

Formal benchmark labels are simulated for reproducibility. Live Streamlit review decisions are stored for demonstration and audit logging only.
