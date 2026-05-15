# Methodology Summary

The final method is AI-assisted active learning record linkage using FEBRL4.

The workflow loads FEBRL4, preprocesses identity fields, creates candidate pairs with multi-pass blocking, compares records using field-level similarity features, trains ML classifiers, predicts match probability, selects uncertain pairs near p(match)=0.5 for simulated review, and retrains the classifier in batches.

The final report-facing comparison is limited to:

- Human-only Clerical Review Baseline
- AI-only ML Matcher
- AI + HITL Active Learning Matcher
