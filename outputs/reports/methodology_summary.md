# Methodology Summary

The final method is AI-Assisted Active Learning HITL Record Linkage using FEBRL4. The workflow loads FEBRL4, preprocesses identity fields, creates candidate pairs with multi-pass blocking, compares fields with Jaro-Winkler and exact agreement, trains ML classifiers on field-level evidence, selects uncertain pairs for simulated professional review, and retrains the classifier in batches. The Hybrid EMPI-style score is retained as a transparent non-ML baseline and fallback.
