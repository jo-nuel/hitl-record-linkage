# Active Learning Summary

The Active Learning ML Matcher is the main proposed AI + HITL method. It learns from field-level comparison features, selects uncertain pairs for review, receives simulated reviewer labels from FEBRL ground truth for reproducible benchmarking, and retrains in batches.

Formal active-learning experiments use FEBRL ground truth to simulate reviewer labels. This keeps the benchmark reproducible and avoids mixing live Streamlit clicks with the frozen evaluation set. Live clicks remain useful for demonstration, audit logging, and possible future training-label collection.

## Configuration

- Seed positive labels: 50
- Seed negative labels: 50
- Batch size: 150
- Rounds: 8
- Random state: 42
- Frozen test size: 0.25

## Best Active-Learning Round

- Strategy: Active Learning
- Classifier: Random Forest
- Labelled pairs: 700
- Precision: 1.000
- Recall: 0.999
- F1-score: 1.000

## Final Research Comparison

- Main proposed method: AI + HITL Active Learning Matcher
- Precision: 1.000
- Recall: 0.999
- F1-score: 1.000
- Candidate pairs reviewed: 700
- Review workload percentage: 0.464%

## Best Model Comparison Result

- Method: Hybrid EMPI Score
- Precision: 0.998
- Recall: 0.999
- F1-score: 0.998

## Interpretation

The EMPI-inspired pipeline provides the healthcare-style record linkage structure: preprocessing, blocking, field-level comparison, threshold decisions, and human review.

The Hybrid EMPI Score is retained as a transparent non-ML baseline and fallback scoring method. It is not presented as the main AI model.

The Active Learning ML Matcher is the main proposed method because it uses reviewer labels to improve future predictions over training rounds. Random Sampling HITL is included as a baseline to show whether uncertainty sampling is more label-efficient than reviewing randomly selected pairs.
