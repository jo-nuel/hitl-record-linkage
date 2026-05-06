# Active Learning Summary

Formal active-learning experiments use FEBRL ground truth to simulate reviewer labels. This keeps the benchmark reproducible and avoids mixing live Streamlit clicks with the frozen evaluation set.

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

## Best Model Comparison Result

- Method: Hybrid EMPI Score
- Precision: 0.998
- Recall: 0.999
- F1-score: 0.998

## Interpretation

Active learning selects uncertain pairs near the classifier decision boundary for review. In this prototype, FEBRL labels simulate reviewer feedback so the experiment can be rerun consistently. Live review decisions remain useful for demonstrating audit logging and future training-label collection, but they are not used as the default benchmark labels.

The Hybrid EMPI Score is kept as a transparent non-ML baseline and fallback scoring method. The active-learning ML matcher is the main AI extension because it learns from field-level comparison features and simulated reviewer labels, then retrains in batches.
