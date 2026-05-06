# Active Learning Summary

Formal active-learning experiments use FEBRL ground truth to simulate reviewer labels. This keeps the benchmark reproducible and avoids mixing live Streamlit clicks with the frozen evaluation set.

## Configuration

- Seed positive labels: 50
- Seed negative labels: 150
- Batch size: 150
- Rounds: 8
- Random state: 42
- Frozen test size: 0.25

## Best Active-Learning Round

- Strategy: Active Learning
- Classifier: Logistic Regression
- Labelled pairs: 800
- Precision: 0.998
- Recall: 0.995
- F1-score: 0.996

## Best Model Comparison Result

- Method: Hybrid EMPI Score
- Precision: 0.998
- Recall: 0.999
- F1-score: 0.998

## Interpretation

Active learning selects uncertain pairs near the classifier decision boundary for review. In this prototype, FEBRL labels simulate reviewer feedback so the experiment can be rerun consistently. Live review decisions remain useful for demonstrating audit logging and future training-label collection, but they are not used as the default benchmark labels.
