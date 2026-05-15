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

## Hyperparameter Tuning

Model hyperparameters are tuned with `GridSearchCV` on the initial active-learning seed labels only. The frozen test set is not used for tuning or model selection. Tuning evidence is saved to `outputs\tables\hyperparameter_tuning.csv`.

## Best Active-Learning Round

- Strategy: Active Learning
- Classifier: Logistic Regression
- Labelled pairs: 550
- Precision: 0.997
- Recall: 0.996
- F1-score: 0.996

## Final Research Comparison

- Main proposed method: AI + HITL Active Learning Matcher
- Precision: 0.997
- Recall: 0.996
- F1-score: 0.996
- Candidate pairs reviewed: 550
- Review workload percentage: 0.364%

## Selected Tuned ML Classifier

- Method: Logistic Regression
- Best CV F1-score: 0.990
- Frozen test precision: 0.993
- Frozen test recall: 0.994
- Frozen test F1-score: 0.993

## Interpretation

The final method uses FEBRL4, preprocessing, blocking, field-level comparison features, ML match probability, uncertainty sampling, simulated reviewer labels, and batch retraining.

Uncertain pairs are selected for review because their predicted match probability is close to 0.5. The simulated reviewer label is then added to the training set before the next round.

The final report-facing evaluation is limited to Human-only Clerical Review Baseline, AI-only ML Matcher, and AI + HITL Active Learning Matcher.
