# ML Match Probability Summary

The final method trains lightweight supervised classifiers on field-level comparison features from FEBRL4 candidate pairs.

## Features

The model uses fields such as:

- given name similarity
- surname similarity
- date-of-birth agreement
- address similarity
- suburb or place similarity
- state agreement
- postcode agreement
- sex or gender agreement where available

## Model Selection

Logistic Regression, Random Forest, and Gradient Boosting are tuned with `GridSearchCV` on the initial active-learning seed labels. The frozen test set is reserved for evaluation.

## Review Selection

The model outputs an ML match probability. Active learning selects uncertain pairs whose predicted probability is closest to 0.5. Simulated reviewer labels are then added to the training data before the next retraining round.

This keeps the final method focused on field-level evidence, ML match probability, uncertainty sampling, simulated reviewer labels, and batch retraining.
