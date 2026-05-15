# Hyperparameter Tuning Summary

The active-learning experiment tunes Logistic Regression, Random Forest, and Gradient Boosting with `GridSearchCV`. Tuning uses the initial seed labels only, so the frozen test set remains reserved for evaluation.

## Best Tuned Classifier

- Method: Logistic Regression
- Best CV F1-score: 0.990
- Best parameters: `{'C': 10.0}`

## Full Tuning Table

```
             Method  Best CV F1-score                                                                     Best parameters  Tuning runtime seconds                                                                              Selection note
Logistic Regression          0.989899                                                                         {'C': 10.0}                0.104728 Tuned on active-learning seed labels only; frozen test set is not used for model selection.
      Random Forest          0.989899 {'max_depth': 4, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'n_estimators': 60}                6.755625 Tuned on active-learning seed labels only; frozen test set is not used for model selection.
  Gradient Boosting          0.971380                         {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 50}                1.128549 Tuned on active-learning seed labels only; frozen test set is not used for model selection.
```
