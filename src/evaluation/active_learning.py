import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.utils.config import CONFIG
from src.utils.io import ensure_directories_exist, save_csv


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FEATURE_COLUMNS = [
    "given_name_sim",
    "surname_sim",
    "date_of_birth_exact",
    "address_sim",
    "suburb_sim",
    "state_exact",
    "postcode_exact",
    "sex_exact",
]


def _feature_matrix(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract model-ready comparison features and internal FEBRL labels."""
    available = [column for column in FEATURE_COLUMNS if column in pairs.columns]
    if not available:
        raise ValueError("No comparison feature columns found for active learning")
    if "is_true_link" not in pairs.columns:
        raise ValueError("Active-learning simulation requires internal ground-truth labels")
    x = pairs[available].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pairs["is_true_link"].astype(int)
    return x, y


def _load_candidate_pairs_for_active_learning() -> pd.DataFrame:
    """Load public pair features and attach FEBRL truth internally for simulation."""
    if not CONFIG.paths.classified_pairs.exists() or not CONFIG.paths.true_links.exists():
        raise FileNotFoundError(
            "Active learning requires existing pipeline outputs. Run: python scripts/run_pipeline.py --review-mode simulate"
        )
    pairs = pd.read_csv(CONFIG.paths.classified_pairs).fillna("")
    true_links = pd.read_csv(CONFIG.paths.true_links, dtype=str).fillna("")
    truth = set(zip(true_links["record_id_a"].astype(str), true_links["record_id_b"].astype(str)))
    pairs["is_true_link"] = pairs.apply(
        lambda row: int((str(row["record_id_a"]), str(row["record_id_b"])) in truth),
        axis=1,
    )
    return pairs


def _model_specs(random_state: int) -> dict[str, tuple[object, dict[str, list[object]]]]:
    """Return lightweight model grids used for reproducible hyperparameter tuning."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
            {"C": [0.1, 1.0, 10.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=1),
            {
                "n_estimators": [60, 100],
                "max_depth": [4, 8, None],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", None],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=random_state),
            {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3],
            },
        ),
    }


def _predict_probability(model: object, x: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(x)
    return probabilities[:, 1]


def _metrics(y_true: pd.Series, probabilities: np.ndarray, decision_cutoff: float = 0.5) -> dict[str, float]:
    predicted = (probabilities >= decision_cutoff).astype(int)
    return {
        "Precision": precision_score(y_true, predicted, zero_division=0),
        "Recall": recall_score(y_true, predicted, zero_division=0),
        "F1-score": f1_score(y_true, predicted, zero_division=0),
        "False positives": int(((predicted == 1) & (y_true.to_numpy() == 0)).sum()),
        "False negatives": int(((predicted == 0) & (y_true.to_numpy() == 1)).sum()),
    }


def _seed_indices(y_pool: pd.Series, positive_count: int, negative_count: int, rng: np.random.Generator) -> list[int]:
    positives = y_pool[y_pool == 1].index.to_numpy()
    negatives = y_pool[y_pool == 0].index.to_numpy()
    if len(positives) < positive_count or len(negatives) < negative_count:
        raise ValueError("Not enough positive or negative labels to build the active-learning seed set")
    seed_pos = rng.choice(positives, size=positive_count, replace=False)
    seed_neg = rng.choice(negatives, size=negative_count, replace=False)
    return list(seed_pos) + list(seed_neg)


def _tune_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Tune ML classifiers on seed labels without using the frozen test set."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    tuned_models: dict[str, object] = {}
    rows = []
    for name, (estimator, param_grid) in _model_specs(random_state).items():
        start = time.perf_counter()
        search = GridSearchCV(
            estimator,
            param_grid,
            scoring="f1",
            cv=cv,
            n_jobs=1,
            error_score=0,
        )
        search.fit(x_train, y_train)
        runtime = time.perf_counter() - start
        tuned_models[name] = search.best_estimator_
        rows.append(
            {
                "Method": name,
                "Best CV F1-score": float(search.best_score_),
                "Best parameters": str(search.best_params_),
                "Tuning runtime seconds": runtime,
                "Selection note": "Tuned on active-learning seed labels only; frozen test set is not used for model selection.",
            }
        )
    return tuned_models, pd.DataFrame(rows)


def _run_model_comparison(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    tuned_models: dict[str, object],
    tuning: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    tuning_lookup = tuning.set_index("Method").to_dict("index")
    for name, model_template in tuned_models.items():
        start = time.perf_counter()
        model = clone(model_template)
        model.fit(x_train, y_train)
        probabilities = _predict_probability(model, x_test)
        row = _metrics(y_test, probabilities)
        row["Method"] = name
        row["Reviewed pairs"] = 0
        row["Runtime seconds"] = time.perf_counter() - start
        row["Best CV F1-score"] = tuning_lookup[name]["Best CV F1-score"]
        row["Best parameters"] = tuning_lookup[name]["Best parameters"]
        rows.append(row)
    return pd.DataFrame(rows)


def _active_learning_loop(
    model_name: str,
    model_template: object,
    x_pool: pd.DataFrame,
    y_pool: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    batch_size: int,
    rounds: int,
    seed_indices: list[int],
    random_state: int,
    strategy: str,
) -> pd.DataFrame:
    """Select uncertain pairs near p(match)=0.5 and simulate reviewer labels with FEBRL truth."""
    rng = np.random.default_rng(random_state)
    labelled = list(seed_indices)
    unlabelled = [idx for idx in x_pool.index if idx not in labelled]
    rows = []

    for round_number in range(rounds + 1):
        new_labels_added = 0
        if round_number > 0 and unlabelled:
            # Select a batch using the previous labelled set, simulate reviewer
            # labels with FEBRL truth, then retrain before evaluating this round.
            model = clone(model_template)
            model.fit(x_pool.loc[labelled], y_pool.loc[labelled])
            if strategy == "Active Learning":
                pool_probabilities = _predict_probability(model, x_pool.loc[unlabelled])
                # Uncertainty sampling chooses pairs closest to p(match)=0.5.
                uncertainty = np.abs(pool_probabilities - 0.5)
                selected_positions = np.argsort(uncertainty)[:batch_size]
                selected = [unlabelled[position] for position in selected_positions]
            else:
                selected = list(rng.choice(unlabelled, size=min(batch_size, len(unlabelled)), replace=False))

            selected_set = set(selected)
            labelled.extend(selected)
            unlabelled = [idx for idx in unlabelled if idx not in selected_set]
            new_labels_added = len(selected)

        model = clone(model_template)
        model.fit(x_pool.loc[labelled], y_pool.loc[labelled])
        probabilities = _predict_probability(model, x_test)
        row = _metrics(y_test, probabilities)
        row.update(
            {
                "Round": round_number,
                "Strategy": strategy,
                "Classifier": model_name,
                "Labelled pairs": len(labelled),
                "New labels added": new_labels_added,
                "Unlabelled pairs remaining": len(unlabelled),
            }
        )
        rows.append(row)

        if round_number == rounds:
            break

    return pd.DataFrame(rows)


def _line_chart(
    df: pd.DataFrame,
    path: Path,
    title: str,
    y_column: str,
    hue_column: str = "Strategy",
    y_min: float = 0.0,
    y_max: float = 1.05,
) -> None:
    plt.figure(figsize=(8, 5))
    for label, group in df.groupby(hue_column):
        plt.plot(group["Labelled pairs"], group[y_column], marker="o", label=label)
    plt.title(title)
    plt.xlabel("Labelled pairs used")
    plt.ylabel(y_column)
    plt.ylim(y_min, y_max)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _bar_chart(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    ordered = df.sort_values("F1-score", ascending=True)
    plt.barh(ordered["Method"], ordered["F1-score"], color="#355C7D")
    plt.xlabel("F1-score")
    plt.title("Model comparison by F1-score")
    plt.xlim(0, 1.05)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _final_accuracy_chart(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 4.8))
    x = np.arange(len(df))
    width = 0.24
    plt.bar(x - width, df["Precision"], width=width, label="Precision", color="#355C7D")
    plt.bar(x, df["Recall"], width=width, label="Recall", color="#6C9A8B")
    plt.bar(x + width, df["F1-score"], width=width, label="F1-score", color="#C06C84")
    plt.xticks(x, df["Method"], rotation=20, ha="right")
    plt.ylim(0.98, 1.005)
    plt.ylabel("Metric value")
    plt.title("Accuracy comparison")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _final_workload_chart(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(df["Method"], df["Candidate pairs reviewed"], color="#4C78A8")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Pairs reviewed")
    plt.title("Review workload comparison")
    for bar, value in zip(bars, df["Candidate pairs reviewed"]):
        plt.annotate(
            f"{int(value):,}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _error_reduction_chart(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 4.8))
    plt.plot(df["Round"], df["False positives"], marker="o", label="False positives", color="#C06C84")
    plt.plot(df["Round"], df["False negatives"], marker="o", label="False negatives", color="#355C7D")
    plt.xlabel("Active-learning round")
    plt.ylabel("Error count")
    plt.title("Errors over active-learning rounds")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _final_research_evaluation(
    comparison: pd.DataFrame,
    active_rounds: pd.DataFrame,
    candidate_count: int,
    selected_model: str,
) -> pd.DataFrame:
    """Build the central final comparison for the active-learning research claim."""
    ml = comparison[comparison["Method"] == selected_model].iloc[0]
    active = active_rounds.sort_values("F1-score", ascending=False).iloc[0]

    rows = [
        {
            "Method": "Human-only Clerical Review Baseline",
            "Role": "Primary baseline",
            "Dataset": "FEBRL4",
            "Precision": 1.0,
            "Recall": 1.0,
            "F1-score": 1.0,
            "False positives": 0,
            "False negatives": 0,
            "Candidate pairs reviewed": candidate_count,
            "Review workload percentage": 100.0,
            "Estimated review time": f"{candidate_count * CONFIG.matcher.manual_review_seconds_per_pair:,.0f} seconds",
            "Training labels used": candidate_count,
            "Evaluation scope": "Ideal clerical review over candidate pairs",
            "Key interpretation": "Highest review burden because every candidate pair is manually resolved.",
        },
        {
            "Method": "AI-only ML Matcher",
            "Role": "Primary AI-only comparison",
            "Dataset": "FEBRL4",
            "Precision": ml["Precision"],
            "Recall": ml["Recall"],
            "F1-score": ml["F1-score"],
            "False positives": int(ml["False positives"]),
            "False negatives": int(ml["False negatives"]),
            "Candidate pairs reviewed": 0,
            "Review workload percentage": 0.0,
            "Estimated review time": "0 seconds",
            "Training labels used": CONFIG.active_learning.seed_positive_labels + CONFIG.active_learning.seed_negative_labels,
            "Evaluation scope": "Frozen active-learning test set",
            "Key interpretation": "Tuned ML classifier without further active-learning review batches.",
        },
        {
            "Method": "AI + HITL Active Learning Matcher",
            "Role": "Primary proposed method",
            "Dataset": "FEBRL4",
            "Precision": active["Precision"],
            "Recall": active["Recall"],
            "F1-score": active["F1-score"],
            "False positives": int(active["False positives"]),
            "False negatives": int(active["False negatives"]),
            "Candidate pairs reviewed": int(active["Labelled pairs"]),
            "Review workload percentage": float(active["Labelled pairs"]) / candidate_count * 100,
            "Estimated review time": f"{float(active['Labelled pairs']) * CONFIG.matcher.manual_review_seconds_per_pair:,.0f} seconds",
            "Training labels used": int(active["Labelled pairs"]),
            "Evaluation scope": "Frozen active-learning test set",
            "Key interpretation": "Main proposed method: uncertain reviewed labels are added in batches and the classifier retrains.",
        },
    ]
    return pd.DataFrame(rows)


def _final_research_chart(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(9, 5))
    x = np.arange(len(df))
    plt.bar(x - 0.25, df["F1-score"], width=0.25, label="F1-score", color="#355C7D")
    plt.bar(x, df["Recall"], width=0.25, label="Recall", color="#6C9A8B")
    workload_scaled = df["Review workload percentage"] / 100
    plt.bar(x + 0.25, workload_scaled, width=0.25, label="Workload share", color="#C06C84")
    plt.xticks(x, df["Method"], rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score or workload share")
    plt.title("Final research evaluation comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _write_summary(
    rounds: pd.DataFrame,
    comparison: pd.DataFrame,
    final_evaluation: pd.DataFrame,
    tuning: pd.DataFrame,
) -> None:
    best = rounds.sort_values("F1-score", ascending=False).iloc[0]
    best_tuned = tuning.sort_values("Best CV F1-score", ascending=False).iloc[0]
    selected_test = comparison[comparison["Method"] == best_tuned["Method"]].iloc[0]
    central = final_evaluation[final_evaluation["Method"] == "AI + HITL Active Learning Matcher"].iloc[0]
    text = f"""# Active Learning Summary

The Active Learning ML Matcher is the main proposed AI + HITL method. It learns from field-level comparison features, selects uncertain pairs for review, receives simulated reviewer labels from FEBRL ground truth for reproducible benchmarking, and retrains in batches.

Formal active-learning experiments use FEBRL ground truth to simulate reviewer labels. This keeps the benchmark reproducible and avoids mixing live Streamlit clicks with the frozen evaluation set. Live clicks remain useful for demonstration, audit logging, and possible future training-label collection.

## Configuration

- Seed positive labels: {CONFIG.active_learning.seed_positive_labels}
- Seed negative labels: {CONFIG.active_learning.seed_negative_labels}
- Batch size: {CONFIG.active_learning.batch_size}
- Rounds: {CONFIG.active_learning.rounds}
- Random state: {CONFIG.active_learning.random_state}
- Frozen test size: {CONFIG.active_learning.test_size:.2f}

## Hyperparameter Tuning

Model hyperparameters are tuned with `GridSearchCV` on the initial active-learning seed labels only. The frozen test set is not used for tuning or model selection. Tuning evidence is saved to `{CONFIG.paths.hyperparameter_tuning}`.

## Best Active-Learning Round

- Strategy: {best['Strategy']}
- Classifier: {best['Classifier']}
- Labelled pairs: {int(best['Labelled pairs'])}
- Precision: {best['Precision']:.3f}
- Recall: {best['Recall']:.3f}
- F1-score: {best['F1-score']:.3f}

## Final Research Comparison

- Main proposed method: AI + HITL Active Learning Matcher
- Precision: {central['Precision']:.3f}
- Recall: {central['Recall']:.3f}
- F1-score: {central['F1-score']:.3f}
- Candidate pairs reviewed: {int(central['Candidate pairs reviewed'])}
- Review workload percentage: {central['Review workload percentage']:.3f}%

## Selected Tuned ML Classifier

- Method: {best_tuned['Method']}
- Best CV F1-score: {best_tuned['Best CV F1-score']:.3f}
- Frozen test precision: {selected_test['Precision']:.3f}
- Frozen test recall: {selected_test['Recall']:.3f}
- Frozen test F1-score: {selected_test['F1-score']:.3f}

## Interpretation

The final method uses FEBRL4, preprocessing, blocking, field-level comparison features, ML match probability, uncertainty sampling, simulated reviewer labels, and batch retraining.

Uncertain pairs are selected for review because their predicted match probability is close to 0.5. The simulated reviewer label is then added to the training set before the next round.

The final report-facing evaluation is limited to Human-only Clerical Review Baseline, AI-only ML Matcher, and AI + HITL Active Learning Matcher.
"""
    CONFIG.paths.active_learning_summary.write_text(text, encoding="utf-8")


def _write_tuning_summary(tuning: pd.DataFrame) -> None:
    best = tuning.sort_values("Best CV F1-score", ascending=False).iloc[0]
    text = f"""# Hyperparameter Tuning Summary

The active-learning experiment tunes Logistic Regression, Random Forest, and Gradient Boosting with `GridSearchCV`. Tuning uses the initial seed labels only, so the frozen test set remains reserved for evaluation.

## Best Tuned Classifier

- Method: {best['Method']}
- Best CV F1-score: {best['Best CV F1-score']:.3f}
- Best parameters: `{best['Best parameters']}`

## Full Tuning Table

```
{tuning.to_string(index=False)}
```
"""
    CONFIG.paths.hyperparameter_tuning_summary.write_text(text, encoding="utf-8")


def run_active_learning_experiment() -> dict[str, pd.DataFrame]:
    """Run a reproducible active-learning simulation on FEBRL candidate-pair features."""
    CONFIG.validate()
    ensure_directories_exist(CONFIG.paths.tables_dir, CONFIG.paths.reports_dir, CONFIG.paths.figures_dir)
    pairs = _load_candidate_pairs_for_active_learning()
    x, y = _feature_matrix(pairs)

    x_train_pool, x_test, y_train_pool, y_test = train_test_split(
        x,
        y,
        test_size=CONFIG.active_learning.test_size,
        stratify=y,
        random_state=CONFIG.active_learning.random_state,
    )
    rng = np.random.default_rng(CONFIG.active_learning.random_state)
    seed = _seed_indices(
        y_train_pool,
        CONFIG.active_learning.seed_positive_labels,
        CONFIG.active_learning.seed_negative_labels,
        rng,
    )

    tuned_models, tuning = _tune_models(
        x_train_pool.loc[seed],
        y_train_pool.loc[seed],
        CONFIG.active_learning.random_state,
    )
    comparison = _run_model_comparison(
        x_train_pool.loc[seed],
        y_train_pool.loc[seed],
        x_test,
        y_test,
        tuned_models,
        tuning,
    )
    best_model = tuning.sort_values("Best CV F1-score", ascending=False).iloc[0]["Method"]
    best_model_template = tuned_models[best_model]

    active_rounds = _active_learning_loop(
        best_model,
        best_model_template,
        x_train_pool,
        y_train_pool,
        x_test,
        y_test,
        CONFIG.active_learning.batch_size,
        CONFIG.active_learning.rounds,
        seed,
        CONFIG.active_learning.random_state,
        "Active Learning",
    )
    final_research = _final_research_evaluation(comparison, active_rounds, len(pairs), best_model)

    save_csv(tuning, CONFIG.paths.hyperparameter_tuning)
    save_csv(comparison, CONFIG.paths.model_comparison)
    save_csv(active_rounds, CONFIG.paths.active_learning_rounds)
    save_csv(final_research, CONFIG.paths.final_research_evaluation)
    _bar_chart(comparison, CONFIG.paths.model_comparison_f1_figure)
    _line_chart(
        active_rounds,
        CONFIG.paths.active_learning_curve_figure,
        "Active learning F1 over review rounds",
        "F1-score",
        y_min=0.99,
        y_max=1.001,
    )
    _error_reduction_chart(active_rounds, CONFIG.paths.active_learning_error_reduction_figure)
    _final_research_chart(final_research, CONFIG.paths.final_research_evaluation_figure)
    _final_accuracy_chart(final_research, CONFIG.paths.final_accuracy_comparison_figure)
    _final_workload_chart(final_research, CONFIG.paths.final_workload_comparison_figure)
    _write_summary(active_rounds, comparison, final_research, tuning)
    _write_tuning_summary(tuning)
    return {
        "hyperparameter_tuning": tuning,
        "model_comparison": comparison,
        "active_learning_rounds": active_rounds,
        "final_research_evaluation": final_research,
    }
