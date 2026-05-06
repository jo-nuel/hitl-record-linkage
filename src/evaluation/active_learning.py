import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

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


def _models(random_state: int) -> dict[str, object]:
    """Return small classifiers suitable for fast local active-learning demos."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=80,
            max_depth=8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }


def _predict_probability(model: object, x: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(x)
    return probabilities[:, 1]


def _metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    predicted = (probabilities >= threshold).astype(int)
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


def _hybrid_baseline(test_pairs: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    probabilities = pd.to_numeric(test_pairs["model_score"], errors="coerce").fillna(0.0).to_numpy()
    row = _metrics(y_test, probabilities)
    row["Method"] = "Hybrid EMPI Score"
    row["Reviewed pairs"] = 0
    row["Runtime seconds"] = 0.0
    return row


def _run_model_comparison(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    test_pairs: pd.DataFrame,
    random_state: int,
) -> pd.DataFrame:
    rows = [_hybrid_baseline(test_pairs, y_test)]
    for name, model in _models(random_state).items():
        start = time.perf_counter()
        model.fit(x_train, y_train)
        probabilities = _predict_probability(model, x_test)
        row = _metrics(y_test, probabilities)
        row["Method"] = name
        row["Reviewed pairs"] = 0
        row["Runtime seconds"] = time.perf_counter() - start
        rows.append(row)
    return pd.DataFrame(rows)


def _active_learning_loop(
    model_name: str,
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
    """Simulate batch active learning using FEBRL truth as the reviewer label source."""
    rng = np.random.default_rng(random_state)
    labelled = list(seed_indices)
    unlabelled = [idx for idx in x_pool.index if idx not in labelled]
    rows = []
    model = _models(random_state)[model_name]

    for round_number in range(rounds + 1):
        new_labels_added = 0
        if round_number > 0 and unlabelled:
            # Select a batch using the previous labelled set, simulate reviewer
            # labels with FEBRL truth, then retrain before evaluating this round.
            model.fit(x_pool.loc[labelled], y_pool.loc[labelled])
            if strategy == "Active Learning":
                pool_probabilities = _predict_probability(model, x_pool.loc[unlabelled])
                uncertainty = np.abs(pool_probabilities - 0.5)
                selected_positions = np.argsort(uncertainty)[:batch_size]
                selected = [unlabelled[position] for position in selected_positions]
            else:
                selected = list(rng.choice(unlabelled, size=min(batch_size, len(unlabelled)), replace=False))

            selected_set = set(selected)
            labelled.extend(selected)
            unlabelled = [idx for idx in unlabelled if idx not in selected_set]
            new_labels_added = len(selected)

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


def _line_chart(df: pd.DataFrame, path: Path, title: str, y_column: str, hue_column: str = "Strategy") -> None:
    plt.figure(figsize=(8, 5))
    for label, group in df.groupby(hue_column):
        plt.plot(group["Labelled pairs"], group[y_column], marker="o", label=label)
    plt.title(title)
    plt.xlabel("Labelled pairs used")
    plt.ylabel(y_column)
    plt.ylim(0, 1.05)
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


def _final_research_evaluation(
    comparison: pd.DataFrame,
    active_rounds: pd.DataFrame,
    random_rounds: pd.DataFrame,
    candidate_count: int,
) -> pd.DataFrame:
    """Build the central final comparison for the active-learning research claim."""
    hybrid = comparison[comparison["Method"] == "Hybrid EMPI Score"].iloc[0]
    ml = comparison[comparison["Method"] != "Hybrid EMPI Score"].sort_values("F1-score", ascending=False).iloc[0]
    active = active_rounds.sort_values("F1-score", ascending=False).iloc[0]
    random = random_rounds.sort_values("F1-score", ascending=False).iloc[0]

    rows = [
        {
            "Method": "Human-only Clerical Review Baseline",
            "Role": "Primary baseline",
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
            "Key interpretation": "Best seed-trained ML classifier without further active-learning review batches.",
        },
        {
            "Method": "AI + HITL Active Learning Matcher",
            "Role": "Primary proposed method",
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
        {
            "Method": "Random Sampling HITL Baseline",
            "Role": "Supporting baseline",
            "Precision": random["Precision"],
            "Recall": random["Recall"],
            "F1-score": random["F1-score"],
            "False positives": int(random["False positives"]),
            "False negatives": int(random["False negatives"]),
            "Candidate pairs reviewed": int(random["Labelled pairs"]),
            "Review workload percentage": float(random["Labelled pairs"]) / candidate_count * 100,
            "Estimated review time": f"{float(random['Labelled pairs']) * CONFIG.matcher.manual_review_seconds_per_pair:,.0f} seconds",
            "Training labels used": int(random["Labelled pairs"]),
            "Evaluation scope": "Frozen active-learning test set",
            "Key interpretation": "Same label budget as active learning, but review pairs are selected randomly.",
        },
        {
            "Method": "Hybrid EMPI Baseline",
            "Role": "Supporting transparent baseline",
            "Precision": hybrid["Precision"],
            "Recall": hybrid["Recall"],
            "F1-score": hybrid["F1-score"],
            "False positives": int(hybrid["False positives"]),
            "False negatives": int(hybrid["False negatives"]),
            "Candidate pairs reviewed": 0,
            "Review workload percentage": 0.0,
            "Estimated review time": "0 seconds",
            "Training labels used": 0,
            "Evaluation scope": "Frozen active-learning test set",
            "Key interpretation": "Transparent non-ML baseline and fallback using field weights and penalties.",
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


def _write_summary(rounds: pd.DataFrame, comparison: pd.DataFrame, final_evaluation: pd.DataFrame) -> None:
    best = rounds.sort_values("F1-score", ascending=False).iloc[0]
    best_model = comparison.sort_values("F1-score", ascending=False).iloc[0]
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

## Best Model Comparison Result

- Method: {best_model['Method']}
- Precision: {best_model['Precision']:.3f}
- Recall: {best_model['Recall']:.3f}
- F1-score: {best_model['F1-score']:.3f}

## Interpretation

The EMPI-inspired pipeline provides the healthcare-style record linkage structure: preprocessing, blocking, field-level comparison, threshold decisions, and human review.

The Hybrid EMPI Score is retained as a transparent non-ML baseline and fallback scoring method. It is not presented as the main AI model.

The Active Learning ML Matcher is the main proposed method because it uses reviewer labels to improve future predictions over training rounds. Random Sampling HITL is included as a baseline to show whether uncertainty sampling is more label-efficient than reviewing randomly selected pairs.
"""
    CONFIG.paths.active_learning_summary.write_text(text, encoding="utf-8")


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
    test_pairs = pairs.loc[x_test.index]
    rng = np.random.default_rng(CONFIG.active_learning.random_state)
    seed = _seed_indices(
        y_train_pool,
        CONFIG.active_learning.seed_positive_labels,
        CONFIG.active_learning.seed_negative_labels,
        rng,
    )

    comparison = _run_model_comparison(
        x_train_pool.loc[seed],
        y_train_pool.loc[seed],
        x_test,
        y_test,
        test_pairs,
        CONFIG.active_learning.random_state,
    )
    best_model = comparison[comparison["Method"] != "Hybrid EMPI Score"].sort_values("F1-score", ascending=False).iloc[0][
        "Method"
    ]

    active_rounds = _active_learning_loop(
        best_model,
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
    random_rounds = _active_learning_loop(
        best_model,
        x_train_pool,
        y_train_pool,
        x_test,
        y_test,
        CONFIG.active_learning.batch_size,
        CONFIG.active_learning.rounds,
        seed,
        CONFIG.active_learning.random_state,
        "Random Sampling",
    )
    random_vs_active = pd.concat([active_rounds, random_rounds], ignore_index=True)
    final_research = _final_research_evaluation(comparison, active_rounds, random_rounds, len(pairs))

    save_csv(comparison, CONFIG.paths.model_comparison)
    save_csv(active_rounds, CONFIG.paths.active_learning_rounds)
    save_csv(random_vs_active, CONFIG.paths.random_vs_active_learning)
    save_csv(final_research, CONFIG.paths.final_research_evaluation)
    _bar_chart(comparison, CONFIG.paths.model_comparison_f1_figure)
    _line_chart(active_rounds, CONFIG.paths.active_learning_curve_figure, "Active learning F1 over review rounds", "F1-score")
    _line_chart(
        random_vs_active,
        CONFIG.paths.random_vs_active_learning_figure,
        "Active learning vs random sampling",
        "F1-score",
    )
    _line_chart(
        random_vs_active,
        CONFIG.paths.label_efficiency_curve_figure,
        "Recall gained per labelled-pair budget",
        "Recall",
    )
    _final_research_chart(final_research, CONFIG.paths.final_research_evaluation_figure)
    _write_summary(active_rounds, comparison, final_research)
    return {
        "model_comparison": comparison,
        "active_learning_rounds": active_rounds,
        "random_vs_active_learning": random_vs_active,
        "final_research_evaluation": final_research,
    }
