import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import CONFIG  # noqa: E402


REQUIRED_OUTPUTS = [
    CONFIG.paths.final_evaluation_comparison,
    CONFIG.paths.threshold_sweep,
    CONFIG.paths.model_comparison,
    CONFIG.paths.active_learning_rounds,
    CONFIG.paths.random_vs_active_learning,
    CONFIG.paths.review_queue,
    CONFIG.paths.classified_pairs,
    CONFIG.paths.final_decisions,
    CONFIG.paths.review_decisions_export,
    CONFIG.paths.simulated_review_decisions,
    CONFIG.paths.evaluation_summary,
    CONFIG.paths.dataset_profile,
    CONFIG.paths.blocking_summary,
    CONFIG.paths.scoring_method_summary,
    CONFIG.paths.active_learning_summary,
    CONFIG.paths.benchmark_figure,
    CONFIG.paths.workload_figure,
    CONFIG.paths.workload_percentage_figure,
    CONFIG.paths.threshold_f1_figure,
    CONFIG.paths.threshold_workload_figure,
    CONFIG.paths.recall_workload_figure,
    CONFIG.paths.model_comparison_f1_figure,
    CONFIG.paths.active_learning_curve_figure,
    CONFIG.paths.random_vs_active_learning_figure,
    CONFIG.paths.label_efficiency_curve_figure,
]


EXPECTED_METHODS = {
    "Human-only Clerical Review Baseline",
    "AI-only EMPI Matcher",
    "AI + HITL Grey-Zone Review",
}

REQUIRED_METRIC_COLUMNS = {
    "Precision",
    "Recall",
    "F1-score",
    "False positives",
    "False negatives",
    "Candidate pairs reviewed",
    "Review workload percentage",
    "Estimated review time",
}


def _pass(message: str) -> None:
    print(f"PASS: {message}")


def _failures_for_files() -> list[str]:
    return [f"Missing required output: {path}" for path in REQUIRED_OUTPUTS if not path.exists()]


def _validate_final_comparison() -> list[str]:
    df = pd.read_csv(CONFIG.paths.final_evaluation_comparison)
    failures: list[str] = []
    methods = set(df["Method"]) if "Method" in df.columns else set()
    if methods != EXPECTED_METHODS:
        failures.append(f"final_evaluation_comparison.csv methods are {sorted(methods)}, expected {sorted(EXPECTED_METHODS)}")
    missing_columns = REQUIRED_METRIC_COLUMNS - set(df.columns)
    if missing_columns:
        failures.append(f"final_evaluation_comparison.csv missing columns: {sorted(missing_columns)}")
    if len(df) != 3:
        failures.append("final_evaluation_comparison.csv must contain exactly three rows")
    if not failures:
        _pass("final evaluation comparison has the expected three methods and metric columns")
    return failures


def _validate_threshold_sweep() -> list[str]:
    df = pd.read_csv(CONFIG.paths.threshold_sweep)
    if df.empty:
        return ["threshold_sweep.csv has no rows"]
    _pass("threshold sweep has rows")
    return []


def _validate_active_learning_outputs() -> list[str]:
    failures: list[str] = []
    model_comparison = pd.read_csv(CONFIG.paths.model_comparison)
    rounds = pd.read_csv(CONFIG.paths.active_learning_rounds)
    random_vs_active = pd.read_csv(CONFIG.paths.random_vs_active_learning)
    expected_models = {"Hybrid EMPI Score", "Logistic Regression", "Random Forest", "Gradient Boosting"}
    if set(model_comparison.get("Method", [])) != expected_models:
        failures.append("model_comparison.csv must contain Hybrid EMPI Score, Logistic Regression, Random Forest, and Gradient Boosting")
    required_round_columns = {
        "Round",
        "Strategy",
        "Classifier",
        "Labelled pairs",
        "New labels added",
        "Unlabelled pairs remaining",
        "Precision",
        "Recall",
        "F1-score",
        "False positives",
        "False negatives",
    }
    missing_round_columns = required_round_columns - set(rounds.columns)
    if missing_round_columns:
        failures.append(f"active_learning_rounds.csv missing columns: {sorted(missing_round_columns)}")
    if rounds.empty or random_vs_active.empty:
        failures.append("active-learning tables must have rows")
    if "Active Learning" not in set(random_vs_active.get("Strategy", [])):
        failures.append("random_vs_active_learning.csv must include Active Learning rows")
    if "Random Sampling" not in set(random_vs_active.get("Strategy", [])):
        failures.append("random_vs_active_learning.csv must include Random Sampling rows")
    for strategy_name, group in random_vs_active.groupby("Strategy"):
        labelled_counts = group.sort_values("Round")["Labelled pairs"].astype(int).to_list()
        if labelled_counts != sorted(labelled_counts):
            failures.append(f"{strategy_name} labelled pair counts must increase monotonically across rounds")
    first_round = rounds.sort_values("Round").iloc[0]
    if int(first_round["Round"]) != 0 or int(first_round["New labels added"]) != 0:
        failures.append("Round 0 must be the seed-only model with New labels added = 0")
    if not failures:
        _pass("active-learning model comparison and learning-curve outputs are present")
    return failures


def _validate_review_queue() -> list[str]:
    df = pd.read_csv(CONFIG.paths.review_queue, nrows=5)
    if "is_true_link" in df.columns:
        return ["review_queue.csv exposes is_true_link; ground truth must not be visible in the HITL queue"]
    _pass("review queue does not expose ground-truth labels")
    return []


def _validate_public_pair_exports() -> list[str]:
    failures: list[str] = []
    for path in [CONFIG.paths.classified_pairs, CONFIG.paths.final_decisions]:
        df = pd.read_csv(path, nrows=5)
        if "is_true_link" in df.columns:
            failures.append(f"{path} exposes is_true_link; public pair exports should not expose ground truth")
    if not failures:
        _pass("classified and final decision exports do not expose ground-truth labels")
    return failures


def _validate_config_and_blocking() -> list[str]:
    failures: list[str] = []
    try:
        CONFIG.validate()
        _pass("configuration thresholds are valid")
    except ValueError as error:
        failures.append(str(error))

    if CONFIG.paths.blocking_stats.exists():
        blocking = pd.read_csv(CONFIG.paths.blocking_stats)
        if blocking.empty or int(blocking.iloc[0].get("candidate_pairs", 0)) <= 0:
            failures.append("blocking_stats.csv must report candidate_pairs greater than zero")
        else:
            _pass("blocking stats report candidate pairs greater than zero")
    return failures


def main() -> None:
    """Validate that generated evidence outputs are present and internally plausible."""
    failures = []
    failures.extend(_failures_for_files())
    if failures:
        print("Output validation failed:")
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    failures.extend(_validate_final_comparison())
    failures.extend(_validate_threshold_sweep())
    failures.extend(_validate_active_learning_outputs())
    failures.extend(_validate_review_queue())
    failures.extend(_validate_public_pair_exports())
    failures.extend(_validate_config_and_blocking())

    if failures:
        print("Output validation failed:")
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    print("Output validation passed.")
    for path in REQUIRED_OUTPUTS:
        print(f"- {path}")


if __name__ == "__main__":
    main()
