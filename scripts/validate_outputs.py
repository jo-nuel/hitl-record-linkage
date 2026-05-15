import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import CONFIG  # noqa: E402


REQUIRED_OUTPUTS = [
    CONFIG.paths.final_research_evaluation,
    CONFIG.paths.model_comparison,
    CONFIG.paths.hyperparameter_tuning,
    CONFIG.paths.active_learning_rounds,
    CONFIG.paths.review_queue,
    CONFIG.paths.classified_pairs,
    CONFIG.paths.final_decisions,
    CONFIG.paths.review_decisions_export,
    CONFIG.paths.simulated_review_decisions,
    CONFIG.paths.evaluation_summary,
    CONFIG.paths.dataset_profile,
    CONFIG.paths.blocking_summary,
    CONFIG.paths.active_learning_summary,
    CONFIG.paths.hyperparameter_tuning_summary,
    CONFIG.paths.model_comparison_f1_figure,
    CONFIG.paths.active_learning_curve_figure,
    CONFIG.paths.active_learning_error_reduction_figure,
    CONFIG.paths.final_accuracy_comparison_figure,
    CONFIG.paths.final_workload_comparison_figure,
]

EXPECTED_FINAL_RESEARCH_METHODS = {
    "Human-only Clerical Review Baseline",
    "AI-only ML Matcher",
    "AI + HITL Active Learning Matcher",
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


def _warn(message: str) -> None:
    print(f"WARN: {message}")


def _failures_for_files() -> list[str]:
    return [f"Missing required output: {path}" for path in REQUIRED_OUTPUTS if not path.exists()]


def _validate_final_research_evaluation() -> list[str]:
    df = pd.read_csv(CONFIG.paths.final_research_evaluation)
    failures: list[str] = []
    methods = set(df["Method"]) if "Method" in df.columns else set()
    if methods != EXPECTED_FINAL_RESEARCH_METHODS:
        failures.append(
            f"final_research_evaluation.csv methods are {sorted(methods)}, expected {sorted(EXPECTED_FINAL_RESEARCH_METHODS)}"
        )
    missing_columns = REQUIRED_METRIC_COLUMNS - set(df.columns)
    if missing_columns:
        failures.append(f"final_research_evaluation.csv missing columns: {sorted(missing_columns)}")
    for required_column in ["Role", "Dataset", "Evaluation scope"]:
        if required_column not in df.columns:
            failures.append(f"final_research_evaluation.csv missing column: {required_column}")
    if "Dataset" in df.columns and set(df["Dataset"]) != {"FEBRL4"}:
        failures.append("final_research_evaluation.csv must identify FEBRL4 as the evaluation dataset")
    if len(df) != 3:
        failures.append("final_research_evaluation.csv must contain exactly three rows")
    expected_order = [
        "Human-only Clerical Review Baseline",
        "AI-only ML Matcher",
        "AI + HITL Active Learning Matcher",
    ]
    if "Method" in df.columns and df["Method"].to_list() != expected_order:
        failures.append("final_research_evaluation.csv should list the three final methods in report order")
    if not failures:
        _pass("final research evaluation has the expected three report-facing methods")
    return failures


def _validate_active_learning_outputs() -> list[str]:
    failures: list[str] = []
    model_comparison = pd.read_csv(CONFIG.paths.model_comparison)
    tuning = pd.read_csv(CONFIG.paths.hyperparameter_tuning)
    rounds = pd.read_csv(CONFIG.paths.active_learning_rounds)
    expected_models = {"Logistic Regression", "Random Forest", "Gradient Boosting"}
    if set(model_comparison.get("Method", [])) != expected_models:
        failures.append("model_comparison.csv must contain Logistic Regression, Random Forest, and Gradient Boosting")
    required_tuning_columns = {"Method", "Best CV F1-score", "Best parameters", "Tuning runtime seconds", "Selection note"}
    missing_tuning_columns = required_tuning_columns - set(tuning.columns)
    if missing_tuning_columns:
        failures.append(f"hyperparameter_tuning.csv missing columns: {sorted(missing_tuning_columns)}")
    expected_tuned_models = {"Logistic Regression", "Random Forest", "Gradient Boosting"}
    if set(tuning.get("Method", [])) != expected_tuned_models:
        failures.append("hyperparameter_tuning.csv must contain Logistic Regression, Random Forest, and Gradient Boosting")
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
    if rounds.empty:
        failures.append("active_learning_rounds.csv must have rows")
    else:
        sorted_rounds = rounds.sort_values("Round")
        labelled_counts = sorted_rounds["Labelled pairs"].astype(int).to_list()
        if labelled_counts != sorted(labelled_counts):
            failures.append("active_learning_rounds.csv labelled pair counts must increase monotonically")
        first_round = sorted_rounds.iloc[0]
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
        _pass("configuration values are valid")
    except ValueError as error:
        failures.append(str(error))

    if CONFIG.paths.blocking_stats.exists():
        blocking = pd.read_csv(CONFIG.paths.blocking_stats)
        if blocking.empty or int(blocking.iloc[0].get("candidate_pairs", 0)) <= 0:
            failures.append("blocking_stats.csv must report candidate_pairs greater than zero")
        else:
            _pass("blocking stats report candidate pairs greater than zero")
    return failures


def _validate_readme_framing() -> list[str]:
    """Catch wording drift that would confuse the final research method."""
    readme_path = REPO_ROOT / "README.md"
    text = readme_path.read_text(encoding="utf-8").lower()
    failures: list[str] = []
    if "ai-assisted active learning record linkage using febrl4" not in text:
        failures.append("README must name the final method as AI-assisted active learning record linkage using FEBRL4")
    if "synthea is the final" in text or "final dataset: synthea" in text or "final method uses synthea" in text:
        failures.append("README must not describe Synthea as the final dataset or final method")
    forbidden_active_learning_phrases = [
        "active learning is only an extension",
        "active learning is a side extension",
        "active learning as a side extension",
        "minor extension",
    ]
    for phrase in forbidden_active_learning_phrases:
        if phrase in text:
            failures.append(f"README should not frame active learning as a side feature: '{phrase}'")
    forbidden_final_method_phrases = [
        "empi-inspired hitl workflow",
        "hybrid empi",
        "ecm",
        "threshold sweep",
        "run threshold",
        "supporting baselines",
        "five-method",
        "five methods",
        "random sampling hitl baseline",
    ]
    for phrase in forbidden_final_method_phrases:
        if phrase in text:
            failures.append(f"README should not describe supporting methods as final evaluation methods: '{phrase}'")
    if not failures:
        _pass("README frames FEBRL4 and active learning as the final method")
    return failures


def _validate_streamlit_import() -> list[str]:
    """Import the dashboard module to catch syntax errors without launching Streamlit."""
    app_path = REPO_ROOT / "app" / "streamlit_app.py"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        spec = importlib.util.spec_from_file_location("streamlit_app_validation", app_path)
        if spec is None or spec.loader is None:
            return [f"Could not create import spec for {app_path}"]
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as error:  # noqa: BLE001 - validation should report any import failure clearly.
        return [f"Streamlit app import failed: {error}"]
    _pass("Streamlit app imports without syntax errors")
    return []


def main() -> None:
    """Validate that generated evidence outputs are present and internally plausible."""
    failures = []
    failures.extend(_failures_for_files())
    if failures:
        print("Output validation failed:")
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    failures.extend(_validate_final_research_evaluation())
    failures.extend(_validate_active_learning_outputs())
    failures.extend(_validate_review_queue())
    failures.extend(_validate_public_pair_exports())
    failures.extend(_validate_config_and_blocking())
    failures.extend(_validate_readme_framing())
    failures.extend(_validate_streamlit_import())

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
