from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .config import CONFIG
from .utils import ensure_directories_exist


BENCHMARK_COLUMNS = [
    "approach",
    "precision",
    "recall",
    "f1_score",
    "true_positives",
    "false_positives",
    "false_negatives",
]

WORKLOAD_COLUMNS = [
    "approach",
    "candidate_pairs",
    "uncertain_pairs",
    "pairs_reviewed",
    "auto_matches",
    "auto_non_matches",
    "review_pending_pairs",
    "estimated_total_effort_seconds",
]

DISPLAY_LABELS = {
    "manual_only": "Manual Review Benchmark",
    "ai_only": "AI Only",
    "ai_human_hitl": "AI + HITL",
}


def _round_numeric_columns(df: pd.DataFrame, places: int = 3) -> pd.DataFrame:
    rounded = df.copy()
    for column in rounded.select_dtypes(include="number").columns:
        rounded[column] = rounded[column].round(places)
    return rounded


def build_benchmark_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    benchmark_df = metrics_df[BENCHMARK_COLUMNS].copy()
    benchmark_df["approach"] = benchmark_df["approach"].replace(DISPLAY_LABELS)
    return _round_numeric_columns(benchmark_df, places=3)


def build_workload_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    workload_df = metrics_df[WORKLOAD_COLUMNS].copy()
    workload_df["approach"] = workload_df["approach"].replace(DISPLAY_LABELS)
    return _round_numeric_columns(workload_df, places=2)


def build_decision_counts_table(
    classified_pairs_df: pd.DataFrame,
    final_decisions_df: pd.DataFrame,
    review_decisions_df: pd.DataFrame,
) -> pd.DataFrame:
    auto_match_count = int((classified_pairs_df["system_decision"] == "Match").sum())
    auto_non_match_count = int(
        (classified_pairs_df["system_decision"] == "Non-match").sum()
    )
    review_needed_count = int(
        (classified_pairs_df["system_decision"] == "Review Needed").sum()
    )
    final_match_count = int((final_decisions_df["final_decision"] == "Match").sum())
    final_non_match_count = int(
        (final_decisions_df["final_decision"] == "Non-match").sum()
    )
    review_pending_count = int(
        (final_decisions_df["final_decision"] == "Review Pending").sum()
    )

    rows = [
        {"category": "Auto Match", "count": auto_match_count},
        {"category": "Auto Non-match", "count": auto_non_match_count},
        {"category": "Review Needed", "count": review_needed_count},
        {"category": "Final Match", "count": final_match_count},
        {"category": "Final Non-match", "count": final_non_match_count},
        {"category": "Review Pending", "count": review_pending_count},
    ]

    if not review_decisions_df.empty:
        decision_counts = (
            review_decisions_df["reviewer_decision"].value_counts().sort_index()
        )
        for decision, count in decision_counts.items():
            rows.append({"category": f"Review: {decision}", "count": int(count)})

    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    ensure_directories_exist(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_metrics_figure(metrics_df: pd.DataFrame) -> plt.Figure:
    chart_df = metrics_df[["approach", "precision", "recall", "f1_score"]].copy()
    chart_df["approach"] = chart_df["approach"].replace(DISPLAY_LABELS)

    x = range(len(chart_df))
    width = 0.22
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar([pos - width for pos in x], chart_df["precision"], width=width, label="Precision")
    ax.bar(x, chart_df["recall"], width=width, label="Recall")
    ax.bar([pos + width for pos in x], chart_df["f1_score"], width=width, label="F1-score")
    ax.set_xticks(list(x))
    ax.set_xticklabels(chart_df["approach"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Benchmark Comparison")
    ax.legend(frameon=False)
    return fig


def create_metrics_figure(metrics_df: pd.DataFrame, path: Path) -> None:
    _save_figure(build_metrics_figure(metrics_df), path)


def build_workload_figure(metrics_df: pd.DataFrame) -> plt.Figure:
    chart_df = metrics_df[["approach", "pairs_reviewed", "review_pending_pairs"]].copy()
    chart_df["approach"] = chart_df["approach"].replace(DISPLAY_LABELS)

    x = range(len(chart_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar([pos - width / 2 for pos in x], chart_df["pairs_reviewed"], width=width, label="Pairs reviewed")
    ax.bar([pos + width / 2 for pos in x], chart_df["review_pending_pairs"], width=width, label="Review pending")
    ax.set_xticks(list(x))
    ax.set_xticklabels(chart_df["approach"])
    ax.set_ylabel("Pair count")
    ax.set_title("Review Workload by Approach")
    ax.legend(frameon=False)
    return fig


def create_workload_figure(metrics_df: pd.DataFrame, path: Path) -> None:
    _save_figure(build_workload_figure(metrics_df), path)


def build_decision_distribution_figure(classified_pairs_df: pd.DataFrame) -> plt.Figure:
    counts = (
        classified_pairs_df["system_decision"]
        .value_counts()
        .reindex(["Match", "Non-match", "Review Needed"], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(counts.index, counts.values, color=["#2E8B57", "#4C78A8", "#F4A259"])
    ax.set_ylabel("Pair count")
    ax.set_title("System Decision Distribution")
    return fig


def create_decision_distribution_figure(classified_pairs_df: pd.DataFrame, path: Path) -> None:
    _save_figure(build_decision_distribution_figure(classified_pairs_df), path)


def build_similarity_distribution_figure(classified_pairs_df: pd.DataFrame) -> plt.Figure:
    scores = pd.to_numeric(classified_pairs_df["overall_score"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.hist(scores, bins=20, color="#4C78A8", edgecolor="white")
    ax.set_xlabel("Overall similarity score")
    ax.set_ylabel("Pair count")
    ax.set_title("Similarity Score Distribution")
    return fig


def create_similarity_distribution_figure(classified_pairs_df: pd.DataFrame, path: Path) -> None:
    _save_figure(build_similarity_distribution_figure(classified_pairs_df), path)


def build_resolution_flow_figure(final_decisions_df: pd.DataFrame) -> plt.Figure:
    auto_resolved = int(
        final_decisions_df["system_decision"].isin(["Match", "Non-match"]).sum()
    )
    human_resolved = int(
        final_decisions_df["reviewer_decision"].isin(["Confirm Match", "Reject Match"]).sum()
    )
    pending = int((final_decisions_df["final_decision"] == "Review Pending").sum())

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    labels = ["Resolved by AI", "Resolved by Human", "Still Pending"]
    values = [auto_resolved, human_resolved, pending]
    ax.bar(labels, values, color=["#4C78A8", "#2E8B57", "#F4A259"])
    ax.set_ylabel("Pair count")
    ax.set_title("How Candidate Pairs Were Resolved")
    return fig


def create_resolution_flow_figure(final_decisions_df: pd.DataFrame, path: Path) -> None:
    _save_figure(build_resolution_flow_figure(final_decisions_df), path)


def generate_report_artifacts(
    metrics_df: pd.DataFrame,
    classified_pairs_df: pd.DataFrame,
    final_decisions_df: pd.DataFrame,
    review_decisions_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    benchmark_table = build_benchmark_table(metrics_df)
    workload_table = build_workload_table(metrics_df)
    decision_counts_table = build_decision_counts_table(
        classified_pairs_df,
        final_decisions_df,
        review_decisions_df,
    )

    ensure_directories_exist(CONFIG.paths.results_dir, CONFIG.paths.figures_dir)
    benchmark_table.to_csv(CONFIG.paths.benchmark_table, index=False)
    workload_table.to_csv(CONFIG.paths.workload_table, index=False)
    decision_counts_table.to_csv(CONFIG.paths.decision_counts_table, index=False)

    create_metrics_figure(metrics_df, CONFIG.paths.metrics_figure)
    create_workload_figure(metrics_df, CONFIG.paths.workload_figure)
    create_decision_distribution_figure(
        classified_pairs_df, CONFIG.paths.decision_distribution_figure
    )
    create_similarity_distribution_figure(
        classified_pairs_df, CONFIG.paths.similarity_distribution_figure
    )
    create_resolution_flow_figure(
        final_decisions_df, CONFIG.paths.resolution_flow_figure
    )

    return {
        "benchmark_table": benchmark_table,
        "workload_table": workload_table,
        "decision_counts_table": decision_counts_table,
    }
