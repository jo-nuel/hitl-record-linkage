from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import CONFIG
from src.utils.io import ensure_directories_exist, save_csv


DISPLAY_LABELS = {
    "manual_blocked_benchmark": "Manual Benchmark, Blocked Set",
    "ai_only": "AI Only",
    "ai_hitl_simulated": "AI + HITL Simulated",
    "ai_hitl_review_file": "AI + HITL Review File",
}


def display_label(value: str) -> str:
    return DISPLAY_LABELS.get(value, value)


def benchmark_table(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = ["approach", "precision", "recall", "f1_score", "true_positives", "false_positives", "false_negatives"]
    table = metrics[columns].copy()
    table["approach"] = table["approach"].map(display_label)
    return table.round(3)


def workload_table(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "approach",
        "candidate_pairs",
        "review_needed_pairs",
        "pairs_reviewed",
        "review_workload_percent",
        "workload_reduction_vs_full_review",
        "auto_matches",
        "auto_non_matches",
        "review_pending_pairs",
    ]
    table = metrics[columns].copy()
    table["approach"] = table["approach"].map(display_label)
    return table.round(4)


def decision_counts_table(classified: pd.DataFrame, final_pairs: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"category": "Auto Match", "count": int((classified["model_decision"] == "Auto Match").sum())},
        {"category": "Auto Non-match", "count": int((classified["model_decision"] == "Auto Non-match").sum())},
        {"category": "Needs Human Review", "count": int((classified["model_decision"] == "Needs Human Review").sum())},
        {"category": "Final Match", "count": int((final_pairs["final_decision"] == "Match").sum())},
        {"category": "Final Non-match", "count": int((final_pairs["final_decision"] == "Non-match").sum())},
        {"category": "Still Needs Review", "count": int((final_pairs["final_decision"] == "Needs Human Review").sum())},
    ]
    if not decisions.empty:
        for decision, count in decisions["reviewer_decision"].value_counts().items():
            rows.append({"category": f"Review: {decision}", "count": int(count)})
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    ensure_directories_exist(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _labeled_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    df = metrics.copy()
    df["approach"] = df["approach"].map(display_label)
    return df


def build_benchmark_figure(metrics: pd.DataFrame) -> plt.Figure:
    df = _labeled_metrics(metrics)
    x = range(len(df))
    width = 0.22
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar([i - width for i in x], df["precision"], width, label="Precision")
    ax.bar(x, df["recall"], width, label="Recall")
    ax.bar([i + width for i in x], df["f1_score"], width, label="F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["approach"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Benchmark Comparison")
    ax.legend(frameon=False)
    return fig


def build_workload_figure(metrics: pd.DataFrame) -> plt.Figure:
    df = _labeled_metrics(metrics)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(df["approach"], df["pairs_reviewed"], color="#4C78A8")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["approach"], rotation=15, ha="right")
    ax.set_ylabel("Pairs reviewed")
    ax.set_title("Review Workload")
    return fig


def build_decision_distribution_figure(classified: pd.DataFrame) -> plt.Figure:
    counts = classified["model_decision"].value_counts().reindex(
        ["Auto Match", "Auto Non-match", "Needs Human Review"], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(counts.index, counts.values, color=["#2E8B57", "#4C78A8", "#F4A259"])
    ax.set_ylabel("Pair count")
    ax.set_title("Decision Distribution")
    return fig


def build_score_distribution_figure(classified: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(pd.to_numeric(classified["model_score"], errors="coerce").dropna(), bins=25, color="#4C78A8", edgecolor="white")
    ax.set_xlabel("Model score")
    ax.set_ylabel("Pair count")
    ax.set_title("Score Distribution")
    return fig


def build_resolution_flow_figure(final_pairs: pd.DataFrame) -> plt.Figure:
    values = [
        int(final_pairs["model_decision"].isin(["Auto Match", "Auto Non-match"]).sum()),
        int(final_pairs["reviewer_decision"].isin(["Confirm Match", "Reject Match"]).sum()),
        int((final_pairs["final_decision"] == "Needs Human Review").sum()),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(["Resolved by model", "Resolved by reviewer", "Still pending"], values, color=["#4C78A8", "#2E8B57", "#F4A259"])
    ax.set_ylabel("Pair count")
    ax.set_title("Resolution Flow")
    return fig


def write_report_texts(metrics: pd.DataFrame, blocking_stats: dict[str, float]) -> None:
    ensure_directories_exist(CONFIG.paths.reports_dir)
    CONFIG.paths.problem_formulation.write_text(
        "# Problem Formulation\n\nDuplicate patient records can fragment identity information across systems. This project evaluates an EMPI-inspired linkage workflow using FEBRL benchmark data and human review for ambiguous pairs.\n",
        encoding="utf-8",
    )
    CONFIG.paths.methodology_summary.write_text(
        "# Methodology Summary\n\nThe workflow loads FEBRL4, preprocesses identity fields, creates candidate pairs with multi-pass blocking, compares fields with Jaro-Winkler and exact agreement, combines ECM probability with a Hybrid EMPI-style evidence score, and sends uncertain pairs to human review.\n",
        encoding="utf-8",
    )
    CONFIG.paths.evaluation_summary.write_text(
        "# Evaluation Summary\n\n```\n" + benchmark_table(metrics).to_string(index=False) + "\n```\n",
        encoding="utf-8",
    )
    CONFIG.paths.limitations.write_text(
        "# Limitations\n\nFEBRL is benchmark data, not production hospital data. Simulated HITL uses ground truth as an ideal reviewer. Blocking can still miss true links before human review sees them. Thresholds need further validation before deployment claims.\n",
        encoding="utf-8",
    )
    CONFIG.paths.weekly_reflection_change_summary.write_text(
        "# Weekly Reflection Change Summary\n\nAfter feedback, we removed the Synthea-based duplicate generation pipeline because it weakened evaluation validity. We moved to FEBRL as the primary dataset because it provides benchmark linkage data with ground-truth links. We also changed the matching design toward an EMPI-inspired workflow using multi-pass blocking, field-level evidence, probabilistic or hybrid scoring, threshold tuning, and human review of uncertain pairs. This made the project more realistic, more defensible, and better aligned with healthcare patient matching systems.\n",
        encoding="utf-8",
    )
    CONFIG.paths.experiment_summary.write_text(
        "# Experiment Summary\n\n"
        f"Candidate pairs after blocking: {int(blocking_stats['candidate_pairs']):,}\n\n"
        f"Blocking recall: {blocking_stats['blocking_recall']:.3f}\n\n"
        + "```\n"
        + benchmark_table(metrics).to_string(index=False)
        + "\n```\n",
        encoding="utf-8",
    )


def generate_report_outputs(
    metrics: pd.DataFrame,
    classified: pd.DataFrame,
    final_pairs: pd.DataFrame,
    decisions: pd.DataFrame,
    blocking_stats: dict[str, float],
) -> dict[str, pd.DataFrame]:
    ensure_directories_exist(CONFIG.paths.tables_dir, CONFIG.paths.figures_dir, CONFIG.paths.reports_dir)
    benchmark = benchmark_table(metrics)
    workload = workload_table(metrics)
    decisions_table = decision_counts_table(classified, final_pairs, decisions)
    save_csv(metrics, CONFIG.paths.evaluation_metrics)
    save_csv(benchmark, CONFIG.paths.benchmark_table)
    save_csv(workload, CONFIG.paths.workload_table)
    save_csv(decisions_table, CONFIG.paths.decision_counts_table)
    _save_figure(build_benchmark_figure(metrics), CONFIG.paths.benchmark_figure)
    _save_figure(build_workload_figure(metrics), CONFIG.paths.workload_figure)
    _save_figure(build_decision_distribution_figure(classified), CONFIG.paths.decision_distribution_figure)
    _save_figure(build_score_distribution_figure(classified), CONFIG.paths.score_distribution_figure)
    _save_figure(build_resolution_flow_figure(final_pairs), CONFIG.paths.resolution_flow_figure)
    write_report_texts(metrics, blocking_stats)
    return {"benchmark_table": benchmark, "workload_table": workload, "decision_counts_table": decisions_table}
