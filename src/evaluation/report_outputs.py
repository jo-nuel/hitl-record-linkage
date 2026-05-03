from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.empi.matcher import FIELD_WEIGHTS
from src.utils.config import CONFIG
from src.utils.io import ensure_directories_exist, save_csv


DISPLAY_LABELS = {
    "manual_blocked_benchmark": "Human-only Clerical Review Baseline",
    "ai_only": "AI-only EMPI Matcher",
    "ai_hitl_simulated": "AI + HITL Grey-Zone Review",
}


def display_label(value: str) -> str:
    return DISPLAY_LABELS.get(value, value)


def benchmark_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create a compact precision, recall, and F1 table for report evidence."""
    columns = ["approach", "precision", "recall", "f1_score", "true_positives", "false_positives", "false_negatives"]
    table = metrics[columns].copy()
    table["approach"] = table["approach"].map(display_label)
    return table.round(3)


def final_evaluation_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create the main three-method comparison table for the final submission."""
    rows = []
    interpretations = {
        "manual_blocked_benchmark": "Highest workload because every blocked candidate pair is reviewed.",
        "ai_only": "No manual workload, but grey-zone true links are missed when left unresolved.",
        "ai_hitl_simulated": "Best accuracy-efficiency trade-off by reviewing only grey-zone pairs.",
    }
    for _, row in metrics.iterrows():
        rows.append(
            {
                "Method": display_label(row["approach"]),
                "Precision": round(float(row["precision"]), 3),
                "Recall": round(float(row["recall"]), 3),
                "F1-score": round(float(row["f1_score"]), 3),
                "False positives": int(row["false_positives"]),
                "False negatives": int(row["false_negatives"]),
                "Candidate pairs reviewed": int(row["pairs_reviewed"]),
                "Review workload percentage": round(float(row["review_workload_percent"]) * 100, 3),
                "Estimated review time": f"{float(row['estimated_review_seconds']):,.0f} seconds",
                "Key interpretation": interpretations[row["approach"]],
            }
        )
    return pd.DataFrame(rows)


def workload_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create a workload table showing how much review effort each method needs."""
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
    """Plot precision, recall, and F1 for the three formal evaluation methods."""
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
    """Plot reviewed-pair workload with labels so small HITL counts remain visible."""
    df = _labeled_metrics(metrics)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(df["approach"], df["pairs_reviewed"], color="#4C78A8")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["approach"], rotation=15, ha="right")
    ax.set_ylabel("Pairs reviewed")
    ax.set_title("Review Workload")
    for bar, value in zip(bars, df["pairs_reviewed"]):
        ax.annotate(
            f"{int(value):,}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    return fig


def build_workload_percentage_figure(metrics: pd.DataFrame) -> plt.Figure:
    """Plot review workload as a percentage for easier comparison in reports."""
    df = _labeled_metrics(metrics)
    percentages = df["review_workload_percent"] * 100
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(df["approach"], percentages, color="#2E8B57")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["approach"], rotation=15, ha="right")
    ax.set_ylabel("Candidate pairs reviewed (%)")
    ax.set_title("Review Workload Percentage")
    for bar, value in zip(bars, percentages):
        ax.annotate(
            f"{value:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
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
    """Write concise technical summaries used as evidence, not a full report draft."""
    ensure_directories_exist(CONFIG.paths.reports_dir)
    CONFIG.paths.methodology_summary.write_text(
        "# Methodology Summary\n\nThe workflow loads FEBRL4, preprocesses identity fields, creates candidate pairs with multi-pass blocking, compares fields with Jaro-Winkler and exact agreement, combines ECM probability with a Hybrid EMPI-style evidence score, and sends grey-zone pairs to human review.\n",
        encoding="utf-8",
    )
    CONFIG.paths.blocking_summary.write_text(
        "# Blocking Summary\n\n"
        f"Total possible pairs: {int(blocking_stats['total_possible_pairs']):,}\n\n"
        f"Candidate pairs after blocking: {int(blocking_stats['candidate_pairs']):,}\n\n"
        f"Reduction ratio: {blocking_stats['reduction_ratio']:.3f}\n\n"
        f"True links retained: {int(blocking_stats['true_links_retained']):,}\n\n"
        f"Missed true links: {int(blocking_stats['true_links_missed']):,}\n\n"
        f"Blocking recall / pair completeness: {blocking_stats['blocking_recall']:.3f}\n\n"
        f"Blocking rules: {blocking_stats['blocking_rules']}\n",
        encoding="utf-8",
    )
    CONFIG.paths.evaluation_summary.write_text(
        "# Evaluation Summary\n\n"
        "The final comparison uses exactly three methods: Human-only Clerical Review Baseline, AI-only EMPI Matcher, and AI + HITL Grey-Zone Review.\n\n"
        "AI-only treats grey-zone pairs as unresolved non-positive predictions, so true links in the grey zone count as missed links.\n\n"
        "Formal benchmark metrics are generated from the evaluation pipeline. The AI + HITL result uses simulated grey-zone review based on FEBRL ground truth to represent an idealised human reviewer. Live reviewer decisions in Streamlit are stored for demonstration and audit logging, but they do not automatically overwrite formal benchmark metrics unless the pipeline is explicitly rerun in merge mode.\n\n"
        "```\n"
        + final_evaluation_comparison(metrics).to_string(index=False)
        + "\n```\n",
        encoding="utf-8",
    )
    weight_lines = "\n".join(f"- {field}: {weight:.2f}" for field, weight in FIELD_WEIGHTS.items())
    CONFIG.paths.scoring_method_summary.write_text(
        "# Scoring Method Summary\n\n"
        "The matcher uses a blended EMPI-inspired score. It first attempts to estimate pair-level probability with `recordlinkage.ECMClassifier`. It also calculates a transparent Hybrid EMPI-style evidence score from field-level agreement values. The final model score blends ECM probability with the hybrid score using the configured ECM weight.\n\n"
        "## Field Weights\n\n"
        f"{weight_lines}\n\n"
        "Date of birth, surname, postcode, and address receive stronger weight because they provide stronger identity evidence than broad or sometimes missing fields. The hybrid score also applies penalties for strong disagreement in date of birth, surname, postcode, and sex/gender where those fields are available.\n\n"
        f"Default lower threshold: {CONFIG.matcher.lower_threshold:.2f}\n\n"
        f"Default upper threshold: {CONFIG.matcher.upper_threshold:.2f}\n\n"
        "Pairs at or above the upper threshold become Auto Match. Pairs at or below the lower threshold become Auto Non-match. Pairs between the thresholds are grey-zone cases sent to human review. This makes the method explainable because reviewers can inspect both the total score and the field-level evidence.\n",
        encoding="utf-8",
    )
    CONFIG.paths.limitations.write_text(
        "# Limitations\n\nFEBRL is benchmark data, not production hospital data. Simulated HITL uses ground truth as an ideal reviewer. Blocking can still miss true links before human review sees them. Thresholds need further validation before deployment claims.\n",
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
    final_comparison = final_evaluation_comparison(metrics)
    workload = workload_table(metrics)
    decisions_table = decision_counts_table(classified, final_pairs, decisions)
    save_csv(metrics, CONFIG.paths.evaluation_metrics)
    save_csv(final_comparison, CONFIG.paths.final_evaluation_comparison)
    save_csv(benchmark, CONFIG.paths.benchmark_table)
    save_csv(workload, CONFIG.paths.workload_table)
    save_csv(decisions_table, CONFIG.paths.decision_counts_table)
    _save_figure(build_benchmark_figure(metrics), CONFIG.paths.benchmark_figure)
    _save_figure(build_workload_figure(metrics), CONFIG.paths.workload_figure)
    _save_figure(build_workload_percentage_figure(metrics), CONFIG.paths.workload_percentage_figure)
    _save_figure(build_decision_distribution_figure(classified), CONFIG.paths.decision_distribution_figure)
    _save_figure(build_score_distribution_figure(classified), CONFIG.paths.score_distribution_figure)
    _save_figure(build_resolution_flow_figure(final_pairs), CONFIG.paths.resolution_flow_figure)
    write_report_texts(metrics, blocking_stats)
    return {
        "benchmark_table": benchmark,
        "final_evaluation_comparison": final_comparison,
        "workload_table": workload,
        "decision_counts_table": decisions_table,
    }
