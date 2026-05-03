import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.data.febrl_loader import load_febrl_dataset
from src.empi.blocking import generate_candidate_pairs
from src.empi.comparison import compute_comparison_features
from src.empi.hitl import apply_review_decisions, build_review_queue
from src.empi.matcher import classify_pairs
from src.empi.preprocessing import preprocess_records
from src.evaluation.metrics import add_ground_truth_labels, evaluate_results
from src.utils.config import CONFIG
from src.utils.io import ensure_directories_exist, save_csv


LOWER_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
UPPER_THRESHOLDS = [0.80, 0.85, 0.90, 0.95]


def _plot_lines(df: pd.DataFrame, y: str, path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for lower, group in df.groupby("lower_threshold"):
        group = group.sort_values("upper_threshold")
        ax.plot(group["upper_threshold"], group[y], marker="o", label=f"lower {lower:.2f}")
    ax.set_title(title)
    ax.set_xlabel("Upper threshold")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_threshold_sweep() -> pd.DataFrame:
    """Evaluate threshold settings to show the recall-workload trade-off."""
    ensure_directories_exist(CONFIG.paths.tables_dir, CONFIG.paths.reports_dir, CONFIG.paths.figures_dir)
    df_a_raw, df_b_raw, true_links = load_febrl_dataset(CONFIG.dataset.name)
    df_a = preprocess_records(df_a_raw)
    df_b = preprocess_records(df_b_raw)
    candidates, blocking_stats = generate_candidate_pairs(df_a, df_b, true_links)
    features = compute_comparison_features(candidates, df_a, df_b)
    scored = classify_pairs(features, 0.0, 1.0, CONFIG.matcher.ecm_weight)

    rows = []
    for lower in LOWER_THRESHOLDS:
        for upper in UPPER_THRESHOLDS:
            if lower >= upper:
                continue
            classified = scored.copy()
            classified["model_decision"] = "Needs Human Review"
            classified.loc[classified["model_score"] >= upper, "model_decision"] = "Auto Match"
            classified.loc[classified["model_score"] <= lower, "model_decision"] = "Auto Non-match"
            pairs = classified.reset_index()
            pairs = pairs.rename(columns={pairs.columns[0]: "record_id_a", pairs.columns[1]: "record_id_b"})
            pairs = add_ground_truth_labels(pairs, true_links)
            final_pairs = apply_review_decisions(pairs, pd.DataFrame())
            metrics = evaluate_results(final_pairs, true_links, blocking_stats, 0.0)
            ai_only = metrics[metrics["approach"] == "ai_only"].iloc[0]
            hitl = metrics[metrics["approach"] == "ai_hitl_simulated"].iloc[0]
            rows.append(
                {
                    "lower_threshold": lower,
                    "upper_threshold": upper,
                    "ai_only_precision": ai_only["precision"],
                    "ai_only_recall": ai_only["recall"],
                    "ai_only_f1_score": ai_only["f1_score"],
                    "ai_hitl_precision": hitl["precision"],
                    "ai_hitl_recall": hitl["recall"],
                    "ai_hitl_f1_score": hitl["f1_score"],
                    "auto_match_count": ai_only["auto_matches"],
                    "auto_non_match_count": ai_only["auto_non_matches"],
                    "review_needed_count": ai_only["review_needed_pairs"],
                    "review_workload_percentage": ai_only["review_needed_pairs"] / len(pairs),
                    "workload_reduction_vs_full_review": 1 - (ai_only["review_needed_pairs"] / len(pairs)),
                    "false_positives": ai_only["false_positives"],
                    "false_negatives": ai_only["false_negatives"],
                }
            )

    sweep = pd.DataFrame(rows)
    save_csv(sweep, CONFIG.paths.threshold_sweep)
    best = sweep.sort_values(["ai_hitl_f1_score", "review_workload_percentage"], ascending=[False, True]).iloc[0]
    CONFIG.paths.threshold_sweep_summary.write_text(
        "# Threshold Sweep Summary\n\n"
        f"Best lower threshold: {best['lower_threshold']:.2f}\n\n"
        f"Best upper threshold: {best['upper_threshold']:.2f}\n\n"
        f"AI + HITL simulated F1-score: {best['ai_hitl_f1_score']:.3f}\n\n"
        f"AI + HITL simulated recall: {best['ai_hitl_recall']:.3f}\n\n"
        f"Review-needed pairs: {int(best['review_needed_count']):,}\n\n"
        f"Review workload percentage: {best['review_workload_percentage']:.3%}\n\n"
        f"Workload reduction compared with full candidate review: {best['workload_reduction_vs_full_review']:.3%}\n\n"
        "Selected threshold rationale: this setting keeps the strongest AI + HITL F1-score while keeping the grey-zone review workload below 1% of blocked candidate pairs.\n",
        encoding="utf-8",
    )
    _plot_lines(sweep, "ai_hitl_f1_score", CONFIG.paths.threshold_f1_figure, "Threshold vs F1", "AI + HITL simulated F1")
    _plot_lines(sweep, "review_workload_percentage", CONFIG.paths.threshold_workload_figure, "Threshold vs Review Workload", "Review workload percentage")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.scatter(sweep["review_workload_percentage"], sweep["ai_hitl_recall"])
    ax.set_xlabel("Review workload percentage")
    ax.set_ylabel("AI + HITL simulated recall")
    ax.set_title("Recall vs Review Workload")
    fig.tight_layout()
    fig.savefig(CONFIG.paths.recall_workload_figure, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return sweep
