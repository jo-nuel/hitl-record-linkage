import pandas as pd

from src.utils.config import CONFIG


def true_link_set(true_links: pd.MultiIndex) -> set[tuple[str, str]]:
    return {(str(left), str(right)) for left, right in true_links}


def add_ground_truth_labels(pairs: pd.DataFrame, true_links: pd.MultiIndex) -> pd.DataFrame:
    truth = true_link_set(true_links)
    labeled = pairs.copy()
    labeled["is_true_link"] = labeled.apply(
        lambda row: int((str(row["record_id_a"]), str(row["record_id_b"])) in truth),
        axis=1,
    )
    return labeled


def _metric_row(predicted: set[tuple[str, str]], truth: set[tuple[str, str]]) -> dict[str, float]:
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def _keys(df: pd.DataFrame, column: str, value: str) -> set[tuple[str, str]]:
    selected = df[df[column] == value]
    return set(zip(selected["record_id_a"].astype(str), selected["record_id_b"].astype(str)))


def evaluate_results(
    final_pairs: pd.DataFrame,
    true_links: pd.MultiIndex,
    blocking_stats: dict[str, float],
    runtime_seconds: float,
) -> pd.DataFrame:
    truth = true_link_set(true_links)
    candidate_count = len(final_pairs)
    review_needed = int((final_pairs["model_decision"] == "Needs Human Review").sum())
    auto_matches = int((final_pairs["model_decision"] == "Auto Match").sum())
    auto_non_matches = int((final_pairs["model_decision"] == "Auto Non-match").sum())
    review_truth = final_pairs[
        (final_pairs["model_decision"] == "Needs Human Review") & (final_pairs["is_true_link"] == 1)
    ]
    simulated_hitl = _keys(final_pairs, "model_decision", "Auto Match") | set(
        zip(review_truth["record_id_a"].astype(str), review_truth["record_id_b"].astype(str))
    )
    manual_blocked = set(
        zip(
            final_pairs[final_pairs["is_true_link"] == 1]["record_id_a"].astype(str),
            final_pairs[final_pairs["is_true_link"] == 1]["record_id_b"].astype(str),
        )
    )

    approach_specs = [
        ("manual_blocked_benchmark", manual_blocked, candidate_count, 0),
        ("ai_only", _keys(final_pairs, "model_decision", "Auto Match"), 0, review_needed),
        ("ai_hitl_simulated", simulated_hitl, review_needed, 0),
    ]

    rows = []
    for approach, predicted, reviewed, pending_count in approach_specs:
        row = {
            "approach": approach,
            **_metric_row(predicted, truth),
            "candidate_pairs": candidate_count,
            "true_links_total": len(truth),
            "true_links_retained_after_blocking": int(blocking_stats["true_links_retained"]),
            "missed_true_links_after_blocking": int(blocking_stats["true_links_missed"]),
            "blocking_recall": float(blocking_stats["blocking_recall"]),
            "blocking_reduction_ratio": float(blocking_stats["reduction_ratio"]),
            "review_needed_pairs": review_needed,
            "pairs_reviewed": reviewed,
            "review_workload_percent": reviewed / candidate_count if candidate_count else 0.0,
            "workload_reduction_vs_full_review": 1 - (reviewed / candidate_count) if candidate_count else 0.0,
            "auto_matches": auto_matches if approach != "manual_blocked_benchmark" else 0,
            "auto_non_matches": auto_non_matches if approach != "manual_blocked_benchmark" else 0,
            "review_pending_pairs": pending_count,
            "estimated_review_seconds": reviewed * CONFIG.matcher.manual_review_seconds_per_pair,
            "runtime_seconds": runtime_seconds,
        }
        rows.append(row)
    return pd.DataFrame(rows)
