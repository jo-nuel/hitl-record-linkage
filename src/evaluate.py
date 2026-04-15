from itertools import combinations

import pandas as pd

from .config import CONFIG


def build_ground_truth_pairs(records_df: pd.DataFrame) -> pd.DataFrame:
    truth_rows = []

    for duplicate_group_id, group in records_df.groupby("duplicate_group_id", dropna=False):
        record_ids = sorted(group["record_id"].astype(str).unique())
        if duplicate_group_id == "" or len(record_ids) < 2:
            continue

        for record_id_a, record_id_b in combinations(record_ids, 2):
            truth_rows.append(
                {
                    "record_id_a": record_id_a,
                    "record_id_b": record_id_b,
                    "is_duplicate": 1,
                    "duplicate_group_id": duplicate_group_id,
                }
            )

    return pd.DataFrame(
        truth_rows,
        columns=["record_id_a", "record_id_b", "is_duplicate", "duplicate_group_id"],
    )


def add_ground_truth_labels(
    candidate_pairs_df: pd.DataFrame, ground_truth_df: pd.DataFrame
) -> pd.DataFrame:
    labeled = candidate_pairs_df.copy()
    truth_keys = set(zip(ground_truth_df["record_id_a"], ground_truth_df["record_id_b"]))
    labeled["is_duplicate"] = labeled.apply(
        lambda row: int((row["record_id_a"], row["record_id_b"]) in truth_keys),
        axis=1,
    )
    return labeled


def _pair_metrics(
    predicted_match_keys: set[tuple[str, str]],
    truth_match_keys: set[tuple[str, str]],
) -> dict:
    tp = len(predicted_match_keys & truth_match_keys)
    fp = len(predicted_match_keys - truth_match_keys)
    fn = len(truth_match_keys - predicted_match_keys)
    tn = 0

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }


def _pairs_to_key_set(
    df: pd.DataFrame, decision_column: str, positive_value: str
) -> set[tuple[str, str]]:
    matched = df[df[decision_column] == positive_value]
    return set(zip(matched["record_id_a"], matched["record_id_b"]))


def _manual_baseline_keys(classified_pairs_df: pd.DataFrame) -> set[tuple[str, str]]:
    """
    Simulated clerical-review baseline for the research prototype.

    This assumes a human reviewer inspects every blocked candidate pair and correctly
    identifies every true duplicate within that candidate set. It is therefore a
    benchmark for full clerical review effort within the blocked candidate set, not a
    literal end-to-end manual matching workflow across all possible record pairs.
    """
    manual_matches = classified_pairs_df[classified_pairs_df["is_duplicate"] == 1]
    return set(zip(manual_matches["record_id_a"], manual_matches["record_id_b"]))


def compare_approaches(
    classified_pairs_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    runtime_seconds: float,
) -> pd.DataFrame:
    truth_match_keys = set(zip(ground_truth_df["record_id_a"], ground_truth_df["record_id_b"]))
    manual_baseline_keys = _manual_baseline_keys(classified_pairs_df)
    uncertain_pair_count = int(
        (classified_pairs_df["system_decision"] == "Review Needed").sum()
    )
    human_reviewed_count = int(classified_pairs_df["human_reviewed"].sum())
    manual_review_seconds = CONFIG.review.manual_review_seconds_per_pair
    auto_match_count = int((classified_pairs_df["system_decision"] == "Match").sum())
    auto_non_match_count = int(
        (classified_pairs_df["system_decision"] == "Non-match").sum()
    )
    final_match_count = int((classified_pairs_df["final_decision"] == "Match").sum())
    final_non_match_count = int(
        (classified_pairs_df["final_decision"] == "Non-match").sum()
    )
    final_pending_count = int(
        (classified_pairs_df["final_decision"] == "Review Pending").sum()
    )

    approach_specs = [
        {
            "approach": "manual_only",
            "decision_keys": manual_baseline_keys,
            "pairs_reviewed": len(classified_pairs_df),
            "auto_matches": 0,
            "auto_non_matches": 0,
            "final_matches": len(manual_baseline_keys),
            "final_non_matches": len(classified_pairs_df)
            - len(manual_baseline_keys),
            "review_pending_pairs": 0,
            "runtime_seconds": 0.0,
        },
        {
            "approach": "ai_only",
            "decision_keys": _pairs_to_key_set(classified_pairs_df, "system_decision", "Match"),
            "pairs_reviewed": 0,
            "auto_matches": auto_match_count,
            "auto_non_matches": auto_non_match_count,
            "final_matches": auto_match_count,
            "final_non_matches": auto_non_match_count,
            "review_pending_pairs": uncertain_pair_count,
            "runtime_seconds": runtime_seconds,
        },
        {
            "approach": "ai_human_hitl",
            "decision_keys": _pairs_to_key_set(classified_pairs_df, "final_decision", "Match"),
            "pairs_reviewed": human_reviewed_count,
            "auto_matches": auto_match_count,
            "auto_non_matches": auto_non_match_count,
            "final_matches": final_match_count,
            "final_non_matches": final_non_match_count,
            "review_pending_pairs": final_pending_count,
            "runtime_seconds": runtime_seconds,
        },
    ]

    rows = []
    for spec in approach_specs:
        metrics = _pair_metrics(spec["decision_keys"], truth_match_keys)
        estimated_review_seconds = spec["pairs_reviewed"] * manual_review_seconds
        rows.append(
            {
                "approach": spec["approach"],
                **metrics,
                "candidate_pairs": len(classified_pairs_df),
                "true_duplicate_pairs": len(truth_match_keys),
                "uncertain_pairs": uncertain_pair_count,
                "pairs_reviewed": spec["pairs_reviewed"],
                "auto_matches": spec["auto_matches"],
                "auto_non_matches": spec["auto_non_matches"],
                "final_matches": spec["final_matches"],
                "final_non_matches": spec["final_non_matches"],
                "review_pending_pairs": spec["review_pending_pairs"],
                "estimated_manual_review_seconds": estimated_review_seconds,
                "runtime_seconds": spec["runtime_seconds"],
                "estimated_total_effort_seconds": spec["runtime_seconds"]
                + estimated_review_seconds,
            }
        )

    return pd.DataFrame(rows)[
        [
            "approach",
            "precision",
            "recall",
            "f1_score",
            "true_positives",
            "false_positives",
            "false_negatives",
            "candidate_pairs",
            "true_duplicate_pairs",
            "uncertain_pairs",
            "pairs_reviewed",
            "auto_matches",
            "auto_non_matches",
            "final_matches",
            "final_non_matches",
            "review_pending_pairs",
            "estimated_manual_review_seconds",
            "runtime_seconds",
            "estimated_total_effort_seconds",
        ]
    ]
