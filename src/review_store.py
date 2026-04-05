from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_directories_exist


REVIEW_DECISION_COLUMNS = [
    "record_id_a",
    "record_id_b",
    "reviewer_decision",
    "review_source",
]

QUEUE_COLUMNS = [
    "record_id_a",
    "record_id_b",
    "overall_score",
    "block_rule",
    "sim_first_name",
    "sim_last_name",
    "sim_dob",
    "sim_address",
    "sim_postcode",
    "sim_gender",
    "first_name_a",
    "last_name_a",
    "date_of_birth_a",
    "gender_a",
    "address_a",
    "city_a",
    "state_a",
    "postcode_a",
    "first_name_b",
    "last_name_b",
    "date_of_birth_b",
    "gender_b",
    "address_b",
    "city_b",
    "state_b",
    "postcode_b",
]


def _normalize_pair_order(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if normalized.empty:
        return normalized

    ordered = np.sort(normalized[["record_id_a", "record_id_b"]].astype(str), axis=1)
    normalized["record_id_a"] = ordered[:, 0]
    normalized["record_id_b"] = ordered[:, 1]
    return normalized


def build_review_queue(classified_pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Create the operational HITL queue from threshold-uncertain candidate pairs."""
    queue = classified_pairs_df[
        classified_pairs_df["system_decision"] == "Review Needed"
    ].copy()
    if queue.empty:
        return pd.DataFrame(columns=QUEUE_COLUMNS)
    return queue[QUEUE_COLUMNS].sort_values(
        by=["overall_score", "record_id_a", "record_id_b"], ascending=[False, True, True]
    )


def save_review_queue(review_queue_df: pd.DataFrame, path: Path) -> None:
    ensure_directories_exist(path.parent)
    review_queue_df.to_csv(path, index=False)


def load_review_decisions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=REVIEW_DECISION_COLUMNS)

    review_df = pd.read_csv(path, dtype=str).fillna("")
    missing_cols = [
        column
        for column in ["record_id_a", "record_id_b", "reviewer_decision"]
        if column not in review_df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Review decisions file is missing required columns: {', '.join(missing_cols)}"
        )

    if "review_source" not in review_df.columns:
        review_df["review_source"] = "manual_review"

    review_df = _normalize_pair_order(review_df)
    review_df = review_df.drop_duplicates(
        subset=["record_id_a", "record_id_b"], keep="last"
    )
    return review_df[REVIEW_DECISION_COLUMNS]


def save_review_decisions(
    review_decisions_df: pd.DataFrame,
    path: Path,
    export_path: Path | None = None,
) -> None:
    normalized = _normalize_pair_order(review_decisions_df)
    ensure_directories_exist(path.parent)
    normalized[REVIEW_DECISION_COLUMNS].to_csv(path, index=False)

    if export_path is not None:
        ensure_directories_exist(export_path.parent)
        normalized[REVIEW_DECISION_COLUMNS].to_csv(export_path, index=False)


def upsert_review_decision(
    review_decisions_df: pd.DataFrame,
    record_id_a: str,
    record_id_b: str,
    reviewer_decision: str,
    review_source: str = "manual_review",
) -> pd.DataFrame:
    new_row = pd.DataFrame(
        [
            {
                "record_id_a": record_id_a,
                "record_id_b": record_id_b,
                "reviewer_decision": reviewer_decision,
                "review_source": review_source,
            }
        ]
    )

    updated = pd.concat([review_decisions_df, new_row], ignore_index=True)
    updated = _normalize_pair_order(updated)
    updated = updated.drop_duplicates(
        subset=["record_id_a", "record_id_b"], keep="last"
    )
    return updated[REVIEW_DECISION_COLUMNS]


def get_pending_review_queue(
    review_queue_df: pd.DataFrame, review_decisions_df: pd.DataFrame
) -> pd.DataFrame:
    reviewed_keys = set(
        zip(review_decisions_df["record_id_a"], review_decisions_df["record_id_b"])
    )

    pending_df = review_queue_df[
        ~review_queue_df.apply(
            lambda row: (row["record_id_a"], row["record_id_b"]) in reviewed_keys, axis=1
        )
    ].reset_index(drop=True)
    return pending_df


def simulate_review_decisions(review_queue_df: pd.DataFrame) -> pd.DataFrame:
    simulated = review_queue_df[["record_id_a", "record_id_b", "is_duplicate"]].copy()
    simulated["reviewer_decision"] = simulated["is_duplicate"].astype(int).map(
        {1: "Confirm Match", 0: "Reject Match"}
    )
    simulated["review_source"] = "ground_truth_simulation"
    return simulated[REVIEW_DECISION_COLUMNS]


def apply_review_decisions(
    classified_pairs_df: pd.DataFrame, review_decisions_df: pd.DataFrame
) -> pd.DataFrame:
    resolved = classified_pairs_df.copy()
    decisions = _normalize_pair_order(review_decisions_df)

    resolved = resolved.merge(
        decisions,
        on=["record_id_a", "record_id_b"],
        how="left",
    )
    resolved["reviewer_decision"] = resolved["reviewer_decision"].fillna("")
    resolved["review_source"] = resolved["review_source"].fillna("")

    resolved["final_decision"] = resolved["system_decision"]
    resolved.loc[
        resolved["reviewer_decision"] == "Confirm Match", "final_decision"
    ] = "Match"
    resolved.loc[
        resolved["reviewer_decision"] == "Reject Match", "final_decision"
    ] = "Non-match"
    resolved.loc[
        (resolved["system_decision"] == "Review Needed")
        & (resolved["reviewer_decision"] == ""),
        "final_decision",
    ] = "Review Pending"
    resolved.loc[
        resolved["reviewer_decision"] == "Skip", "final_decision"
    ] = "Review Pending"

    resolved["review_status"] = "Not Reviewed"
    resolved.loc[
        resolved["reviewer_decision"] == "Confirm Match", "review_status"
    ] = "Reviewed: Confirmed"
    resolved.loc[
        resolved["reviewer_decision"] == "Reject Match", "review_status"
    ] = "Reviewed: Rejected"
    resolved.loc[
        resolved["reviewer_decision"] == "Skip", "review_status"
    ] = "Reviewed: Skipped"

    resolved["human_reviewed"] = resolved["reviewer_decision"].ne("")
    return resolved
