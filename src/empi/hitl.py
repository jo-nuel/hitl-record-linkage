from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.utils.io import ensure_directories_exist


DECISION_COLUMNS = [
    "timestamp",
    "record_id_a",
    "record_id_b",
    "model_score",
    "model_decision",
    "reviewer_decision",
    "final_decision",
    "notes",
    "lower_threshold",
    "upper_threshold",
    "given_name_sim",
    "surname_sim",
    "date_of_birth_exact",
    "address_sim",
    "suburb_sim",
    "state_exact",
    "postcode_exact",
    "sex_exact",
]


def build_review_queue(classified_pairs: pd.DataFrame) -> pd.DataFrame:
    queue = classified_pairs[classified_pairs["model_decision"] == "Needs Human Review"].copy()
    return queue.sort_values("model_score", ascending=False).reset_index(drop=True)


def load_review_decisions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=DECISION_COLUMNS)
    decisions = pd.read_csv(path, dtype=str).fillna("")
    for column in DECISION_COLUMNS:
        if column not in decisions.columns:
            decisions[column] = ""
    return decisions.drop_duplicates(["record_id_a", "record_id_b"], keep="last")[DECISION_COLUMNS]


def save_review_decisions(decisions: pd.DataFrame, path: Path, export_path: Path | None = None) -> None:
    ensure_directories_exist(path.parent)
    for column in DECISION_COLUMNS:
        if column not in decisions.columns:
            decisions[column] = ""
    decisions[DECISION_COLUMNS].to_csv(path, index=False)
    if export_path is not None:
        ensure_directories_exist(export_path.parent)
        decisions[DECISION_COLUMNS].to_csv(export_path, index=False)


def upsert_review_decision(
    decisions: pd.DataFrame,
    pair: pd.Series,
    reviewer_decision: str,
    notes: str,
    lower_threshold: float,
    upper_threshold: float,
) -> pd.DataFrame:
    final_decision = {
        "Confirm Match": "Match",
        "Reject Match": "Non-match",
        "Skip": "Needs Human Review",
    }[reviewer_decision]
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "record_id_a": pair["record_id_a"],
        "record_id_b": pair["record_id_b"],
        "model_score": pair["model_score"],
        "model_decision": pair["model_decision"],
        "reviewer_decision": reviewer_decision,
        "final_decision": final_decision,
        "notes": notes,
        "lower_threshold": lower_threshold,
        "upper_threshold": upper_threshold,
        "given_name_sim": pair.get("given_name_sim", ""),
        "surname_sim": pair.get("surname_sim", ""),
        "date_of_birth_exact": pair.get("date_of_birth_exact", ""),
        "address_sim": pair.get("address_sim", ""),
        "suburb_sim": pair.get("suburb_sim", ""),
        "state_exact": pair.get("state_exact", ""),
        "postcode_exact": pair.get("postcode_exact", ""),
        "sex_exact": pair.get("sex_exact", ""),
    }
    updated = pd.concat([decisions, pd.DataFrame([row])], ignore_index=True)
    return updated.drop_duplicates(["record_id_a", "record_id_b"], keep="last")[DECISION_COLUMNS]


def pending_review_queue(queue: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    resolved = decisions[decisions["reviewer_decision"].isin(["Confirm Match", "Reject Match"])]
    resolved_keys = set(zip(resolved["record_id_a"], resolved["record_id_b"]))
    pending = queue[
        ~queue.apply(lambda row: (row["record_id_a"], row["record_id_b"]) in resolved_keys, axis=1)
    ].copy()
    skipped = decisions[decisions["reviewer_decision"] == "Skip"]
    skipped_keys = set(zip(skipped["record_id_a"], skipped["record_id_b"]))
    pending["was_skipped"] = pending.apply(
        lambda row: (row["record_id_a"], row["record_id_b"]) in skipped_keys,
        axis=1,
    )
    return pending.sort_values(["was_skipped", "model_score"], ascending=[True, False]).drop(
        columns=["was_skipped"]
    ).reset_index(drop=True)


def apply_review_decisions(classified_pairs: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    resolved = classified_pairs.copy()
    resolved = resolved.merge(
        decisions[["record_id_a", "record_id_b", "reviewer_decision", "final_decision", "notes"]]
        if not decisions.empty
        else pd.DataFrame(columns=["record_id_a", "record_id_b", "reviewer_decision", "final_decision", "notes"]),
        on=["record_id_a", "record_id_b"],
        how="left",
    )
    resolved["reviewer_decision"] = resolved["reviewer_decision"].fillna("")
    resolved["final_decision"] = resolved["model_decision"].map(
        {
            "Auto Match": "Match",
            "Auto Non-match": "Non-match",
            "Needs Human Review": "Needs Human Review",
        }
    )
    resolved.loc[resolved["reviewer_decision"] == "Confirm Match", "final_decision"] = "Match"
    resolved.loc[resolved["reviewer_decision"] == "Reject Match", "final_decision"] = "Non-match"
    resolved.loc[resolved["reviewer_decision"] == "Skip", "final_decision"] = "Needs Human Review"
    return resolved
