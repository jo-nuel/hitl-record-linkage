from typing import List

import pandas as pd

from .config import CONFIG
from .utils import deduplicate_sorted_pairs


def _self_join_pairs(
    df: pd.DataFrame, block_cols: List[str], block_rule_name: str
) -> pd.DataFrame:
    valid_df = df.copy()

    for col in block_cols:
        valid_df = valid_df[valid_df[col].fillna("").astype(str) != ""]

    if valid_df.empty:
        return pd.DataFrame(columns=["record_id_a", "record_id_b", "block_rule"])

    merged = valid_df.merge(valid_df, on=block_cols, suffixes=("_a", "_b"))
    merged = merged[merged["record_id_a"] < merged["record_id_b"]].copy()
    merged["block_rule"] = block_rule_name

    return merged[["record_id_a", "record_id_b", "block_rule"]]


def generate_candidate_pairs(df: pd.DataFrame) -> pd.DataFrame:
    candidate_frames = []

    if CONFIG.blocking.use_birth_year:
        candidate_frames.append(
            _self_join_pairs(
                df=df,
                block_cols=["birth_year"],
                block_rule_name="birth_year",
            )
        )

    if CONFIG.blocking.use_name_initial:
        candidate_frames.append(
            _self_join_pairs(
                df=df,
                block_cols=["first_initial", "last_name"],
                block_rule_name="first_initial_last_name",
            )
        )

    if CONFIG.blocking.use_postcode:
        candidate_frames.append(
            _self_join_pairs(
                df=df,
                block_cols=["postcode"],
                block_rule_name="postcode",
            )
        )

    if not candidate_frames:
        raise ValueError("No blocking rules are enabled in config.")

    pairs = pd.concat(candidate_frames, ignore_index=True)
    pairs = deduplicate_sorted_pairs(pairs, "record_id_a", "record_id_b")

    block_rule_map = (
        pd.concat(candidate_frames, ignore_index=True)
        .groupby(["record_id_a", "record_id_b"])["block_rule"]
        .apply(lambda values: "|".join(sorted(set(values))))
        .reset_index()
    )

    pairs = pairs.merge(block_rule_map, on=["record_id_a", "record_id_b"], how="left")
    return pairs
