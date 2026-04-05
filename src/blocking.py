from collections import defaultdict
from typing import Iterable, List

import pandas as pd

from .config import CONFIG
from .utils import deduplicate_sorted_pairs


def _iter_block_pairs(
    record_ids: Iterable[str], max_candidates_per_record: int
) -> list[tuple[str, str]]:
    ids = sorted(set(record_ids))
    counts: defaultdict[str, int] = defaultdict(int)
    pairs: list[tuple[str, str]] = []

    for index, record_id_a in enumerate(ids):
        if counts[record_id_a] >= max_candidates_per_record:
            continue

        for record_id_b in ids[index + 1 :]:
            if counts[record_id_a] >= max_candidates_per_record:
                break
            if counts[record_id_b] >= max_candidates_per_record:
                continue

            pairs.append((record_id_a, record_id_b))
            counts[record_id_a] += 1
            counts[record_id_b] += 1

    return pairs


def _group_block_pairs(
    df: pd.DataFrame, block_cols: List[str], block_rule_name: str
) -> pd.DataFrame:
    valid_df = df.copy()

    for col in block_cols:
        valid_df = valid_df[valid_df[col].fillna("").astype(str) != ""]

    if valid_df.empty:
        return pd.DataFrame(columns=["record_id_a", "record_id_b", "block_rule"])

    max_candidates = CONFIG.blocking.max_candidates_per_record
    pair_rows = []

    for _, group in valid_df.groupby(block_cols, sort=False, dropna=False):
        block_pairs = _iter_block_pairs(group["record_id"].tolist(), max_candidates)
        pair_rows.extend(
            {
                "record_id_a": record_id_a,
                "record_id_b": record_id_b,
                "block_rule": block_rule_name,
            }
            for record_id_a, record_id_b in block_pairs
        )

    return pd.DataFrame(pair_rows, columns=["record_id_a", "record_id_b", "block_rule"])


def generate_candidate_pairs(df: pd.DataFrame) -> pd.DataFrame:
    candidate_frames = []

    if CONFIG.blocking.use_birth_year:
        candidate_frames.append(
            _group_block_pairs(
                df=df,
                block_cols=["birth_year"],
                block_rule_name="birth_year",
            )
        )

    if CONFIG.blocking.use_name_initial:
        candidate_frames.append(
            _group_block_pairs(
                df=df,
                block_cols=["first_initial", "last_name"],
                block_rule_name="first_initial_last_name",
            )
        )

    if CONFIG.blocking.use_postcode:
        candidate_frames.append(
            _group_block_pairs(
                df=df,
                block_cols=["postcode"],
                block_rule_name="postcode",
            )
        )

    if not candidate_frames:
        raise ValueError("No blocking rules are enabled in config.")

    all_pairs = pd.concat(candidate_frames, ignore_index=True)
    pairs = deduplicate_sorted_pairs(
        all_pairs[["record_id_a", "record_id_b"]], "record_id_a", "record_id_b"
    )

    block_rule_map = (
        all_pairs
        .groupby(["record_id_a", "record_id_b"])["block_rule"]
        .apply(lambda values: "|".join(sorted(set(values))))
        .reset_index()
    )

    pairs = pairs.merge(block_rule_map, on=["record_id_a", "record_id_b"], how="left")
    return pairs
