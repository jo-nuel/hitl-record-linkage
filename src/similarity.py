import pandas as pd
from rapidfuzz import fuzz

from .config import CONFIG
from .utils import exact_similarity, partial_postcode_similarity


def fuzzy_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def build_pairwise_dataset(
    records_df: pd.DataFrame, candidate_pairs_df: pd.DataFrame
) -> pd.DataFrame:
    left_cols = {col: f"{col}_a" for col in records_df.columns if col != "record_id"}
    right_cols = {col: f"{col}_b" for col in records_df.columns if col != "record_id"}

    left_df = records_df.rename(columns=left_cols)
    right_df = records_df.rename(columns=right_cols)

    pairs = candidate_pairs_df.merge(
        left_df,
        left_on="record_id_a",
        right_on="record_id",
        how="left",
    ).drop(columns=["record_id"])

    pairs = pairs.merge(
        right_df,
        left_on="record_id_b",
        right_on="record_id",
        how="left",
    ).drop(columns=["record_id"])

    return pairs


def compute_similarity_features(pair_df: pd.DataFrame) -> pd.DataFrame:
    df = pair_df.copy()

    df["sim_first_name"] = df.apply(
        lambda row: fuzzy_similarity(row["first_name_a"], row["first_name_b"]), axis=1
    )
    df["sim_last_name"] = df.apply(
        lambda row: fuzzy_similarity(row["last_name_a"], row["last_name_b"]), axis=1
    )
    df["sim_dob"] = df.apply(
        lambda row: exact_similarity(row["date_of_birth_a"], row["date_of_birth_b"]),
        axis=1,
    )
    df["sim_address"] = df.apply(
        lambda row: fuzzy_similarity(row["address_a"], row["address_b"]), axis=1
    )
    df["sim_postcode"] = df.apply(
        lambda row: partial_postcode_similarity(row["postcode_a"], row["postcode_b"]),
        axis=1,
    )
    df["sim_gender"] = df.apply(
        lambda row: exact_similarity(row["gender_a"], row["gender_b"]), axis=1
    )

    cfg = CONFIG.similarity
    cfg.validate()

    df["overall_score"] = (
        df["sim_first_name"] * cfg.weight_first_name
        + df["sim_last_name"] * cfg.weight_last_name
        + df["sim_dob"] * cfg.weight_dob
        + df["sim_address"] * cfg.weight_address
        + df["sim_postcode"] * cfg.weight_postcode
        + df["sim_gender"] * cfg.weight_gender
    )

    return df
