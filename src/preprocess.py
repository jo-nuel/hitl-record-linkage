from typing import Dict, Optional

import pandas as pd

from .config import CONFIG, ColumnConfig
from .utils import (
    get_birth_year,
    normalize_postcode,
    normalize_text,
    parse_date_series,
    safe_str,
)


CANONICAL_COLUMNS = [
    "record_id",
    "patient_id",
    "first_name",
    "last_name",
    "date_of_birth",
    "gender",
    "address",
    "city",
    "state",
    "postcode",
]


def get_default_column_mapping(columns: ColumnConfig) -> Dict[str, str]:
    return {
        columns.source_id: "record_id",
        columns.first_name: "first_name",
        columns.last_name: "last_name",
        columns.dob: "date_of_birth",
        columns.gender: "gender",
        columns.address: "address",
        columns.city: "city",
        columns.state: "state",
        columns.postcode: "postcode",
    }


def map_columns(
    df: pd.DataFrame, custom_mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    mapping = get_default_column_mapping(CONFIG.columns)
    if custom_mapping:
        mapping.update(custom_mapping)

    renamed = df.rename(columns=mapping).copy()

    if "record_id" not in renamed.columns:
        raise ValueError(
            "Input data must contain a record_id column or a mapped source ID column."
        )

    if "patient_id" not in renamed.columns:
        renamed["patient_id"] = renamed["record_id"]

    for column in CANONICAL_COLUMNS:
        if column not in renamed.columns:
            renamed[column] = ""

    return renamed[CANONICAL_COLUMNS].copy()


def preprocess_records(
    df: pd.DataFrame, custom_mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    records = map_columns(df, custom_mapping=custom_mapping)

    for column in ["record_id", "patient_id"]:
        records[column] = records[column].apply(safe_str)

    records["first_name_raw"] = records["first_name"].apply(safe_str)
    records["last_name_raw"] = records["last_name"].apply(safe_str)
    records["address_raw"] = records["address"].apply(safe_str)
    records["date_of_birth_raw"] = records["date_of_birth"].apply(safe_str)

    records["first_name"] = records["first_name"].apply(normalize_text)
    records["last_name"] = records["last_name"].apply(normalize_text)
    records["gender"] = records["gender"].apply(normalize_text)
    records["address"] = records["address"].apply(normalize_text)
    records["city"] = records["city"].apply(normalize_text)
    records["state"] = records["state"].apply(normalize_text)
    records["postcode"] = records["postcode"].apply(normalize_postcode)

    records["date_of_birth"] = parse_date_series(records["date_of_birth"])
    records["birth_year"] = records["date_of_birth"].apply(get_birth_year)
    records["first_initial"] = records["first_name"].str[:1].fillna("")
    records["full_name"] = (
        records["first_name"].fillna("").str.strip()
        + " "
        + records["last_name"].fillna("").str.strip()
    ).str.strip()

    return records
