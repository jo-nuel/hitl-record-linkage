import re
import string
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_directories_exist(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def safe_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_text(
    value: Any, lowercase: bool = True, remove_punctuation: bool = True
) -> str:
    text = safe_str(value)

    if lowercase:
        text = text.lower()

    if remove_punctuation:
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_phone(value: Any) -> str:
    text = safe_str(value)
    return re.sub(r"\D", "", text)


def normalize_postcode(value: Any) -> str:
    return safe_str(value).strip()


def parse_date_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d").fillna("")


def exact_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return 1.0 if a == b else 0.0


def partial_postcode_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) >= 3 and len(b) >= 3 and a[:3] == b[:3]:
        return 0.7
    return 0.0


def get_birth_year(date_str: str) -> str:
    if not date_str:
        return ""
    return date_str[:4]


def deduplicate_sorted_pairs(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    ordered = df.copy()
    ordered[[col_a, col_b]] = np.sort(ordered[[col_a, col_b]], axis=1)
    ordered = ordered[ordered[col_a] != ordered[col_b]]
    ordered = ordered.drop_duplicates(subset=[col_a, col_b]).reset_index(drop=True)
    return ordered
