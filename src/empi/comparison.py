import pandas as pd
import recordlinkage


FEATURE_FIELDS = [
    "given_name",
    "surname",
    "date_of_birth",
    "address",
    "suburb",
    "state",
    "postcode",
    "sex",
]


def _has_values(df_a: pd.DataFrame, df_b: pd.DataFrame, field: str) -> bool:
    return (
        field in df_a.columns
        and field in df_b.columns
        and df_a[field].astype(str).str.strip().ne("").any()
        and df_b[field].astype(str).str.strip().ne("").any()
    )


def compute_comparison_features(
    candidate_pairs: pd.MultiIndex,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> pd.DataFrame:
    compare = recordlinkage.Compare()
    labels: list[str] = []

    for field in ["given_name", "surname", "address", "suburb"]:
        if _has_values(df_a, df_b, field):
            label = f"{field}_sim"
            compare.string(field, field, method="jarowinkler", label=label)
            labels.append(label)

    for field in ["date_of_birth", "state", "postcode", "sex"]:
        if _has_values(df_a, df_b, field):
            label = f"{field}_exact"
            compare.exact(field, field, label=label)
            labels.append(label)

    if not labels:
        raise ValueError("No comparable FEBRL identity fields were found.")

    return compare.compute(candidate_pairs, df_a, df_b).fillna(0.0)[labels]


def interpret_evidence(value_a: str, value_b: str, similarity: float) -> str:
    a_missing = str(value_a).strip() == ""
    b_missing = str(value_b).strip() == ""
    if a_missing and b_missing:
        return "Missing in both records"
    if a_missing or b_missing:
        return "Missing in one record"
    if similarity >= 0.90:
        return "Strong agreement"
    if similarity >= 0.60:
        return "Partial agreement"
    return "Disagreement"
