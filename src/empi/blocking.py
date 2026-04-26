import pandas as pd
import recordlinkage


def _has_values(df_a: pd.DataFrame, df_b: pd.DataFrame, *columns: str) -> bool:
    return all(
        column in df_a.columns
        and column in df_b.columns
        and (df_a[column].astype(str).str.len().sum() + df_b[column].astype(str).str.len().sum()) > 0
        for column in columns
    )


def generate_candidate_pairs(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    true_links: pd.MultiIndex | None = None,
) -> tuple[pd.MultiIndex, dict[str, float]]:
    indexer = recordlinkage.Index()
    applied_rules: list[str] = []

    if _has_values(df_a, df_b, "surname", "birth_year"):
        indexer.block(left_on=["surname", "birth_year"], right_on=["surname", "birth_year"])
        applied_rules.append("surname + birth_year")
    if _has_values(df_a, df_b, "postcode", "surname_initial"):
        indexer.block(left_on=["postcode", "surname_initial"], right_on=["postcode", "surname_initial"])
        applied_rules.append("postcode + surname_initial")
    if _has_values(df_a, df_b, "given_initial", "surname_initial"):
        indexer.block(left_on=["given_initial", "surname_initial"], right_on=["given_initial", "surname_initial"])
        applied_rules.append("given_initial + surname_initial")
    if _has_values(df_a, df_b, "date_of_birth"):
        indexer.block(left_on="date_of_birth", right_on="date_of_birth")
        applied_rules.append("date_of_birth")
    if _has_values(df_a, df_b, "postcode"):
        indexer.block(left_on="postcode", right_on="postcode")
        applied_rules.append("postcode")

    if not applied_rules:
        raise ValueError("No usable blocking fields found in the selected FEBRL dataset.")

    candidate_pairs = indexer.index(df_a, df_b).drop_duplicates()
    total_possible = len(df_a) * len(df_b)
    retained = len(candidate_pairs.intersection(true_links)) if true_links is not None else 0
    true_total = len(true_links) if true_links is not None else 0
    stats = {
        "total_possible_pairs": total_possible,
        "candidate_pairs": len(candidate_pairs),
        "reduction_ratio": 1 - (len(candidate_pairs) / total_possible if total_possible else 0),
        "true_links_total": true_total,
        "true_links_retained": retained,
        "true_links_missed": true_total - retained,
        "blocking_recall": retained / true_total if true_total else 0.0,
        "blocking_rules": "; ".join(applied_rules),
    }
    return candidate_pairs, stats
