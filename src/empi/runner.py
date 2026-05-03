import time

import pandas as pd

from src.data.febrl_loader import load_febrl_dataset
from src.empi.blocking import generate_candidate_pairs
from src.empi.comparison import compute_comparison_features
from src.empi.hitl import (
    apply_review_decisions,
    build_review_queue,
    load_review_decisions,
    save_review_decisions,
)
from src.empi.matcher import classify_pairs
from src.empi.preprocessing import preprocess_records
from src.evaluation.metrics import add_ground_truth_labels, evaluate_results
from src.evaluation.report_outputs import generate_report_outputs
from src.utils.config import CONFIG
from src.utils.io import ensure_directories_exist, save_csv


VALID_REVIEW_MODES = {"merge", "simulate", "ignore"}


def _save_true_links(true_links: pd.MultiIndex) -> None:
    """Save FEBRL ground-truth links for internal evaluation traceability."""
    save_csv(
        pd.DataFrame([{"record_id_a": left, "record_id_b": right} for left, right in true_links]),
        CONFIG.paths.true_links,
    )


def _attach_record_fields(classified_features: pd.DataFrame, df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """Attach readable record fields so the Streamlit reviewer can inspect each pair."""
    pairs = classified_features.reset_index()
    pairs = pairs.rename(columns={pairs.columns[0]: "record_id_a", pairs.columns[1]: "record_id_b"})
    fields = [
        "given_name",
        "surname",
        "date_of_birth",
        "address",
        "suburb",
        "state",
        "postcode",
        "sex",
        "identifier",
    ]
    a_fields = df_a[fields].add_suffix("_a").reset_index().rename(columns={"record_id": "record_id_a"})
    b_fields = df_b[fields].add_suffix("_b").reset_index().rename(columns={"record_id": "record_id_b"})
    return pairs.merge(a_fields, on="record_id_a", how="left").merge(b_fields, on="record_id_b", how="left")


def _simulated_review_decisions(review_queue: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    """Simulate an ideal reviewer by resolving grey-zone pairs with FEBRL ground truth."""
    rows = []
    for _, pair in review_queue.iterrows():
        reviewer_decision = "Confirm Match" if int(pair["is_true_link"]) == 1 else "Reject Match"
        rows.append(
            {
                "timestamp": "",
                "record_id_a": pair["record_id_a"],
                "record_id_b": pair["record_id_b"],
                "model_score": pair["model_score"],
                "model_decision": pair["model_decision"],
                "reviewer_decision": reviewer_decision,
                "final_decision": "Match" if reviewer_decision == "Confirm Match" else "Non-match",
                "notes": "Simulated using FEBRL true links",
                "lower_threshold": lower,
                "upper_threshold": upper,
                "given_name_sim": pair.get("given_name_sim", ""),
                "surname_sim": pair.get("surname_sim", ""),
                "date_of_birth_exact": pair.get("date_of_birth_exact", ""),
                "address_sim": pair.get("address_sim", ""),
                "suburb_sim": pair.get("suburb_sim", ""),
                "state_exact": pair.get("state_exact", ""),
                "postcode_exact": pair.get("postcode_exact", ""),
                "sex_exact": pair.get("sex_exact", ""),
            }
        )
    return pd.DataFrame(rows)


def _drop_ground_truth_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Remove internal labels before writing public-facing pair outputs."""
    return df.drop(columns=["is_true_link"], errors="ignore")


def run_experiment(
    review_mode: str = "merge",
    lower_threshold: float | None = None,
    upper_threshold: float | None = None,
) -> dict[str, object]:
    """Run the FEBRL EMPI pipeline and write evidence outputs for the prototype."""
    CONFIG.validate()
    if review_mode not in VALID_REVIEW_MODES:
        raise ValueError(f"review_mode must be one of {sorted(VALID_REVIEW_MODES)}")

    lower = CONFIG.matcher.lower_threshold if lower_threshold is None else lower_threshold
    upper = CONFIG.matcher.upper_threshold if upper_threshold is None else upper_threshold
    ensure_directories_exist(CONFIG.paths.processed_dir, CONFIG.paths.tables_dir, CONFIG.paths.reports_dir, CONFIG.paths.figures_dir)

    df_a_raw, df_b_raw, true_links = load_febrl_dataset(CONFIG.dataset.name)
    df_a = preprocess_records(df_a_raw)
    df_b = preprocess_records(df_b_raw)
    df_a.to_csv(CONFIG.paths.febrl_a)
    df_b.to_csv(CONFIG.paths.febrl_b)
    _save_true_links(true_links)

    start = time.perf_counter()
    candidate_pairs, blocking_stats = generate_candidate_pairs(df_a, df_b, true_links)
    features = compute_comparison_features(candidate_pairs, df_a, df_b)
    classified_features = classify_pairs(features, lower, upper, CONFIG.matcher.ecm_weight)
    classified_pairs = _attach_record_fields(classified_features, df_a, df_b)
    classified_pairs = add_ground_truth_labels(classified_pairs, true_links)
    runtime_seconds = time.perf_counter() - start

    review_queue_internal = build_review_queue(classified_pairs, include_ground_truth=True)
    review_queue = build_review_queue(classified_pairs, include_ground_truth=False)
    save_csv(_drop_ground_truth_for_export(classified_pairs), CONFIG.paths.classified_pairs)
    save_csv(review_queue, CONFIG.paths.review_queue)
    save_csv(pd.DataFrame([blocking_stats]), CONFIG.paths.blocking_stats)

    if review_mode == "simulate":
        review_decisions = _simulated_review_decisions(review_queue_internal, lower, upper)
        save_review_decisions(review_decisions, CONFIG.paths.simulated_review_decisions)
        save_review_decisions(pd.DataFrame(), CONFIG.paths.review_decisions_export)
    elif review_mode == "merge":
        review_decisions = load_review_decisions(CONFIG.paths.review_decisions)
        valid_keys = set(zip(classified_pairs["record_id_a"], classified_pairs["record_id_b"]))
        if not review_decisions.empty:
            review_decisions = review_decisions[
                review_decisions.apply(lambda row: (row["record_id_a"], row["record_id_b"]) in valid_keys, axis=1)
            ]
        save_review_decisions(pd.DataFrame(), CONFIG.paths.simulated_review_decisions)
        save_review_decisions(review_decisions, CONFIG.paths.review_decisions_export)
    else:
        review_decisions = pd.DataFrame()
        save_review_decisions(pd.DataFrame(), CONFIG.paths.simulated_review_decisions)
        save_review_decisions(review_decisions, CONFIG.paths.review_decisions_export)

    final_decisions = apply_review_decisions(classified_pairs, review_decisions)
    save_csv(_drop_ground_truth_for_export(final_decisions), CONFIG.paths.final_decisions)
    metrics = evaluate_results(final_decisions, true_links, blocking_stats, runtime_seconds)
    report_tables = generate_report_outputs(metrics, classified_pairs, final_decisions, review_decisions, blocking_stats)

    return {
        "df_a": df_a,
        "df_b": df_b,
        "true_links": true_links,
        "candidate_pairs": candidate_pairs,
        "classified_pairs": classified_pairs,
        "review_queue": review_queue,
        "review_decisions": review_decisions,
        "final_decisions": final_decisions,
        "metrics": metrics,
        "blocking_stats": blocking_stats,
        **report_tables,
    }
