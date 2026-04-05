import json
import time
from pathlib import Path

import pandas as pd

from .blocking import generate_candidate_pairs
from .classify import classify_pairs
from .config import CONFIG
from .evaluate import add_ground_truth_labels, build_ground_truth_pairs, compare_approaches
from .review_store import (
    apply_review_decisions,
    build_review_queue,
    load_review_decisions,
    save_review_decisions,
    save_review_queue,
    simulate_review_decisions,
)
from .similarity import build_pairwise_dataset, compute_similarity_features
from .synthetic_duplicates import run_pipeline as generate_synthetic_data
from .utils import ensure_directories_exist

VALID_REVIEW_MODES = {"merge", "simulate", "ignore"}


def _load_records(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed records file not found at {path}. Run data generation first."
        )
    return pd.read_csv(path, dtype=str).fillna("")


def _resolve_sample_size(sample_size: int | None) -> int | None:
    return CONFIG.duplicates.sample_size if sample_size is None else sample_size


def _current_generated_sample_size() -> int | None:
    if not CONFIG.paths.base_records.exists():
        return None

    base_df = pd.read_csv(CONFIG.paths.base_records, usecols=["record_id"])
    return len(base_df)


def _load_generation_manifest() -> dict:
    if not CONFIG.paths.generation_manifest.exists():
        return {}
    try:
        return json.loads(CONFIG.paths.generation_manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _ensure_generated_data(requested_sample_size: int | None, regenerate_data: bool) -> str:
    current_sample_size = _current_generated_sample_size()
    generation_reason = "reused_existing_generated_dataset"
    should_regenerate = regenerate_data

    required_paths = [
        CONFIG.paths.base_records,
        CONFIG.paths.synthetic_records,
        CONFIG.paths.ground_truth,
    ]
    if not all(path.exists() for path in required_paths):
        should_regenerate = True
        generation_reason = "generated_missing_processed_artifacts"

    if (
        not should_regenerate
        and requested_sample_size is not None
        and current_sample_size != requested_sample_size
    ):
        should_regenerate = True
        generation_reason = "regenerated_for_requested_sample_size"

    if should_regenerate:
        generate_synthetic_data(sample_size=requested_sample_size)

    return generation_reason


def _build_interpretation(metrics_df: pd.DataFrame, review_mode_label: str) -> str:
    metrics_lookup = {
        row["approach"]: row for _, row in metrics_df.iterrows()
    }
    ai_only = metrics_lookup["ai_only"]
    ai_human = metrics_lookup["ai_human_hitl"]

    if ai_human["pairs_reviewed"] == 0 and review_mode_label != "ground_truth_simulation":
        return (
            "The pipeline has identified uncertain pairs, but no human review decisions "
            "have been merged yet. This means the current AI + HITL row is behaving like "
            "AI-only until manual review is completed."
        )

    recall_gain = ai_human["recall"] - ai_only["recall"]
    reviewed_pairs = int(ai_human["pairs_reviewed"])
    uncertain_pairs = int(ai_only["uncertain_pairs"])
    return (
        "Initial analysis suggests that routing only the uncertain pairs to a reviewer "
        f"improves recall by {recall_gain:.3f} over AI-only while limiting human review "
        f"to {reviewed_pairs} of {uncertain_pairs} uncertain candidate pairs."
    )


def _write_experiment_summary(
    metrics_df: pd.DataFrame,
    path: Path,
    review_mode: str,
    sample_size: int | None,
    generation_reason: str,
    generation_manifest: dict,
    candidate_pair_count: int,
    review_queue_count: int,
) -> None:
    comparison_table = metrics_df.to_string(index=False)
    interpretation = _build_interpretation(metrics_df, review_mode)
    lines = [
        "# Experiment Summary",
        "",
        "Project scope: AI-assisted human-in-the-loop record linkage for duplicate patient records in healthcare datasets.",
        "",
        "## Run Configuration",
        "",
        f"Review mode: {review_mode}",
        f"Generation reason: {generation_reason}",
        f"Sample size: {sample_size if sample_size is not None else 'all available source records'}",
        f"Base records: {generation_manifest.get('base_record_count', 'unknown')}",
        f"Synthetic duplicates: {generation_manifest.get('synthetic_duplicate_count', 'unknown')}",
        f"Duplicate rate: {generation_manifest.get('duplicate_rate', 'unknown')}",
        f"Match threshold: {generation_manifest.get('match_threshold', CONFIG.thresholds.match_threshold)}",
        f"Non-match threshold: {generation_manifest.get('non_match_threshold', CONFIG.thresholds.non_match_threshold)}",
        "",
        "## Initial Experimental Analysis",
        "",
        f"- Candidate pairs generated after blocking: {candidate_pair_count}",
        f"- Uncertain pairs written to the review queue: {review_queue_count}",
        f"- Evaluation table compares `manual_only`, `ai_only`, and `ai_human_hitl`.",
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "## Full Comparison",
        "",
        "```",
        comparison_table,
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    regenerate_data: bool = False,
    review_mode: str | None = None,
    sample_size: int | None = None,
) -> dict[str, object]:
    CONFIG.validate()
    selected_review_mode = review_mode or CONFIG.review.default_review_mode
    if selected_review_mode not in VALID_REVIEW_MODES:
        raise ValueError(
            f"review_mode must be one of {sorted(VALID_REVIEW_MODES)}, got {selected_review_mode}"
        )

    requested_sample_size = _resolve_sample_size(sample_size)
    ensure_directories_exist(
        CONFIG.paths.processed_dir, CONFIG.paths.reviewed_dir, CONFIG.paths.results_dir
    )

    generation_reason = _ensure_generated_data(
        requested_sample_size=requested_sample_size,
        regenerate_data=regenerate_data,
    )
    generation_manifest = _load_generation_manifest()

    records_df = _load_records(CONFIG.paths.synthetic_records)
    ground_truth_df = build_ground_truth_pairs(records_df)

    start_time = time.perf_counter()
    candidate_pairs_df = generate_candidate_pairs(records_df)
    pairwise_df = build_pairwise_dataset(records_df, candidate_pairs_df)
    scored_pairs_df = compute_similarity_features(pairwise_df)
    classified_pairs_df = classify_pairs(scored_pairs_df)
    runtime_seconds = time.perf_counter() - start_time

    classified_pairs_df = add_ground_truth_labels(classified_pairs_df, ground_truth_df)
    review_queue_df = build_review_queue(classified_pairs_df)
    save_review_queue(review_queue_df, CONFIG.paths.review_queue)
    classified_pairs_df.to_csv(CONFIG.paths.classified_pairs, index=False)

    review_mode_label = "pending_manual_review"
    if selected_review_mode == "simulate" and CONFIG.review.allow_review_simulation:
        review_decisions_df = simulate_review_decisions(
            classified_pairs_df[
                classified_pairs_df["system_decision"] == "Review Needed"
            ][["record_id_a", "record_id_b", "is_duplicate"]].copy()
        )
        review_mode_label = "ground_truth_simulation"
    elif selected_review_mode == "merge":
        review_decisions_df = load_review_decisions(CONFIG.paths.review_decisions)
        if not review_decisions_df.empty:
            review_mode_label = "manual_review_file"
    else:
        review_decisions_df = pd.DataFrame(columns=["record_id_a", "record_id_b", "reviewer_decision", "review_source"])

    current_pair_keys = set(
        zip(classified_pairs_df["record_id_a"], classified_pairs_df["record_id_b"])
    )
    if not review_decisions_df.empty:
        review_decisions_df = review_decisions_df[
            review_decisions_df.apply(
                lambda row: (row["record_id_a"], row["record_id_b"]) in current_pair_keys,
                axis=1,
            )
        ].reset_index(drop=True)

    save_review_decisions(
        review_decisions_df,
        CONFIG.paths.review_decisions_results,
    )
    resolved_pairs_df = apply_review_decisions(classified_pairs_df, review_decisions_df)
    metrics_df = compare_approaches(resolved_pairs_df, ground_truth_df, runtime_seconds)
    resolved_pairs_df.to_csv(CONFIG.paths.final_decisions, index=False)
    metrics_df.to_csv(CONFIG.paths.evaluation_results, index=False)
    _write_experiment_summary(
        metrics_df=metrics_df,
        path=CONFIG.paths.experiment_summary,
        review_mode=review_mode_label,
        sample_size=requested_sample_size,
        generation_reason=generation_reason,
        generation_manifest=generation_manifest,
        candidate_pair_count=len(candidate_pairs_df),
        review_queue_count=len(review_queue_df),
    )

    return {
        "records": records_df,
        "candidate_pairs": candidate_pairs_df,
        "classified_pairs": classified_pairs_df,
        "review_queue": review_queue_df,
        "final_decisions": resolved_pairs_df,
        "metrics": metrics_df,
        "review_mode": review_mode_label,
    }
