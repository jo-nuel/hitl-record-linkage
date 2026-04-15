import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import CONFIG  # noqa: E402
from src.evaluate import build_ground_truth_pairs, compare_approaches  # noqa: E402
from src.pipeline import run_experiment  # noqa: E402
import src.reporting as reporting  # noqa: E402
from src.review_store import (  # noqa: E402
    apply_review_decisions,
    get_pending_review_queue,
    load_review_decisions,
    save_review_decisions,
    upsert_review_decision,
)


def _load_csv_if_exists(path: Path, *, dtype: str | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if dtype is None:
        return pd.read_csv(path)
    return pd.read_csv(path, dtype=dtype).fillna("")


@st.cache_data(show_spinner=False)
def _load_cached_csv(path_str: str, modified_time: float, dtype: str | None = None) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    if dtype is None:
        return pd.read_csv(path)
    return pd.read_csv(path, dtype=dtype).fillna("")


def _cached_csv(path: Path, *, dtype: str | None = None) -> pd.DataFrame:
    modified_time = path.stat().st_mtime if path.exists() else 0.0
    return _load_cached_csv(str(path), modified_time, dtype=dtype)


def _load_review_queue() -> pd.DataFrame:
    return _cached_csv(CONFIG.paths.review_queue, dtype=str)


def _load_generation_manifest() -> dict:
    if not CONFIG.paths.generation_manifest.exists():
        return {}
    try:
        return json.loads(CONFIG.paths.generation_manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _render_image(path: Path, caption: str) -> None:
    try:
        st.image(str(path), caption=caption, use_container_width=True)
    except TypeError:
        st.image(str(path), caption=caption)


def _config_path(name: str) -> Path | None:
    return getattr(CONFIG.paths, name, None)


def _format_int(value: int) -> str:
    return f"{int(value):,}"


def _format_percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{(numerator / denominator) * 100:.1f}%"


def _save_manual_decision(record_id_a: str, record_id_b: str, reviewer_decision: str) -> None:
    review_decisions_df = load_review_decisions(CONFIG.paths.review_decisions)
    updated_decisions_df = upsert_review_decision(
        review_decisions_df=review_decisions_df,
        record_id_a=record_id_a,
        record_id_b=record_id_b,
        reviewer_decision=reviewer_decision,
        review_source="manual_review",
    )
    save_review_decisions(
        updated_decisions_df,
        path=CONFIG.paths.review_decisions,
        export_path=CONFIG.paths.review_decisions_results,
    )


def _run_pipeline_from_app(sample_size: int, regenerate_data: bool) -> None:
    with st.spinner("Running the pipeline and refreshing the review queue..."):
        run_experiment(
            regenerate_data=regenerate_data,
            review_mode="merge",
            sample_size=sample_size,
        )


def _reset_review_decisions() -> None:
    empty_df = pd.DataFrame(
        columns=["record_id_a", "record_id_b", "reviewer_decision", "review_source"]
    )
    save_review_decisions(
        empty_df,
        path=CONFIG.paths.review_decisions,
        export_path=CONFIG.paths.review_decisions_results,
    )


def _build_pair_comparison_df(pair: pd.Series) -> pd.DataFrame:
    rows = [
        ("First name", pair["first_name_a"], pair["first_name_b"], float(pair["sim_first_name"])),
        ("Last name", pair["last_name_a"], pair["last_name_b"], float(pair["sim_last_name"])),
        ("Date of birth", pair["date_of_birth_a"], pair["date_of_birth_b"], float(pair["sim_dob"])),
        ("Gender", pair["gender_a"], pair["gender_b"], float(pair["sim_gender"])),
        ("Address", pair["address_a"], pair["address_b"], float(pair["sim_address"])),
        ("City", pair["city_a"], pair["city_b"], None),
        ("State", pair["state_a"], pair["state_b"], None),
        ("Postcode", pair["postcode_a"], pair["postcode_b"], float(pair["sim_postcode"])),
    ]

    comparison_df = pd.DataFrame(
        rows,
        columns=["Field", "Record A", "Record B", "Similarity"],
    )
    comparison_df["Similarity"] = comparison_df["Similarity"].apply(
        lambda value: "" if value is None or pd.isna(value) else f"{value:.3f}"
    )
    return comparison_df


def _load_runtime_seconds() -> float:
    metrics_df = _load_csv_if_exists(CONFIG.paths.evaluation_results)
    if metrics_df.empty or "runtime_seconds" not in metrics_df.columns:
        return 0.0
    ai_only_df = metrics_df[metrics_df["approach"] == "ai_only"]
    if ai_only_df.empty:
        return 0.0
    return float(ai_only_df.iloc[0]["runtime_seconds"])


@st.cache_data(show_spinner=False)
def _load_ground_truth(path_str: str, modified_time: float) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    records_df = pd.read_csv(path, dtype=str).fillna("")
    return build_ground_truth_pairs(records_df)


def _build_live_summary(review_decisions_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    classified_pairs_df = _cached_csv(CONFIG.paths.classified_pairs)
    if classified_pairs_df.empty:
        return {
            "classified_pairs": pd.DataFrame(),
            "resolved_pairs": pd.DataFrame(),
            "metrics": pd.DataFrame(),
            "benchmark_table": pd.DataFrame(),
            "workload_table": pd.DataFrame(),
            "decision_counts_table": pd.DataFrame(),
        }

    ground_truth_df = _load_ground_truth(
        str(CONFIG.paths.synthetic_records),
        CONFIG.paths.synthetic_records.stat().st_mtime if CONFIG.paths.synthetic_records.exists() else 0.0,
    )
    runtime_seconds = _load_runtime_seconds()
    resolved_pairs_df = apply_review_decisions(classified_pairs_df, review_decisions_df)
    metrics_df = compare_approaches(resolved_pairs_df, ground_truth_df, runtime_seconds)

    return {
        "classified_pairs": classified_pairs_df,
        "resolved_pairs": resolved_pairs_df,
        "metrics": metrics_df,
        "benchmark_table": reporting.build_benchmark_table(metrics_df),
        "workload_table": reporting.build_workload_table(metrics_df),
        "decision_counts_table": reporting.build_decision_counts_table(
            classified_pairs_df,
            resolved_pairs_df,
            review_decisions_df,
        ),
    }


def _render_identity_snapshot(title: str, prefix: str, pair: pd.Series) -> None:
    st.markdown(f"#### {title}")
    st.markdown(f"**Name:** {pair[f'first_name_{prefix}']} {pair[f'last_name_{prefix}']}")
    st.markdown(f"**Date of birth:** {pair[f'date_of_birth_{prefix}']}")
    st.markdown(f"**Gender:** {pair[f'gender_{prefix}']}")
    st.markdown(f"**Address:** {pair[f'address_{prefix}']}")
    st.markdown(
        f"**Location:** {pair[f'city_{prefix}']}, {pair[f'state_{prefix}']} {pair[f'postcode_{prefix}']}"
    )


def _render_overview(
    metrics_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    workload_df: pd.DataFrame,
    decision_counts_df: pd.DataFrame,
    generation_manifest: dict,
    classified_pairs_df: pd.DataFrame,
    review_queue_df: pd.DataFrame,
    pending_df: pd.DataFrame,
    review_decisions_df: pd.DataFrame,
    final_decisions_df: pd.DataFrame,
) -> None:
    candidate_pairs = len(classified_pairs_df)
    final_match_count = int((final_decisions_df["final_decision"] == "Match").sum()) if not final_decisions_df.empty else 0
    auto_match_count = int((classified_pairs_df["system_decision"] == "Match").sum()) if not classified_pairs_df.empty else 0
    auto_non_match_count = int((classified_pairs_df["system_decision"] == "Non-match").sum()) if not classified_pairs_df.empty else 0

    st.markdown("## Project Status")
    card1, card2, card3, card4, card5, card6 = st.columns(6)
    card1.metric("Candidate pairs", _format_int(candidate_pairs))
    card2.metric("Review needed", _format_int(len(review_queue_df)))
    card3.metric("Reviewed pairs", _format_int(len(review_decisions_df)))
    card4.metric("Pending review", _format_int(len(pending_df)))
    card5.metric("Auto matches", _format_int(auto_match_count))
    card6.metric("Final matches", _format_int(final_match_count))

    left_col, right_col = st.columns([1.15, 1])
    with left_col:
        st.markdown("### Current Run")
        base_records = generation_manifest.get("base_record_count", "unknown")
        synthetic_duplicates = generation_manifest.get("synthetic_duplicate_count", "unknown")
        duplicate_rate = generation_manifest.get("duplicate_rate", "unknown")
        st.markdown(
            f"- Base records: `{base_records}`\n"
            f"- Synthetic duplicates: `{synthetic_duplicates}`\n"
            f"- Duplicate rate: `{duplicate_rate}`\n"
            f"- Auto non-matches: `{_format_int(auto_non_match_count)}`\n"
            f"- Review completion: `{_format_percent(len(review_decisions_df), len(review_queue_df))}`"
        )
        if not benchmark_df.empty:
            st.markdown("### Benchmark Snapshot")
            st.table(benchmark_df)
            st.caption(
                "The manual review benchmark is simulated over the blocked candidate set. "
                "It is not a literal full manual-matching run across all possible pairs."
            )

    with right_col:
        st.markdown("### Operational HITL Loop")
        st.markdown(
            "1. AI scores candidate pairs.\n"
            "2. Threshold logic assigns Match, Non-match, or Review Needed.\n"
            "3. Uncertain pairs enter the review queue.\n"
            "4. A reviewer inspects one pair at a time.\n"
            "5. The decision is stored and the next pending pair is shown.\n"
            "6. Reviewed and automated decisions are merged into final outputs."
        )
        st.info(
            "This prototype uses an operational review loop. It does not retrain the matcher during the run."
        )

    if not workload_df.empty:
        st.markdown("### Workload Snapshot")
        st.table(workload_df)

    if not decision_counts_df.empty:
        with st.expander("Decision count summary", expanded=False):
            st.table(decision_counts_df)

    if not metrics_df.empty and "approach" in metrics_df.columns:
        ai_only_row = metrics_df[metrics_df["approach"] == "ai_only"]
        hitl_row = metrics_df[metrics_df["approach"] == "ai_human_hitl"]
        if not ai_only_row.empty and not hitl_row.empty:
            recall_gain = float(hitl_row.iloc[0]["recall"]) - float(ai_only_row.iloc[0]["recall"])
            st.caption(
                f"Current recall gain from AI-only to AI + HITL: {recall_gain:.3f}"
            )


def _render_charts() -> None:
    st.markdown("## Metrics and Charts")


def _render_live_charts(metrics_df: pd.DataFrame, classified_pairs_df: pd.DataFrame, resolved_pairs_df: pd.DataFrame) -> None:
    top_left, top_right = st.columns(2)
    with top_left:
        st.pyplot(reporting.build_metrics_figure(metrics_df))
    with top_right:
        st.pyplot(reporting.build_workload_figure(metrics_df))

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.pyplot(reporting.build_decision_distribution_figure(classified_pairs_df))
    with bottom_right:
        st.pyplot(reporting.build_similarity_distribution_figure(classified_pairs_df))

    st.pyplot(reporting.build_resolution_flow_figure(resolved_pairs_df))
    st.caption(
        "These charts are generated live from the current saved review decisions. "
        "They update after each decision when you revisit Overview or Metrics & Charts."
    )


def _render_review_queue(review_queue_df: pd.DataFrame, review_decisions_df: pd.DataFrame) -> None:
    st.markdown("## Review Queue")
    pending_df = get_pending_review_queue(review_queue_df, review_decisions_df)

    progress_ratio = 0.0 if len(review_queue_df) == 0 else len(review_decisions_df) / len(review_queue_df)
    st.progress(min(max(progress_ratio, 0.0), 1.0))
    st.caption(
        f"Queue progress: {_format_int(len(review_decisions_df))} reviewed out of {_format_int(len(review_queue_df))} uncertain pairs."
    )

    queue_col1, queue_col2, queue_col3 = st.columns(3)
    queue_col1.metric("Pending now", _format_int(len(pending_df)))
    queue_col2.metric("Completed", _format_int(len(review_decisions_df)))
    queue_col3.metric("Completion rate", _format_percent(len(review_decisions_df), len(review_queue_df)))

    if pending_df.empty:
        st.success(
            "There are no pending uncertain pairs. Re-run the pipeline to refresh results or inspect the saved outputs."
        )
        return

    pair = pending_df.iloc[0]
    overall_score = float(pair["overall_score"])

    st.markdown("### Current Pair")
    meta1, meta2, meta3, meta4 = st.columns(4)
    meta1.metric("Overall score", f"{overall_score:.3f}")
    meta2.metric("Queue position", f"1 / {len(pending_df)}")
    meta3.metric("Block rule", pair["block_rule"])
    meta4.metric("Pair IDs", f"{pair['record_id_a']} | {pair['record_id_b']}")
    st.progress(min(max(overall_score, 0.0), 1.0))

    snapshot_left, snapshot_right = st.columns(2)
    with snapshot_left:
        _render_identity_snapshot("Record A", "a", pair)
    with snapshot_right:
        _render_identity_snapshot("Record B", "b", pair)

    st.markdown("### Pair Comparison")
    st.table(_build_pair_comparison_df(pair))

    st.markdown("### Review Decision")
    st.caption("Choose the final human review outcome for this uncertain pair. The app will save the decision and advance to the next pending item.")
    action_col1, action_col2, action_col3 = st.columns(3)

    if action_col1.button("Confirm Match", type="primary", use_container_width=True):
        _save_manual_decision(pair["record_id_a"], pair["record_id_b"], "Confirm Match")
        st.rerun()

    if action_col2.button("Reject Match", use_container_width=True):
        _save_manual_decision(pair["record_id_a"], pair["record_id_b"], "Reject Match")
        st.rerun()

    if action_col3.button("Skip", use_container_width=True):
        _save_manual_decision(pair["record_id_a"], pair["record_id_b"], "Skip")
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="HITL Record Linkage Review", layout="wide")
    st.title("HITL Record Linkage Review")
    st.caption(
        "Research prototype for duplicate patient record detection using weighted similarity scoring and an operational human review loop."
    )

    with st.sidebar:
        st.header("Run or Load Results")
        sample_size = st.number_input(
            "Sample size",
            min_value=100,
            value=int(CONFIG.duplicates.sample_size or 5000),
            step=100,
        )
        regenerate_data = st.checkbox(
            "Regenerate synthetic data",
            value=False,
            help="Enable this when you want a fresh sample and duplicate set.",
        )
        if st.button("Run / Refresh Pipeline", type="primary"):
            _run_pipeline_from_app(sample_size=int(sample_size), regenerate_data=regenerate_data)
            st.success("Pipeline outputs refreshed.")

        if st.button("Reset Review Decisions", use_container_width=True):
            _reset_review_decisions()
            st.success("Saved manual review decisions were cleared.")
            st.rerun()

        st.markdown("### Demo Notes")
        st.markdown(
            "- Start on the overview tab.\n"
            "- Show the benchmark and workload charts.\n"
            "- Move to the review queue and classify one uncertain pair.\n"
            "- Re-run the pipeline if you want the final outputs refreshed."
        )

        st.markdown("### Output Files")
        st.write(f"Review queue: `{CONFIG.paths.review_queue}`")
        st.write(f"Manual decisions: `{CONFIG.paths.review_decisions}`")
        st.write(f"Evaluation metrics: `{CONFIG.paths.evaluation_results}`")

    review_queue_df = _load_review_queue()
    review_decisions_df = load_review_decisions(CONFIG.paths.review_decisions)
    generation_manifest = _load_generation_manifest()

    if review_queue_df.empty:
        st.warning(
            "No review queue found yet. Use the sidebar button to run the pipeline and generate presentation outputs."
        )
        return

    pending_df = get_pending_review_queue(review_queue_df, review_decisions_df)
    try:
        section = st.radio(
            "Section",
            ["Overview", "Metrics & Charts", "Review Queue"],
            horizontal=True,
            key="app_section",
        )
    except TypeError:
        section = st.radio(
            "Section",
            ["Overview", "Metrics & Charts", "Review Queue"],
            key="app_section",
        )

    live_summary = None
    if section in {"Overview", "Metrics & Charts"}:
        live_summary = _build_live_summary(review_decisions_df)

    if section == "Overview":
        _render_overview(
            metrics_df=live_summary["metrics"],
            benchmark_df=live_summary["benchmark_table"],
            workload_df=live_summary["workload_table"],
            decision_counts_df=live_summary["decision_counts_table"],
            generation_manifest=generation_manifest,
            classified_pairs_df=live_summary["classified_pairs"],
            review_queue_df=review_queue_df,
            pending_df=pending_df,
            review_decisions_df=review_decisions_df,
            final_decisions_df=live_summary["resolved_pairs"],
        )
    elif section == "Metrics & Charts":
        _render_live_charts(
            metrics_df=live_summary["metrics"],
            classified_pairs_df=live_summary["classified_pairs"],
            resolved_pairs_df=live_summary["resolved_pairs"],
        )
    else:
        _render_review_queue(review_queue_df, review_decisions_df)


if __name__ == "__main__":
    main()
