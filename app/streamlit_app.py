import sys
from pathlib import Path

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import CONFIG  # noqa: E402
from src.pipeline import run_experiment  # noqa: E402
from src.review_store import (  # noqa: E402
    get_pending_review_queue,
    load_review_decisions,
    save_review_decisions,
    upsert_review_decision,
)


def _load_review_queue() -> pd.DataFrame:
    if not CONFIG.paths.review_queue.exists():
        return pd.DataFrame()
    return pd.read_csv(CONFIG.paths.review_queue, dtype=str).fillna("")


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


def main() -> None:
    st.set_page_config(page_title="HITL Record Linkage Review", layout="wide")
    st.title("HITL Record Linkage Review")
    st.caption(
        "Operational HITL loop: uncertain record pairs are reviewed one pair at a time and saved to CSV."
    )

    with st.sidebar:
        st.header("Pipeline Controls")
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

        st.markdown("### Files")
        st.write(f"Review queue: `{CONFIG.paths.review_queue}`")
        st.write(f"Manual decisions: `{CONFIG.paths.review_decisions}`")
        st.write(f"Final decisions: `{CONFIG.paths.final_decisions}`")

    if not CONFIG.paths.review_queue.exists():
        st.warning(
            "No review queue found yet. Use the sidebar button to generate or refresh pipeline outputs."
        )
        return

    review_queue_df = _load_review_queue()
    review_decisions_df = load_review_decisions(CONFIG.paths.review_decisions)
    pending_df = get_pending_review_queue(review_queue_df, review_decisions_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Uncertain pairs", len(review_queue_df))
    col2.metric("Pending review", len(pending_df))
    col3.metric("Saved decisions", len(review_decisions_df))
    col4.metric(
        "Completed rate",
        f"{(len(review_decisions_df) / len(review_queue_df) * 100):.1f}%"
        if len(review_queue_df)
        else "0.0%",
    )

    if pending_df.empty:
        st.success(
            "There are no pending uncertain pairs. Re-run the pipeline to refresh results or inspect the saved CSV outputs."
        )
        return

    pair = pending_df.iloc[0]
    st.subheader("Current Uncertain Pair")
    st.write(
        {
            "record_id_a": pair["record_id_a"],
            "record_id_b": pair["record_id_b"],
            "overall_score": round(float(pair["overall_score"]), 4),
            "block_rule": pair["block_rule"],
            "queue_position": f"1 of {len(pending_df)} pending pairs",
        }
    )

    similarity_df = pd.DataFrame(
        [
            {
                "field": "first_name",
                "similarity": float(pair["sim_first_name"]),
            },
            {
                "field": "last_name",
                "similarity": float(pair["sim_last_name"]),
            },
            {
                "field": "date_of_birth",
                "similarity": float(pair["sim_dob"]),
            },
            {
                "field": "address",
                "similarity": float(pair["sim_address"]),
            },
            {
                "field": "postcode",
                "similarity": float(pair["sim_postcode"]),
            },
            {
                "field": "gender",
                "similarity": float(pair["sim_gender"]),
            },
        ]
    )
    st.markdown("### Similarity Scores")
    st.dataframe(similarity_df, hide_index=True, use_container_width=True)

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("### Record A")
        st.write(
            {
                "first_name": pair["first_name_a"],
                "last_name": pair["last_name_a"],
                "date_of_birth": pair["date_of_birth_a"],
                "gender": pair["gender_a"],
                "address": pair["address_a"],
                "city": pair["city_a"],
                "state": pair["state_a"],
                "postcode": pair["postcode_a"],
            }
        )

    with right_col:
        st.markdown("### Record B")
        st.write(
            {
                "first_name": pair["first_name_b"],
                "last_name": pair["last_name_b"],
                "date_of_birth": pair["date_of_birth_b"],
                "gender": pair["gender_b"],
                "address": pair["address_b"],
                "city": pair["city_b"],
                "state": pair["state_b"],
                "postcode": pair["postcode_b"],
            }
        )

    action_col1, action_col2, action_col3 = st.columns(3)

    if action_col1.button("Confirm Match", use_container_width=True):
        _save_manual_decision(pair["record_id_a"], pair["record_id_b"], "Confirm Match")
        st.rerun()

    if action_col2.button("Reject Match", use_container_width=True):
        _save_manual_decision(pair["record_id_a"], pair["record_id_b"], "Reject Match")
        st.rerun()

    if action_col3.button("Skip", use_container_width=True):
        _save_manual_decision(pair["record_id_a"], pair["record_id_b"], "Skip")
        st.rerun()


if __name__ == "__main__":
    main()
