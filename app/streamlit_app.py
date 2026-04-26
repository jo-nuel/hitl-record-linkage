import sys
from pathlib import Path

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.empi import run_experiment  # noqa: E402
from src.empi.comparison import interpret_evidence  # noqa: E402
from src.empi.hitl import (  # noqa: E402
    load_review_decisions,
    pending_review_queue,
    save_review_decisions,
    upsert_review_decision,
)
from src.utils.config import CONFIG  # noqa: E402


def _read_csv(path: Path, dtype: str | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if dtype:
        return pd.read_csv(path, dtype=dtype).fillna("")
    return pd.read_csv(path).fillna("")


@st.cache_data(show_spinner=False)
def _cached_csv(path_text: str, modified_time: float, dtype: str | None = None) -> pd.DataFrame:
    return _read_csv(Path(path_text), dtype)


def _load(path: Path, dtype: str | None = None) -> pd.DataFrame:
    return _cached_csv(str(path), path.stat().st_mtime if path.exists() else 0, dtype)


def _show_image(path: Path, caption: str) -> None:
    if not path.exists():
        st.info(f"Run the pipeline to generate {path.name}.")
        return
    try:
        st.image(str(path), caption=caption, use_container_width=True)
    except TypeError:
        st.image(str(path), caption=caption)


def _metric_cards() -> None:
    df_a = _load(CONFIG.paths.febrl_a, dtype=str)
    df_b = _load(CONFIG.paths.febrl_b, dtype=str)
    blocking = _load(CONFIG.paths.blocking_stats)
    queue = _load(CONFIG.paths.review_queue, dtype=str)
    metrics = _load(CONFIG.paths.evaluation_metrics)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Dataset", "FEBRL4")
    c2.metric("Records A", f"{len(df_a):,}")
    c3.metric("Records B", f"{len(df_b):,}")
    c4.metric("Review queue", f"{len(queue):,}")
    if not blocking.empty:
        c5.metric("Blocking recall", f"{float(blocking.iloc[0]['blocking_recall']) * 100:.1f}%")
    else:
        c5.metric("Blocking recall", "n/a")

    if not metrics.empty:
        ai = metrics[metrics["approach"] == "ai_only"].iloc[0]
        hitl = metrics[metrics["approach"] == "ai_hitl_simulated"].iloc[0]
        st.caption(
            f"AI-only F1: {ai['f1_score']:.3f}. "
            f"Simulated AI + HITL F1: {hitl['f1_score']:.3f}."
        )


def _overview() -> None:
    st.markdown("## Overview")
    st.write(
        "This dashboard demonstrates an EMPI-inspired record linkage workflow using FEBRL benchmark data. "
        "The system blocks candidate pairs, compares identity fields, scores match evidence, and sends uncertain pairs to human review."
    )
    _metric_cards()
    st.markdown("### Why HITL is used")
    st.write(
        "The model resolves high-confidence matches and non-matches automatically. "
        "Pairs in the uncertainty band go to a reviewer because field evidence is mixed or incomplete."
    )
    st.markdown("### Workflow")
    st.write(
        "FEBRL records -> preprocessing -> multi-pass blocking -> field comparison -> EMPI-style scoring -> threshold decision -> human review -> final links"
    )


def _dataset_profile() -> None:
    st.markdown("## Dataset Profile")
    if CONFIG.paths.dataset_profile.exists():
        st.markdown(CONFIG.paths.dataset_profile.read_text(encoding="utf-8"))
    else:
        st.info("Run the pipeline to generate the dataset profile.")


def _workflow() -> None:
    st.markdown("## EMPI Workflow")
    cols = st.columns(7)
    steps = [
        "Load FEBRL",
        "Preprocess",
        "Block",
        "Compare",
        "Score",
        "Review",
        "Evaluate",
    ]
    for col, step in zip(cols, steps):
        col.info(step)
    st.write(
        "The matcher uses ECM probability where available and blends it with a Hybrid EMPI-style evidence score. "
        "The hybrid score combines multiple identity fields and applies conflict penalties for major disagreements."
    )


def _matching_dashboard() -> None:
    st.markdown("## Matching Dashboard")
    _metric_cards()
    benchmark = _load(CONFIG.paths.benchmark_table)
    workload = _load(CONFIG.paths.workload_table)
    blocking = _load(CONFIG.paths.blocking_stats)
    if not blocking.empty:
        st.table(blocking)
    if not benchmark.empty:
        st.markdown("### Benchmark Metrics")
        st.table(benchmark)
    if not workload.empty:
        st.markdown("### Workload Metrics")
        st.table(workload)
    left, right = st.columns(2)
    with left:
        _show_image(CONFIG.paths.benchmark_figure, "Benchmark comparison.")
    with right:
        _show_image(CONFIG.paths.workload_figure, "Review workload comparison.")
    left, right = st.columns(2)
    with left:
        _show_image(CONFIG.paths.decision_distribution_figure, "Decision distribution.")
    with right:
        _show_image(CONFIG.paths.score_distribution_figure, "Score distribution.")


def _field_evidence(pair: pd.Series) -> pd.DataFrame:
    field_map = [
        ("Given name", "given_name", "given_name_sim"),
        ("Surname", "surname", "surname_sim"),
        ("Date of birth", "date_of_birth", "date_of_birth_exact"),
        ("Address", "address", "address_sim"),
        ("Suburb/Place", "suburb", "suburb_sim"),
        ("State", "state", "state_exact"),
        ("Postcode", "postcode", "postcode_exact"),
        ("Sex/Gender", "sex", "sex_exact"),
        ("Available identifier", "identifier", ""),
    ]
    rows = []
    for label, field, score_col in field_map:
        a_value = pair.get(f"{field}_a", "")
        b_value = pair.get(f"{field}_b", "")
        if str(a_value).strip() == "" and str(b_value).strip() == "" and (
            score_col == "" or score_col not in pair
        ):
            continue
        score = "" if score_col == "" or score_col not in pair else pair[score_col]
        interpretation = "Context only" if score == "" else interpret_evidence(a_value, b_value, float(score))
        rows.append(
            {
                "Field": label,
                "Record A": a_value,
                "Record B": b_value,
                "Similarity": "" if score == "" else f"{float(score):.3f}",
                "Interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def _review_queue() -> None:
    st.markdown("## Human Review Queue")
    queue = _load(CONFIG.paths.review_queue, dtype=str)
    decisions = load_review_decisions(CONFIG.paths.review_decisions)
    if queue.empty:
        st.warning("Run the pipeline to create a review queue.")
        return
    pending = pending_review_queue(queue, decisions)
    resolved = decisions[decisions["reviewer_decision"].isin(["Confirm Match", "Reject Match"])]
    skipped = decisions[decisions["reviewer_decision"] == "Skip"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Needs review", f"{len(queue):,}")
    c2.metric("Resolved", f"{len(resolved):,}")
    c3.metric("Skipped", f"{len(skipped):,}")
    c4.metric("Pending", f"{len(pending):,}")
    st.caption("Skip does not finalise a pair. It keeps the pair pending for later review.")
    if pending.empty:
        st.success("No pending review pairs remain.")
        return

    pair = pending.iloc[0]
    st.markdown("### Current Pair")
    m1, m2, m3 = st.columns(3)
    m1.metric("Model score", f"{float(pair['model_score']):.3f}")
    m2.metric("Hybrid EMPI score", f"{float(pair['hybrid_empi_score']):.3f}")
    m3.metric("Model decision", pair["model_decision"])
    st.info(pair["decision_reason"])

    left, right = st.columns(2)
    with left:
        st.markdown("#### Record A")
        st.write(f"Name: {pair['given_name_a']} {pair['surname_a']}")
        st.write(f"DOB: {pair['date_of_birth_a']}")
        st.write(f"Address: {pair['address_a']}")
        st.write(f"Location: {pair['suburb_a']}, {pair['state_a']} {pair['postcode_a']}")
    with right:
        st.markdown("#### Record B")
        st.write(f"Name: {pair['given_name_b']} {pair['surname_b']}")
        st.write(f"DOB: {pair['date_of_birth_b']}")
        st.write(f"Address: {pair['address_b']}")
        st.write(f"Location: {pair['suburb_b']}, {pair['state_b']} {pair['postcode_b']}")

    st.markdown("### Field Evidence")
    st.table(_field_evidence(pair))
    notes = st.text_area("Reviewer notes")
    a, b, c = st.columns(3)
    if a.button("Confirm Match", type="primary", use_container_width=True):
        updated = upsert_review_decision(decisions, pair, "Confirm Match", notes, CONFIG.matcher.lower_threshold, CONFIG.matcher.upper_threshold)
        save_review_decisions(updated, CONFIG.paths.review_decisions, CONFIG.paths.review_decisions_export)
        st.rerun()
    if b.button("Reject Match", use_container_width=True):
        updated = upsert_review_decision(decisions, pair, "Reject Match", notes, CONFIG.matcher.lower_threshold, CONFIG.matcher.upper_threshold)
        save_review_decisions(updated, CONFIG.paths.review_decisions, CONFIG.paths.review_decisions_export)
        st.rerun()
    if c.button("Skip", use_container_width=True):
        updated = upsert_review_decision(decisions, pair, "Skip", notes, CONFIG.matcher.lower_threshold, CONFIG.matcher.upper_threshold)
        save_review_decisions(updated, CONFIG.paths.review_decisions, CONFIG.paths.review_decisions_export)
        st.rerun()


def _threshold_analysis() -> None:
    st.markdown("## Threshold Analysis")
    sweep = _load(CONFIG.paths.threshold_sweep)
    if not sweep.empty:
        st.dataframe(sweep)
    else:
        st.info("Run python scripts/run_threshold_sweep.py to generate threshold analysis.")
    _show_image(CONFIG.paths.threshold_f1_figure, "Threshold vs F1.")
    _show_image(CONFIG.paths.threshold_workload_figure, "Threshold vs review workload.")
    _show_image(CONFIG.paths.recall_workload_figure, "Recall vs review workload.")
    if CONFIG.paths.threshold_sweep_summary.exists():
        st.markdown(CONFIG.paths.threshold_sweep_summary.read_text(encoding="utf-8"))


def _report_outputs() -> None:
    st.markdown("## Report Outputs")
    for path in [
        CONFIG.paths.problem_formulation,
        CONFIG.paths.methodology_summary,
        CONFIG.paths.evaluation_summary,
        CONFIG.paths.limitations,
        CONFIG.paths.weekly_reflection_change_summary,
    ]:
        if path.exists():
            with st.expander(path.name):
                st.markdown(path.read_text(encoding="utf-8"))


def main() -> None:
    st.set_page_config(page_title="EMPI HITL Record Linkage", layout="wide")
    st.title("AI-Assisted HITL Record Linkage")
    st.caption("FEBRL benchmark, EMPI-inspired scoring, and operational review of ambiguous links.")

    with st.sidebar:
        st.header("Controls")
        mode = st.selectbox("Review mode", ["merge", "simulate", "ignore"])
        lower = st.slider("Lower threshold", 0.0, 1.0, CONFIG.matcher.lower_threshold, 0.05)
        upper = st.slider("Upper threshold", 0.0, 1.0, CONFIG.matcher.upper_threshold, 0.05)
        if st.button("Run pipeline", type="primary", use_container_width=True):
            with st.spinner("Running FEBRL linkage pipeline..."):
                run_experiment(mode, lower, upper)
            st.cache_data.clear()
            st.success("Pipeline complete.")
        if st.button("Reset review decisions", use_container_width=True):
            save_review_decisions(pd.DataFrame(), CONFIG.paths.review_decisions, CONFIG.paths.review_decisions_export)
            st.cache_data.clear()
            st.rerun()

    page = st.radio(
        "Page",
        [
            "Overview",
            "Dataset Profile",
            "EMPI Workflow",
            "Matching Dashboard",
            "Human Review Queue",
            "Threshold Analysis",
            "Report Outputs",
        ],
        horizontal=True,
    )
    {
        "Overview": _overview,
        "Dataset Profile": _dataset_profile,
        "EMPI Workflow": _workflow,
        "Matching Dashboard": _matching_dashboard,
        "Human Review Queue": _review_queue,
        "Threshold Analysis": _threshold_analysis,
        "Report Outputs": _report_outputs,
    }[page]()


if __name__ == "__main__":
    main()
