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
from src.evaluation.active_learning import run_active_learning_experiment  # noqa: E402
from src.utils.config import CONFIG  # noqa: E402


MODEL_OPTIONS = [
    "Hybrid EMPI Score",
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
    "Best available model",
]


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


def _read_markdown(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _show_missing(path: Path, command: str) -> None:
    st.warning(f"`{path}` is not available yet.")
    st.code(command, language="bash")


def _show_image(path: Path, caption: str, command: str = "python scripts/run_pipeline.py --review-mode simulate") -> None:
    if not path.exists():
        _show_missing(path, command)
        return
    try:
        st.image(str(path), caption=caption, use_container_width=True)
    except TypeError:
        st.image(str(path), caption=caption)


def _metric_value(df: pd.DataFrame, column: str, default: str = "n/a") -> str:
    if df.empty or column not in df.columns:
        return default
    value = df.iloc[0][column]
    if isinstance(value, float):
        return f"{value:,.3f}"
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return str(value)


def _top_metric_cards() -> None:
    df_a = _load(CONFIG.paths.febrl_a, dtype=str)
    df_b = _load(CONFIG.paths.febrl_b, dtype=str)
    true_links = _load(CONFIG.paths.true_links, dtype=str)
    blocking = _load(CONFIG.paths.blocking_stats)
    active_rounds = _load(CONFIG.paths.active_learning_rounds)
    final_eval = _load(CONFIG.paths.final_evaluation_comparison)

    columns = st.columns(7)
    columns[0].metric("Dataset", "FEBRL4")
    columns[1].metric("Records A", f"{len(df_a):,}" if not df_a.empty else "n/a")
    columns[2].metric("Records B", f"{len(df_b):,}" if not df_b.empty else "n/a")
    columns[3].metric("True links", f"{len(true_links):,}" if not true_links.empty else "n/a")
    columns[4].metric("Candidate pairs", _metric_value(blocking, "candidate_pairs"))
    if not blocking.empty and "blocking_recall" in blocking.columns:
        columns[5].metric("Blocking recall", f"{float(blocking.iloc[0]['blocking_recall']) * 100:.1f}%")
    else:
        columns[5].metric("Blocking recall", "n/a")
    if not active_rounds.empty and "F1-score" in active_rounds.columns:
        columns[6].metric("Active-learning best F1", f"{active_rounds['F1-score'].max():.3f}")
    elif not final_eval.empty and "Method" in final_eval.columns:
        hitl = final_eval[final_eval["Method"] == "AI + HITL Grey-Zone Review"]
        columns[6].metric("AI + HITL F1", f"{float(hitl.iloc[0]['F1-score']):.3f}" if not hitl.empty else "n/a")
    else:
        columns[6].metric("Best F1", "n/a")


def _section_note(text: str) -> None:
    st.info(text)


def _workflow_cards(items: list[str]) -> None:
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        col.info(item)


def _overview() -> None:
    st.header("Overview")
    st.subheader("Active Learning EMPI-Inspired Record Linkage with HITL Review")
    st.write(
        "This dashboard presents a research prototype for detecting linked patient-style records. "
        "The main system is EMPI-inspired HITL record linkage: preprocessing, blocking, field-level comparison, "
        "threshold-based decision logic, and grey-zone human review. "
        "The active-learning ML matcher extends this workflow with match probabilities, "
        "uncertainty sampling, human review, and batch retraining."
    )
    st.info(
        "Method positioning: Hybrid EMPI Score is the transparent non-ML baseline and fallback. "
        "Active Learning ML is the main AI extension."
    )
    _section_note(
        "Research aim: evaluate whether active-learning HITL can improve linkage quality while reducing manual review workload."
    )
    _top_metric_cards()
    st.markdown("### Presentation Workflow")
    _workflow_cards(
        [
            "FEBRL4",
            "Preprocessing",
            "Blocking",
            "Field Comparison",
            "ML Match Probability",
            "Uncertainty Selection",
            "Human Review",
            "Batch Retraining",
            "Evaluation",
        ]
    )
    st.markdown("### Important Evaluation Note")
    st.write(
        "Formal benchmark HITL labels are simulated using FEBRL ground truth. "
        "Live review clicks demonstrate the review workflow and audit logging. "
        "A frozen test set must not be used for active-learning training."
    )


def _dataset_and_blocking() -> None:
    st.header("Dataset & Blocking")
    st.write(
        "Blocking reduces the search space from all possible record pairs to likely candidate pairs while trying to preserve true links."
    )
    profile = _read_markdown(CONFIG.paths.dataset_profile)
    if profile:
        with st.expander("FEBRL4 dataset profile", expanded=True):
            st.markdown(profile)
    else:
        _show_missing(CONFIG.paths.dataset_profile, "python scripts/run_pipeline.py --review-mode simulate")

    blocking = _load(CONFIG.paths.blocking_stats)
    if blocking.empty:
        _show_missing(CONFIG.paths.blocking_stats, "python scripts/run_pipeline.py --review-mode simulate")
    else:
        st.markdown("### Blocking Statistics")
        st.dataframe(blocking, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total possible pairs", _metric_value(blocking, "total_possible_pairs"))
        c2.metric("Candidate pairs", _metric_value(blocking, "candidate_pairs"))
        c3.metric("Reduction ratio", f"{float(blocking.iloc[0]['reduction_ratio']) * 100:.2f}%")
        c4.metric("Missed true links", _metric_value(blocking, "true_links_missed"))
    _show_image(CONFIG.paths.resolution_flow_figure, "Candidate-pair resolution flow.")


def _field_evidence_from_pair(pair: pd.Series) -> pd.DataFrame:
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
        score = "" if not score_col or score_col not in pair else pair.get(score_col, "")
        if str(a_value).strip() == "" and str(b_value).strip() == "" and score == "":
            continue
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


def _pair_viewer(pair: pd.Series) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("#### Record A")
        st.write(f"**Name:** {pair.get('given_name_a', '')} {pair.get('surname_a', '')}")
        st.write(f"**Date of birth:** {pair.get('date_of_birth_a', '')}")
        st.write(f"**Address:** {pair.get('address_a', '')}")
        st.write(f"**Location:** {pair.get('suburb_a', '')}, {pair.get('state_a', '')} {pair.get('postcode_a', '')}")
    with right:
        st.markdown("#### Record B")
        st.write(f"**Name:** {pair.get('given_name_b', '')} {pair.get('surname_b', '')}")
        st.write(f"**Date of birth:** {pair.get('date_of_birth_b', '')}")
        st.write(f"**Address:** {pair.get('address_b', '')}")
        st.write(f"**Location:** {pair.get('suburb_b', '')}, {pair.get('state_b', '')} {pair.get('postcode_b', '')}")
    st.markdown("#### Field Evidence")
    st.dataframe(_field_evidence_from_pair(pair), use_container_width=True, hide_index=True)


def _field_evidence_page() -> None:
    st.header("Field Evidence")
    st.write(
        "The ML models do not compare raw records blindly. They learn from structured comparison features such as name similarity, "
        "date-of-birth agreement, address similarity, postcode agreement, and other available identity signals."
    )
    st.markdown("### Feature List")
    feature_rows = [
        ("given_name_sim", "String similarity between given names"),
        ("surname_sim", "String similarity between surnames"),
        ("date_of_birth_exact", "Exact date-of-birth agreement"),
        ("address_sim", "String similarity between address fields"),
        ("suburb_sim", "String similarity between suburb or place"),
        ("state_exact", "Exact state agreement"),
        ("postcode_exact", "Exact postcode agreement"),
        ("sex_exact", "Exact sex/gender agreement if available"),
    ]
    st.dataframe(pd.DataFrame(feature_rows, columns=["Feature", "Meaning"]), use_container_width=True, hide_index=True)

    queue = _load(CONFIG.paths.review_queue, dtype=str)
    classified = _load(CONFIG.paths.classified_pairs, dtype=str)
    sample_source = queue if not queue.empty else classified
    if sample_source.empty:
        _show_missing(CONFIG.paths.review_queue, "python scripts/run_pipeline.py --review-mode simulate")
        return
    st.markdown("### Sample Candidate Pair")
    index = st.slider("Example pair", 0, max(len(sample_source) - 1, 0), 0)
    pair = sample_source.iloc[index]
    st.metric("Model score / probability", f"{float(pair.get('model_score', 0.0)):.3f}")
    _pair_viewer(pair)


def _active_learning_workflow() -> None:
    st.header("Active Learning Workflow")
    st.write(
        "Active learning focuses reviewer effort on uncertain pairs near the classifier decision boundary. "
        "After each review batch, the new labels are added to the training set and the classifier is retrained."
    )
    st.info(
        "Round 0 is the seed-only model. From Round 1 onward, the app reports labels added before retraining and evaluation. "
        "The frozen test set is only used for evaluation."
    )
    _workflow_cards(
        [
            "Initial labelled seed",
            "Train classifier",
            "Score unlabelled pairs",
            "Select uncertain pairs",
            "Simulate or collect labels",
            "Retrain model",
            "Evaluate frozen test set",
        ]
    )
    config_rows = [
        ["Seed positive labels", CONFIG.active_learning.seed_positive_labels],
        ["Seed negative labels", CONFIG.active_learning.seed_negative_labels],
        ["Batch size", CONFIG.active_learning.batch_size],
        ["Rounds", CONFIG.active_learning.rounds],
        ["Random state", CONFIG.active_learning.random_state],
        ["Frozen test size", CONFIG.active_learning.test_size],
    ]
    st.markdown("### Experiment Configuration")
    st.dataframe(pd.DataFrame(config_rows, columns=["Setting", "Value"]), use_container_width=True, hide_index=True)
    st.info(
        "Formal active-learning experiments use FEBRL ground truth to simulate reviewer labels for reproducibility. "
        "Live review decisions are stored separately and should not contaminate the frozen test set."
    )
    rounds = _load(CONFIG.paths.active_learning_rounds)
    if rounds.empty:
        _show_missing(CONFIG.paths.active_learning_rounds, "python scripts/run_active_learning.py")
    else:
        st.markdown("### Active-Learning Rounds")
        st.dataframe(rounds, use_container_width=True)
        best = rounds.sort_values("F1-score", ascending=False).iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best round", int(best["Round"]))
        c2.metric("Labels used", f"{int(best['Labelled pairs']):,}")
        c3.metric("Best F1", f"{float(best['F1-score']):.3f}")
        c4.metric("Best recall", f"{float(best['Recall']):.3f}")
    _show_image(CONFIG.paths.active_learning_curve_figure, "Active-learning F1 over review rounds.", "python scripts/run_active_learning.py")


def _human_review_queue() -> None:
    st.header("Human Review Queue")
    st.caption("Review one uncertain candidate pair at a time. Confirm and reject decisions are final. Skip keeps the pair unresolved.")
    st.info(
        "Live review decisions demonstrate workflow and audit logging. They can become future training labels, but formal benchmark active-learning results use simulated labels from FEBRL ground truth."
    )
    queue = _load(CONFIG.paths.review_queue, dtype=str)
    decisions = load_review_decisions(CONFIG.paths.review_decisions)
    if queue.empty:
        _show_missing(CONFIG.paths.review_queue, "python scripts/run_pipeline.py --review-mode simulate")
        return
    if "is_true_link" in queue.columns:
        st.error("Safety issue: review_queue.csv exposes ground truth. Regenerate outputs before demo.")
        return

    pending = pending_review_queue(queue, decisions)
    resolved = decisions[decisions["reviewer_decision"].isin(["Confirm Match", "Reject Match"])]
    skipped = decisions[decisions["reviewer_decision"] == "Skip"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grey-zone pairs", f"{len(queue):,}")
    c2.metric("Resolved live", f"{len(resolved):,}")
    c3.metric("Skipped", f"{len(skipped):,}")
    c4.metric("Pending", f"{len(pending):,}")
    if pending.empty:
        st.success("No pending review pairs remain.")
        return

    pair = pending.iloc[0]
    st.markdown("### Current Pair")
    m1, m2, m3 = st.columns(3)
    m1.metric("Model probability / score", f"{float(pair.get('model_score', 0.0)):.3f}")
    m2.metric("Hybrid EMPI score", f"{float(pair.get('hybrid_empi_score', 0.0)):.3f}")
    m3.metric("Current decision", pair.get("model_decision", "Needs Human Review"))
    st.warning(pair.get("decision_reason", "Sent to review because the pair is in the uncertainty band."))
    _pair_viewer(pair)

    notes = st.text_area("Reviewer notes")
    a, b, c = st.columns(3)
    if a.button("Confirm Match", type="primary", use_container_width=True):
        updated = upsert_review_decision(
            decisions,
            pair,
            "Confirm Match",
            notes,
            CONFIG.matcher.lower_threshold,
            CONFIG.matcher.upper_threshold,
        )
        save_review_decisions(updated, CONFIG.paths.review_decisions, CONFIG.paths.review_decisions_export)
        st.rerun()
    if b.button("Reject Match", use_container_width=True):
        updated = upsert_review_decision(
            decisions,
            pair,
            "Reject Match",
            notes,
            CONFIG.matcher.lower_threshold,
            CONFIG.matcher.upper_threshold,
        )
        save_review_decisions(updated, CONFIG.paths.review_decisions, CONFIG.paths.review_decisions_export)
        st.rerun()
    if c.button("Skip", use_container_width=True):
        updated = upsert_review_decision(
            decisions,
            pair,
            "Skip",
            notes,
            CONFIG.matcher.lower_threshold,
            CONFIG.matcher.upper_threshold,
        )
        save_review_decisions(updated, CONFIG.paths.review_decisions, CONFIG.paths.review_decisions_export)
        st.rerun()


def _model_performance() -> None:
    st.header("Model Performance")
    st.write(
        "This page compares the transparent Hybrid EMPI Score baseline with supervised classifiers trained on field-level evidence features. "
        "Logistic Regression is the explainable ML baseline. Random Forest and Gradient Boosting test whether nonlinear tabular models improve linkage performance."
    )
    comparison = _load(CONFIG.paths.model_comparison)
    if comparison.empty:
        _show_missing(CONFIG.paths.model_comparison, "python scripts/run_active_learning.py")
    else:
        st.dataframe(comparison, use_container_width=True)
        best = comparison.sort_values("F1-score", ascending=False).iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Best method", best["Method"])
        c2.metric("Best F1", f"{float(best['F1-score']):.3f}")
        c3.metric("Best recall", f"{float(best['Recall']):.3f}")
    _show_image(CONFIG.paths.model_comparison_f1_figure, "Model comparison by F1-score.", "python scripts/run_active_learning.py")


def _learning_curves() -> None:
    st.header("Learning Curves")
    st.write(
        "Learning curves show whether active learning reaches strong performance with fewer labelled examples than random sampling."
    )
    active_rounds = _load(CONFIG.paths.active_learning_rounds)
    random_vs_active = _load(CONFIG.paths.random_vs_active_learning)
    if active_rounds.empty:
        _show_missing(CONFIG.paths.active_learning_rounds, "python scripts/run_active_learning.py")
    else:
        st.markdown("### Active-Learning Rounds")
        st.dataframe(active_rounds, use_container_width=True)
    if not random_vs_active.empty:
        st.markdown("### Random Sampling vs Active Learning")
        st.dataframe(random_vs_active, use_container_width=True)
    _show_image(CONFIG.paths.active_learning_curve_figure, "Active-learning curve.", "python scripts/run_active_learning.py")
    _show_image(CONFIG.paths.random_vs_active_learning_figure, "Random sampling vs active learning.", "python scripts/run_active_learning.py")
    _show_image(CONFIG.paths.label_efficiency_curve_figure, "Label efficiency curve.", "python scripts/run_active_learning.py")


def _final_evaluation() -> None:
    st.header("Final Evaluation")
    st.write(
        "The main three-condition evaluation remains the anchor comparison. Active-learning outputs are shown separately as an extension, not as a replacement for the original AI + HITL grey-zone review condition."
    )
    comparison = _load(CONFIG.paths.final_evaluation_comparison)
    if comparison.empty:
        _show_missing(CONFIG.paths.final_evaluation_comparison, "python scripts/run_pipeline.py --review-mode simulate")
    else:
        st.markdown("### Three-Condition Evaluation")
        st.dataframe(comparison, use_container_width=True)
    active = _load(CONFIG.paths.active_learning_rounds)
    if not active.empty:
        st.markdown("### Active-Learning Extension")
        best = active.sort_values("F1-score", ascending=False).iloc[0]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Method": "Active Learning ML Extension",
                        "Precision": best["Precision"],
                        "Recall": best["Recall"],
                        "F1-score": best["F1-score"],
                        "False positives": best["False positives"],
                        "False negatives": best["False negatives"],
                        "Candidate pairs reviewed": best["Labelled pairs"],
                        "Review workload percentage": best["Labelled pairs"] / max(len(_load(CONFIG.paths.classified_pairs)), 1),
                        "Key interpretation": "Classifier retrained using simulated review labels selected by uncertainty sampling.",
                    }
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    left, right = st.columns(2)
    with left:
        _show_image(CONFIG.paths.benchmark_figure, "Benchmark comparison.")
    with right:
        _show_image(CONFIG.paths.workload_figure, "Workload comparison.")
    _show_image(CONFIG.paths.workload_percentage_figure, "Review workload percentage.")


def _threshold_analysis() -> None:
    st.header("Threshold Analysis")
    st.write("Thresholds control which pairs are resolved automatically and which pairs enter grey-zone review.")
    sweep = _load(CONFIG.paths.threshold_sweep)
    if sweep.empty:
        _show_missing(CONFIG.paths.threshold_sweep, "python scripts/run_threshold_sweep.py")
    else:
        st.dataframe(sweep, use_container_width=True)
    _show_image(CONFIG.paths.threshold_f1_figure, "Threshold vs F1-score.", "python scripts/run_threshold_sweep.py")
    _show_image(CONFIG.paths.threshold_workload_figure, "Threshold vs review workload.", "python scripts/run_threshold_sweep.py")
    _show_image(CONFIG.paths.recall_workload_figure, "Recall vs review workload.", "python scripts/run_threshold_sweep.py")


def _report_evidence() -> None:
    st.header("Report Evidence")
    st.write("Use these outputs as evidence for the final report and presentation. Do not copy live review ground truth into the demo.")
    evidence = [
        CONFIG.paths.final_evaluation_comparison,
        CONFIG.paths.model_comparison,
        CONFIG.paths.active_learning_rounds,
        CONFIG.paths.random_vs_active_learning,
        CONFIG.paths.benchmark_figure,
        CONFIG.paths.workload_figure,
        CONFIG.paths.model_comparison_f1_figure,
        CONFIG.paths.active_learning_curve_figure,
        CONFIG.paths.random_vs_active_learning_figure,
        CONFIG.paths.evaluation_summary,
        CONFIG.paths.scoring_method_summary,
        CONFIG.paths.active_learning_summary,
    ]
    rows = [{"Evidence file": str(path), "Available": "Yes" if path.exists() else "No"} for path in evidence]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    for path in [
        CONFIG.paths.evaluation_summary,
        CONFIG.paths.scoring_method_summary,
        CONFIG.paths.active_learning_summary,
        CONFIG.paths.limitations,
    ]:
        if path.exists():
            with st.expander(path.name):
                st.markdown(_read_markdown(path))


def _sidebar() -> str:
    with st.sidebar:
        st.markdown("### Project")
        st.write("AI-Assisted HITL Record Linkage")
        st.markdown("### Dataset")
        st.write("FEBRL4 benchmark data")
        st.markdown("### Workflow mode")
        st.write("Active Learning EMPI-inspired matching")
        scoring_method = st.selectbox("Scoring method", MODEL_OPTIONS)
        review_mode = st.selectbox("Review mode", ["simulate", "merge", "ignore"])
        lower = st.slider("Lower threshold", 0.0, 1.0, CONFIG.matcher.lower_threshold, 0.05)
        upper = st.slider("Upper threshold", 0.0, 1.0, CONFIG.matcher.upper_threshold, 0.05)
        batch_size = st.number_input("Active-learning batch size", min_value=1, value=CONFIG.active_learning.batch_size)
        rounds = st.number_input("Active-learning rounds", min_value=1, value=CONFIG.active_learning.rounds)
        random_state = st.number_input("Random state", min_value=0, value=CONFIG.active_learning.random_state)
        CONFIG.active_learning.batch_size = int(batch_size)
        CONFIG.active_learning.rounds = int(rounds)
        CONFIG.active_learning.random_state = int(random_state)
        st.caption(f"Selected scoring method for demo context: {scoring_method}")
        st.caption(f"Batch settings shown here: batch={batch_size}, rounds={rounds}, random_state={random_state}.")
        st.info(
            "Formal benchmark HITL labels are simulated using FEBRL ground truth. Live review clicks demonstrate workflow and audit logging. Frozen test data must not be used for active-learning training."
        )
        if st.button("Run FEBRL pipeline", type="primary", use_container_width=True):
            with st.spinner("Running FEBRL EMPI pipeline..."):
                run_experiment(review_mode, lower, upper)
            st.cache_data.clear()
            st.success("Pipeline complete.")
        if st.button("Run active-learning experiment", use_container_width=True):
            with st.spinner("Running active-learning simulation..."):
                run_active_learning_experiment()
            st.cache_data.clear()
            st.success("Active-learning outputs generated.")
        if st.button("Reset live review decisions", use_container_width=True):
            save_review_decisions(pd.DataFrame(), CONFIG.paths.review_decisions, CONFIG.paths.review_decisions_export)
            st.cache_data.clear()
            st.rerun()

        return st.radio(
            "Pages",
            [
                "Overview",
                "Dataset & Blocking",
                "Field Evidence",
                "Active Learning Workflow",
                "Human Review Queue",
                "Model Performance",
                "Learning Curves",
                "Final Evaluation",
                "Threshold Analysis",
                "Report Evidence",
            ],
        )


def main() -> None:
    st.set_page_config(page_title="Active Learning HITL Record Linkage", layout="wide")
    st.title("Active Learning HITL Record Linkage")
    st.caption("FEBRL4 benchmark data, EMPI-inspired evidence, uncertainty sampling, human review, and batch retraining.")
    page = _sidebar()
    pages = {
        "Overview": _overview,
        "Dataset & Blocking": _dataset_and_blocking,
        "Field Evidence": _field_evidence_page,
        "Active Learning Workflow": _active_learning_workflow,
        "Human Review Queue": _human_review_queue,
        "Model Performance": _model_performance,
        "Learning Curves": _learning_curves,
        "Final Evaluation": _final_evaluation,
        "Threshold Analysis": _threshold_analysis,
        "Report Evidence": _report_evidence,
    }
    pages[page]()


if __name__ == "__main__":
    main()
