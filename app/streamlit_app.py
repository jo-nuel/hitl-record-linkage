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
    # Load a CSV output if it exists; otherwise return an empty table.
    if not path.exists():
        return pd.DataFrame()
    if dtype:
        return pd.read_csv(path, dtype=dtype).fillna("")
    return pd.read_csv(path).fillna("")


@st.cache_data(show_spinner=False)
def _cached_csv(path_text: str, modified_time: float, dtype: str | None = None) -> pd.DataFrame:
    return _read_csv(Path(path_text), dtype)


def _load(path: Path, dtype: str | None = None) -> pd.DataFrame:
    # Include modified time in the cache key so refreshed outputs appear in the demo.
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


def _final_method_row(method: str) -> pd.Series | None:
    final_eval = _load(CONFIG.paths.final_research_evaluation)
    if final_eval.empty or "Method" not in final_eval.columns:
        return None
    rows = final_eval[final_eval["Method"] == method]
    return None if rows.empty else rows.iloc[0]


def _section_note(text: str) -> None:
    st.info(text)


def _workflow_cards(items: list[str]) -> None:
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        col.info(item)


def _overview() -> None:
    st.header("Pitch Overview")
    st.subheader("AI-assisted duplicate record detection with human review")
    st.write(
        "This prototype uses machine learning and active learning to detect duplicate patient records, while sending "
        "uncertain cases to human review."
    )
    _section_note(
        "This is a reproducible research prototype using FEBRL4 benchmark data. It is not a production hospital system."
    )
    blocking = _load(CONFIG.paths.blocking_stats)
    hitl = _final_method_row("AI + HITL Active Learning Matcher")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Dataset", "FEBRL4")
    c2.metric("Candidate pairs", _metric_value(blocking, "candidate_pairs"))
    if hitl is not None:
        workload = float(hitl.get("Review workload percentage", 0.0))
        c3.metric("AI + HITL F1-score", f"{float(hitl.get('F1-score', 0.0)):.3f}")
        c4.metric("Reviewed pairs", f"{int(hitl.get('Candidate pairs reviewed', 0)):,}")
        c5.metric("Workload reduction", f"{100.0 - workload:.2f}%")
    else:
        c3.metric("AI + HITL F1-score", "n/a")
        c4.metric("Reviewed pairs", "n/a")
        c5.metric("Workload reduction", "n/a")
    st.markdown("### Demo Workflow")
    _workflow_cards(
        [
            "FEBRL4",
            "Blocking",
            "Field comparison",
            "ML scoring",
            "Human Review",
            "Batch retraining",
            "Evaluation",
        ]
    )


def _dataset_and_blocking() -> None:
    st.header("Data & Blocking")
    st.write(
        "Blocking reduces the number of record pairs before matching. This makes the problem small enough to evaluate, "
        "but some true links can still be missed at this stage."
    )
    profile = _read_markdown(CONFIG.paths.dataset_profile)
    if profile:
        with st.expander("FEBRL4 dataset profile"):
            st.markdown(profile)
    else:
        _show_missing(CONFIG.paths.dataset_profile, "python scripts/run_pipeline.py --review-mode simulate")

    blocking = _load(CONFIG.paths.blocking_stats)
    if blocking.empty:
        _show_missing(CONFIG.paths.blocking_stats, "python scripts/run_pipeline.py --review-mode simulate")
    else:
        st.markdown("### Key Blocking Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total possible pairs", _metric_value(blocking, "total_possible_pairs"))
        c2.metric("Candidate pairs after blocking", _metric_value(blocking, "candidate_pairs"))
        c3.metric("Blocking recall", f"{float(blocking.iloc[0]['blocking_recall']) * 100:.2f}%")
        c4.metric("Missed true links", _metric_value(blocking, "true_links_missed"))
        with st.expander("Full blocking table"):
            st.dataframe(blocking, use_container_width=True)
        with st.expander("Candidate-pair resolution flow"):
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
    st.metric("ML match probability", f"{float(pair.get('model_score', 0.0)):.3f}")
    _pair_viewer(pair)


def _active_learning_workflow() -> None:
    st.header("Active Learning Details")
    st.write(
        "Active learning focuses reviewer effort on uncertain pairs near the classifier decision boundary. "
        "After each review batch, the new labels are added to the training set and the classifier is retrained."
    )
    st.info(
        "Round 0 is the seed-only model. From Round 1 onward, the app reports labels added before retraining and evaluation. "
        "Classifier hyperparameters are tuned on seed labels only. The frozen test set is only used for evaluation."
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
    st.header("Human Review Demo")
    st.caption("Review one uncertain candidate pair at a time. Skip / Defer leaves the pair unresolved.")
    st.info(
        "Live review decisions are stored for demo and audit use only. Formal benchmark metrics use simulated reviewer labels from FEBRL4."
    )
    queue = _load(CONFIG.paths.review_queue, dtype=str)
    decisions = load_review_decisions(CONFIG.paths.review_decisions)
    if queue.empty:
        _show_missing(CONFIG.paths.review_queue, "python scripts/run_pipeline.py --review-mode simulate")
        return
    # Ground truth must never be shown in the review queue.
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
    m1.metric("ML match probability", f"{float(pair.get('model_score', 0.0)):.3f}")
    m2.metric("Evidence score", f"{float(pair.get('hybrid_empi_score', 0.0)):.3f}")
    m3.metric("Current decision", pair.get("model_decision", "Needs Human Review"))
    st.caption("Evidence score is a rule-based support score from field agreement.")
    st.warning("This pair is in the uncertainty band, so it is sent for human review.")
    _pair_viewer(pair)

    st.write("Review the field evidence, then choose whether the records refer to the same person.")
    st.caption("Confirm and Reject save a decision and move to the next pair. Skip / Defer keeps the pair open for later review.")
    a, b, c = st.columns(3)
    if a.button("Confirm Match", type="primary", use_container_width=True):
        updated = upsert_review_decision(
            decisions,
            pair,
            "Confirm Match",
            st.session_state.get("reviewer_notes", ""),
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
            st.session_state.get("reviewer_notes", ""),
            CONFIG.matcher.lower_threshold,
            CONFIG.matcher.upper_threshold,
        )
        save_review_decisions(updated, CONFIG.paths.review_decisions, CONFIG.paths.review_decisions_export)
        st.rerun()
    if c.button("Skip / Defer", use_container_width=True):
        updated = upsert_review_decision(
            decisions,
            pair,
            "Skip",
            st.session_state.get("reviewer_notes", ""),
            CONFIG.matcher.lower_threshold,
            CONFIG.matcher.upper_threshold,
        )
        save_review_decisions(updated, CONFIG.paths.review_decisions, CONFIG.paths.review_decisions_export)
        st.rerun()
    st.text_area("Reviewer notes", key="reviewer_notes")


def _model_performance() -> None:
    st.header("ML Model Selection")
    st.write(
        "This page shows that the selected ML matcher was chosen through simple tuning, not guessed. "
        "Hybrid EMPI is kept as a fallback and explanation score, not as a final evaluation method."
    )
    tuning = _load(CONFIG.paths.hyperparameter_tuning)
    if tuning.empty:
        _show_missing(CONFIG.paths.hyperparameter_tuning, "python scripts/run_active_learning.py")
    else:
        st.markdown("### Hyperparameter Tuning")
        default_tuning_cols = ["Method", "Best CV F1-score", "Best parameters"]
        st.dataframe(tuning[default_tuning_cols], use_container_width=True, hide_index=True)
        best_tuned = tuning.sort_values("Best CV F1-score", ascending=False).iloc[0]
        st.info(
            f"Selected active-learning classifier: {best_tuned['Method']} "
            f"(best CV F1-score {float(best_tuned['Best CV F1-score']):.3f})."
        )
        with st.expander("Tuning runtime and notes"):
            st.dataframe(tuning, use_container_width=True, hide_index=True)
    comparison = _load(CONFIG.paths.model_comparison)
    if comparison.empty:
        _show_missing(CONFIG.paths.model_comparison, "python scripts/run_active_learning.py")
    else:
        st.markdown("### Frozen Test-Set ML Check")
        ml_comparison = comparison[comparison["Method"] != "Hybrid EMPI Score"].copy()
        metric_cols = ["Method", "Precision", "Recall", "F1-score", "False positives", "False negatives"]
        st.dataframe(ml_comparison[metric_cols], use_container_width=True, hide_index=True)
        ml_only = comparison[comparison["Method"] != "Hybrid EMPI Score"]
        best = ml_only.sort_values("F1-score", ascending=False).iloc[0] if not ml_only.empty else comparison.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Best test-set ML", best["Method"])
        c2.metric("Best F1", f"{float(best['F1-score']):.3f}")
        c3.metric("Best recall", f"{float(best['Recall']):.3f}")
        with st.expander("Internal fallback/evidence score"):
            st.write("Hybrid EMPI is kept as a fallback/evidence score, not as a final evaluation method.")
            st.dataframe(comparison, use_container_width=True, hide_index=True)
    _show_image(CONFIG.paths.model_comparison_f1_figure, "Model comparison by F1-score.", "python scripts/run_active_learning.py")


def _learning_curves() -> None:
    st.header("Learning Progress")
    st.write(
        "Round 0 uses seed labels only. Each later round adds simulated reviewer labels and retrains the model."
    )
    active_rounds = _load(CONFIG.paths.active_learning_rounds)
    if active_rounds.empty:
        _show_missing(CONFIG.paths.active_learning_rounds, "python scripts/run_active_learning.py")
    else:
        st.markdown("### Active-Learning Rounds")
        best = active_rounds.sort_values("F1-score", ascending=False).iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Best round", int(best["Round"]))
        c2.metric("Labels used", f"{int(best['Labelled pairs']):,}")
        c3.metric("F1-score", f"{float(best['F1-score']):.3f}")
        c4.metric("False positives", int(best["False positives"]))
        c5.metric("False negatives", int(best["False negatives"]))
        default_round_cols = [
            "Round",
            "Labelled pairs",
            "Precision",
            "Recall",
            "F1-score",
            "False positives",
            "False negatives",
        ]
        st.dataframe(active_rounds[default_round_cols], use_container_width=True, hide_index=True)
        with st.expander("Full active-learning rounds table"):
            st.dataframe(active_rounds, use_container_width=True, hide_index=True)
    _show_image(
        CONFIG.paths.active_learning_curve_figure,
        "Active-learning F1 curve. The y-axis is zoomed because the scores are close together.",
        "python scripts/run_active_learning.py",
    )
    with st.expander("Error details"):
        _show_image(
            CONFIG.paths.active_learning_error_reduction_figure,
            "False positives and false negatives across active-learning rounds.",
            "python scripts/run_active_learning.py",
        )


def _final_evaluation() -> None:
    st.header("Final Results")
    st.subheader("AI + HITL achieved near-clerical-review accuracy with far lower review workload.")
    st.write(
        "The final comparison focuses on human-only review, AI-only matching, and the proposed AI + HITL active-learning matcher."
    )
    st.info(
        "Scope note: the human-only baseline estimates the workload of reviewing all blocked candidate pairs. "
        "The ML methods are evaluated on a frozen test set to avoid training/test leakage. These are benchmark results from FEBRL4, not clinical deployment results."
    )
    final_research = _load(CONFIG.paths.final_research_evaluation)
    if final_research.empty:
        _show_missing(CONFIG.paths.final_research_evaluation, "python scripts/run_active_learning.py")
    else:
        st.markdown("### Final Research Evaluation")
        table = final_research.copy()
        table = table.rename(
            columns={
                "Candidate pairs reviewed": "Reviewed pairs",
                "Key interpretation": "Interpretation",
            }
        )
        default_eval_cols = [
            "Method",
            "Precision",
            "Recall",
            "F1-score",
            "Reviewed pairs",
            "Interpretation",
        ]
        st.dataframe(table[default_eval_cols], use_container_width=True, hide_index=True)
        with st.expander("More details"):
            detail_cols = [
                "Method",
                "Role",
                "Dataset",
                "False positives",
                "False negatives",
                "Estimated review time",
                "Training labels used",
                "Evaluation scope",
                "Key interpretation",
            ]
            st.dataframe(final_research[detail_cols], use_container_width=True, hide_index=True)
    st.markdown("### Review Workload Comparison")
    _show_image(
        CONFIG.paths.final_workload_comparison_figure,
        "Human-only review checks all candidate pairs. HITL reviews only selected uncertain pairs.",
        "python scripts/run_active_learning.py",
    )
    with st.expander("Accuracy chart"):
        _show_image(
            CONFIG.paths.final_accuracy_comparison_figure,
            "Precision, recall, and F1-score for the three final methods.",
            "python scripts/run_active_learning.py",
        )


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
    st.header("Evidence Files")
    st.write("These are the main generated files to use as report evidence.")
    evidence = [
        CONFIG.paths.final_research_evaluation,
        CONFIG.paths.active_learning_rounds,
        CONFIG.paths.active_learning_curve_figure,
        CONFIG.paths.active_learning_error_reduction_figure,
        CONFIG.paths.final_accuracy_comparison_figure,
        CONFIG.paths.final_workload_comparison_figure,
        CONFIG.paths.dataset_profile,
        CONFIG.paths.blocking_summary,
        CONFIG.paths.active_learning_summary,
        CONFIG.paths.limitations,
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
        st.write("AI-Assisted Active Learning HITL")
        st.markdown("### Dataset")
        st.write("FEBRL4 benchmark data")
        st.caption("Presentation mode uses pre-generated outputs. Regenerate outputs only if needed.")

        # Demo controls are hidden so presentation users do not change experiment settings by accident.
        with st.expander("Advanced settings"):
            scoring_method = st.selectbox("Scoring method", MODEL_OPTIONS)
            review_mode = st.selectbox("Review mode", ["simulate", "merge", "ignore"])
            st.caption("simulate: uses FEBRL ground truth to simulate reviewer labels.")
            st.caption("merge: uses saved live review decisions where available.")
            st.caption("ignore: shows automated decisions without applying review decisions.")
            lower = st.slider("Lower threshold", 0.0, 1.0, CONFIG.matcher.lower_threshold, 0.05)
            upper = st.slider("Upper threshold", 0.0, 1.0, CONFIG.matcher.upper_threshold, 0.05)
            batch_size = st.number_input("Active-learning batch size", min_value=1, value=CONFIG.active_learning.batch_size)
            rounds = st.number_input("Active-learning rounds", min_value=1, value=CONFIG.active_learning.rounds)
            random_state = st.number_input("Random state", min_value=0, value=CONFIG.active_learning.random_state)
            CONFIG.active_learning.batch_size = int(batch_size)
            CONFIG.active_learning.rounds = int(rounds)
            CONFIG.active_learning.random_state = int(random_state)
            st.caption(f"Selected scoring method for demo context: {scoring_method}")
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
                # Live review decisions are stored for audit/demo use only.
                save_review_decisions(pd.DataFrame(), CONFIG.paths.review_decisions, CONFIG.paths.review_decisions_export)
                st.cache_data.clear()
                st.rerun()

        nav_mode = st.radio("Navigation", ["Presentation Mode", "Technical Appendix"])
        if nav_mode == "Presentation Mode":
            return st.radio(
                "Demo pages",
                [
                    "Pitch Overview",
                    "Data & Blocking",
                    "Human Review Demo",
                    "Learning Progress",
                    "Final Results",
                ],
            )
        return st.radio(
            "Appendix pages",
            [
                "Active Learning Details",
                "Field Evidence",
                "ML Model Selection",
                "Evidence Files",
                "Threshold Analysis",
            ],
        )


def main() -> None:
    st.set_page_config(page_title="Active Learning Record Linkage Demo", layout="wide")
    st.title("Active Learning Record Linkage Demo")
    st.caption("FEBRL4 benchmark data, EMPI-inspired evidence, uncertainty sampling, human review, and batch retraining.")
    page = _sidebar()
    pages = {
        "Pitch Overview": _overview,
        "Data & Blocking": _dataset_and_blocking,
        "Human Review Demo": _human_review_queue,
        "Learning Progress": _learning_curves,
        "Final Results": _final_evaluation,
        "Active Learning Details": _active_learning_workflow,
        "Field Evidence": _field_evidence_page,
        "ML Model Selection": _model_performance,
        "Evidence Files": _report_evidence,
        "Threshold Analysis": _threshold_analysis,
    }
    pages[page]()


if __name__ == "__main__":
    main()
