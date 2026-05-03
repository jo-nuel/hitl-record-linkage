# AI-Assisted HITL Record Linkage

University research prototype for:

**AI-Assisted Human-in-the-Loop Record Linkage for Detecting Duplicate Patient Records in Healthcare Datasets**

The prototype demonstrates an EMPI-inspired linkage workflow using FEBRL benchmark data. The system automatically resolves clear record pairs and escalates grey-zone pairs for human review.

## Research Question

To what extent can an AI-assisted human-in-the-loop record linkage system improve duplicate patient record detection compared with clerical review and AI-only matching?

## Dataset Change: Synthea To FEBRL

The project originally used a locally constructed evaluation dataset. Peer feedback showed that FEBRL is a stronger benchmark because it provides known true links for record linkage evaluation.

The active implementation now uses FEBRL4 from the Python Record Linkage Toolkit. FEBRL4 supports two-file linkage with dataset A, dataset B, and ground-truth links.

FEBRL is fictitious benchmark data. It is not real hospital production data.

No raw dataset download is required. FEBRL4 is loaded through the `recordlinkage` Python package.

## Research Claim

An EMPI-inspired AI + HITL workflow can improve the accuracy-efficiency trade-off in duplicate patient record linkage by automatically resolving clear cases and escalating uncertain cases for human review.

## EMPI-Inspired Workflow

1. Load FEBRL4.
2. Preprocess identity fields.
3. Generate candidate pairs with multi-pass blocking.
4. Compare field-level evidence.
5. Score pairs with ECM probability and a Hybrid EMPI-style evidence score.
6. Classify pairs as Auto Match, Auto Non-match, or Needs Human Review.
7. Save reviewer decisions in an audit log.
8. Evaluate the three final methods.

## Three Evaluation Methods

1. Human-only Clerical Review Baseline
   Every blocked candidate pair is reviewed using FEBRL true links. This is a blocked-set clerical benchmark, not review of every possible real-world pair.

2. AI-only EMPI Matcher
   The matcher uses automatic decisions only. Grey-zone pairs are not corrected by a human and are excluded from positive predictions, so true links in the grey zone count as missed links.

3. AI + HITL Grey-Zone Review
   The matcher resolves clear cases automatically. Grey-zone pairs are resolved using simulated ideal human review based on FEBRL true links.

## Setup

Recommended Python version: Python 3.11.

```bash
python --version
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Clean Benchmark Run

Use `simulate` for reproducible benchmark outputs. This mode uses FEBRL ground truth to simulate an ideal human reviewer for grey-zone pairs.

```bash
python scripts/run_pipeline.py --review-mode simulate
```

## Run Threshold Sweep

```bash
python scripts/run_threshold_sweep.py
```

## Validate Outputs

```bash
python scripts/validate_outputs.py
```

Additional reproducibility notes are in `docs/reproducibility_checklist.md`.
Repository structure notes are in `docs/repository_structure.md`.

## Run Dashboard

```bash
streamlit run app.py
```

Dashboard review modes:

- `simulate` uses FEBRL ground truth to simulate an ideal human reviewer resolving grey-zone pairs. This is best for reproducible benchmark evaluation.
- `merge` uses saved live reviewer decisions from `data/review_decisions.csv` where available. This is best for demo continuity, not the default benchmark.
- `ignore` does not apply human correction to grey-zone pairs. This is best for AI-only style inspection.

Formal benchmark metrics are generated from the evaluation pipeline. The AI + HITL result uses simulated grey-zone review based on FEBRL ground truth to represent an idealised human reviewer. Live reviewer decisions in Streamlit are stored for demonstration and audit logging, but they do not automatically overwrite formal benchmark metrics unless the pipeline is explicitly rerun in merge mode.

`review_queue.csv`, `classified_pairs.csv`, and `final_decisions.csv` are exported without ground-truth labels so reviewer-facing and presentation-facing files do not reveal the answer. Simulated benchmark review decisions are written separately to `outputs/tables/simulated_review_decisions.csv`.

## Scoring Method

The matcher blends two signals:

- `recordlinkage.ECMClassifier` probability where available.
- A Hybrid EMPI-style evidence score based on field-level agreement.

The hybrid score gives stronger weight to fields such as date of birth, surname, postcode, and address because they provide stronger identity evidence than broader or often-missing fields. Pairs above the upper threshold become Auto Match. Pairs below the lower threshold become Auto Non-match. Pairs between the thresholds enter grey-zone human review.

## Main Outputs

- `outputs/tables/final_evaluation_comparison.csv`
- `outputs/tables/evaluation_metrics.csv`
- `outputs/tables/threshold_sweep.csv`
- `outputs/tables/review_queue.csv`
- `outputs/tables/final_decisions.csv`
- `outputs/tables/simulated_review_decisions.csv`
- `outputs/reports/dataset_profile.md`
- `outputs/reports/blocking_summary.md`
- `outputs/reports/scoring_method_summary.md`
- `outputs/reports/evaluation_summary.md`
- `outputs/reports/threshold_sweep_summary.md`
- `outputs/reports/methodology_summary.md`
- `outputs/reports/limitations.md`
- `outputs/figures/benchmark_comparison.png`
- `outputs/figures/workload_comparison.png`
- `outputs/figures/workload_percentage.png`
- `outputs/figures/decision_distribution.png`
- `outputs/figures/score_distribution.png`
- `outputs/figures/threshold_vs_f1.png`
- `outputs/figures/threshold_vs_review_workload.png`
- `outputs/figures/recall_vs_review_workload.png`

## Limitations

- FEBRL is benchmark data, not real hospital production data.
- AI + HITL uses simulated ideal review when calculating final evaluation metrics.
- Blocking can miss true links before the matcher or reviewer sees them.
- Thresholds are selected from a small sweep and need further validation before any deployment claim.
