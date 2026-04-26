# AI-Assisted HITL Record Linkage

University research prototype for detecting duplicate or linked patient-style records with an EMPI-inspired workflow.

## Dataset Change: Synthea To FEBRL

Peer feedback identified a stronger benchmark option through the Python Record Linkage Toolkit. The project now uses FEBRL4 as the main dataset because it provides two-file record linkage data with known true links.

This change improves evaluation quality. The previous Synthea-based approach depended on internally created labels. The active implementation now evaluates against FEBRL benchmark links.

FEBRL is fictitious benchmark data. It is not real hospital production data.

## Method

The workflow is:

1. Load FEBRL4.
2. Preprocess identity fields.
3. Generate candidate pairs with multi-pass blocking.
4. Compare fields with Jaro-Winkler and exact agreement.
5. Score pairs with ECM probability and a Hybrid EMPI-style evidence score.
6. Split pairs into Auto Match, Auto Non-match, or Needs Human Review.
7. Store reviewer decisions in an audit log.
8. Evaluate AI-only, simulated AI + HITL, and manual benchmark over the blocked candidate set.

The HITL loop is operational. Reviewer decisions are stored and merged into final outputs. The matcher is not retrained during the run.

## Setup

```bash
pip install -r requirements.txt
```

## Run Pipeline

```bash
python scripts/run_pipeline.py
```

Run simulated HITL evaluation:

```bash
python scripts/run_pipeline.py --review-mode simulate
```

## Run Threshold Sweep

```bash
python scripts/run_threshold_sweep.py
```

## Run Dashboard

```bash
streamlit run app.py
```

## Main Outputs

- `data/processed/febrl_a.csv`
- `data/processed/febrl_b.csv`
- `data/processed/febrl_true_links.csv`
- `data/review_decisions.csv`
- `outputs/tables/evaluation_metrics.csv`
- `outputs/tables/review_queue.csv`
- `outputs/tables/final_decisions.csv`
- `outputs/tables/threshold_sweep.csv`
- `outputs/reports/dataset_profile.md`
- `outputs/reports/methodology_summary.md`
- `outputs/reports/evaluation_summary.md`
- `outputs/reports/limitations.md`
- `outputs/figures/`

## Evaluation Labels

- Manual Benchmark, Blocked Set: every blocked candidate pair is resolved against FEBRL true links.
- AI Only: automatic matches count as links. Review-needed pairs are left unresolved.
- AI + HITL Simulated: review-needed pairs are resolved using FEBRL true links as an ideal reviewer.
- AI + HITL Review File: saved reviewer decisions from `data/review_decisions.csv` are merged into final outputs.
