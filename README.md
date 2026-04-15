# HITL Record Linkage

This repository contains the Week 8 research prototype for:

**AI-Assisted Human-in-the-Loop Record Linkage for Detecting Duplicate Patient Records in Healthcare Datasets**

The prototype uses the Synthea `patients.csv` dataset, creates synthetic duplicates with known ground truth, applies weighted similarity matching, and sends only uncertain cases to a human reviewer.

## HITL Operational Loop

This project uses an **operational HITL review loop**, not an active-learning retraining loop.

The loop is:

1. AI scores blocked record pairs.
2. Threshold logic assigns `Match`, `Non-match`, or `Review Needed`.
3. Only `Review Needed` pairs are written to the review queue.
4. A reviewer inspects one uncertain pair at a time.
5. The reviewer chooses `Confirm Match`, `Reject Match`, or `Skip`.
6. The decision is saved to CSV.
7. Final outputs merge automated decisions with reviewed decisions.

The model is **not retrained** from review feedback in this Week 8 prototype. The goal is to demonstrate explainable human oversight for ambiguous cases.

## Baselines

The evaluation compares exactly three approaches:

- `manual_only`: a simulated clerical-review benchmark where every blocked candidate pair is assumed to be inspected against ground truth
- `ai_only`: threshold-based automated matching with no human review of uncertain pairs
- `ai_human_hitl`: automated matching plus human review of only `Review Needed` pairs

The `manual_only` identifier is kept in the evaluation outputs for consistency, but it does **not** mean full end-to-end manual matching across all possible pairs. It represents full clerical review of the **blocked candidate set** only, and is included as a simulated benchmark for review effort rather than a literal interactive experiment.

## Setup

```bash
pip install -r requirements.txt
```

## Run The Benchmark Version

This run uses simulated review so the repository produces a complete comparison table immediately.

```bash
python scripts/run_pipeline.py --regenerate-data --review-mode simulate --sample-size 5000
```

Main outputs:

- [data/results/classified_pairs.csv](data/results/classified_pairs.csv)
- [data/results/review_queue.csv](data/results/review_queue.csv)
- [data/results/review_decisions.csv](data/results/review_decisions.csv)
- [data/results/final_decisions.csv](data/results/final_decisions.csv)
- [data/results/evaluation_metrics.csv](data/results/evaluation_metrics.csv)
- [data/results/experiment_summary.md](data/results/experiment_summary.md)
- [data/results/benchmark_comparison_table.csv](data/results/benchmark_comparison_table.csv)
- [data/results/workload_summary_table.csv](data/results/workload_summary_table.csv)
- `data/results/figures/` for presentation-ready charts

## Run The Manual HITL Demo

1. Generate or refresh the review queue:

```bash
python scripts/run_pipeline.py --regenerate-data --review-mode merge --sample-size 5000
```

2. Open the Streamlit reviewer:

```bash
streamlit run app/streamlit_app.py
```

The Streamlit app includes:

- an overview page with current run status
- benchmark and workload tables
- presentation-ready charts generated from real outputs
- the live review queue for uncertain pairs

3. After saving manual decisions, re-run the pipeline to merge them into the final outputs:

```bash
python scripts/run_pipeline.py --review-mode merge --sample-size 5000
```

Manual review decisions are stored in [data/reviewed/review_decisions.csv](data/reviewed/review_decisions.csv) and copied into [data/results/review_decisions.csv](data/results/review_decisions.csv) for each run.

## Current Output Story

- `data/processed/` stores the generated experiment dataset and synthetic duplicate artifacts
- `data/results/classified_pairs.csv` stores the threshold-based AI decisions before human review
- `data/results/review_queue.csv` stores the uncertain pairs shown to the reviewer
- `data/results/final_decisions.csv` stores the merged automated and reviewed outputs
- `data/results/evaluation_metrics.csv` stores one clear comparison table for `manual_only`, `ai_only`, and `ai_human_hitl`
- `data/results/experiment_summary.md` stores lightweight initial experimental analysis for the progress report
