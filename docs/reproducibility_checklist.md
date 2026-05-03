# Reproducibility Checklist

Use Python 3.11 for the final prototype run.

## Clean Benchmark Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_pipeline.py --review-mode simulate
python scripts/run_threshold_sweep.py
python scripts/validate_outputs.py
```

## Expected Evidence Outputs

- `outputs/tables/final_evaluation_comparison.csv`
- `outputs/tables/threshold_sweep.csv`
- `outputs/tables/review_queue.csv`
- `outputs/tables/simulated_review_decisions.csv`
- `outputs/reports/dataset_profile.md`
- `outputs/reports/blocking_summary.md`
- `outputs/reports/scoring_method_summary.md`
- `outputs/reports/evaluation_summary.md`
- `outputs/figures/benchmark_comparison.png`
- `outputs/figures/workload_comparison.png`
- `outputs/figures/workload_percentage.png`

## Notes

- FEBRL4 is loaded through the `recordlinkage` package, so no raw dataset download is required.
- `simulate` is the recommended mode for reproducible benchmark evidence.
- Live Streamlit review decisions are stored for demonstration and audit logging.
- Ground-truth labels must not appear in `outputs/tables/review_queue.csv`.
- Ground-truth labels must not appear in `outputs/tables/classified_pairs.csv` or `outputs/tables/final_decisions.csv`.
