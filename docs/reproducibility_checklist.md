# Reproducibility Checklist

Use Python 3.11 for the final prototype run.

## Clean Benchmark Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_pipeline.py --review-mode simulate
python scripts/run_active_learning.py
python scripts/validate_outputs.py
```

## Expected Evidence Outputs

- `outputs/tables/blocking_stats.csv`
- `outputs/tables/hyperparameter_tuning.csv`
- `outputs/tables/model_comparison.csv`
- `outputs/tables/active_learning_rounds.csv`
- `outputs/tables/final_research_evaluation.csv`
- `outputs/tables/review_queue.csv`
- `outputs/tables/simulated_review_decisions.csv`
- `outputs/reports/dataset_profile.md`
- `outputs/reports/blocking_summary.md`
- `outputs/reports/active_learning_summary.md`
- `outputs/reports/hyperparameter_tuning_summary.md`
- `outputs/reports/limitations.md`
- `outputs/figures/active_learning_curve.png`
- `outputs/figures/active_learning_error_reduction.png`
- `outputs/figures/final_accuracy_comparison.png`
- `outputs/figures/final_workload_comparison.png`

## Notes

- FEBRL4 is loaded through the `recordlinkage` package, so no raw dataset download is required.
- Active-learning labels are simulated from FEBRL4 ground truth for reproducible benchmark evidence.
- Live Streamlit review decisions are stored for demonstration and audit logging.
- Ground-truth labels must not appear in `outputs/tables/review_queue.csv`.
- Ground-truth labels must not appear in `outputs/tables/classified_pairs.csv` or `outputs/tables/final_decisions.csv`.
