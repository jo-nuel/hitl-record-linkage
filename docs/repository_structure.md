# Repository Structure

## Entrypoints

- `scripts/run_pipeline.py`: prepares FEBRL4 candidate-pair outputs used by the dashboard and experiments.
- `scripts/run_active_learning.py`: runs the final active-learning experiment and writes evaluation evidence.
- `scripts/validate_outputs.py`: checks generated outputs for basic correctness and label-leakage safety.
- `app.py`: Streamlit entrypoint for the dashboard.

## Source Code

- `src/data/febrl_loader.py`: loads FEBRL4 and writes the dataset profile.
- `src/empi/preprocessing.py`: standardises identity fields and helper blocking fields. The folder name is retained for compatibility.
- `src/empi/blocking.py`: creates candidate record pairs with multi-pass blocking.
- `src/empi/comparison.py`: creates field-level comparison features.
- `src/empi/hitl.py`: manages the review queue and reviewer audit log.
- `src/evaluation/active_learning.py`: trains ML matchers, selects uncertain pairs, simulates reviewer labels, and retrains in batches.
- `src/evaluation/report_outputs.py`: writes tables, figures, and concise technical summaries.

## Outputs

- `outputs/tables/`: CSV evidence tables.
- `outputs/figures/`: report-ready charts.
- `outputs/reports/`: concise technical summaries for the final report.
- `data/review_decisions.csv`: local live review audit log, ignored by Git.
- `outputs/tables/simulated_review_decisions.csv`: benchmark-only simulated review decisions.
