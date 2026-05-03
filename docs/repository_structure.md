# Repository Structure

## Entrypoints

- `scripts/run_pipeline.py`: runs the FEBRL EMPI linkage pipeline and writes evidence outputs.
- `scripts/run_threshold_sweep.py`: evaluates threshold settings and writes sweep tables/charts.
- `scripts/validate_outputs.py`: checks generated outputs for basic correctness.
- `app.py`: Streamlit entrypoint for the dashboard.

## Source Code

- `src/data/febrl_loader.py`: loads FEBRL4 and writes the dataset profile.
- `src/empi/preprocessing.py`: standardises identity fields and helper blocking fields.
- `src/empi/blocking.py`: creates candidate record pairs with multi-pass blocking.
- `src/empi/comparison.py`: creates field-level comparison features.
- `src/empi/matcher.py`: scores candidate pairs and assigns Auto Match, Auto Non-match, or Needs Human Review.
- `src/empi/hitl.py`: manages the grey-zone review queue and reviewer audit log.
- `src/evaluation/metrics.py`: calculates the three formal evaluation conditions.
- `src/evaluation/report_outputs.py`: writes tables, figures, and concise technical summaries.

## Outputs

- `outputs/tables/`: CSV evidence tables.
- `outputs/figures/`: report-ready charts.
- `outputs/reports/`: concise technical summaries for the final report.
- `data/review_decisions.csv`: local live review audit log, ignored by Git.
- `outputs/tables/simulated_review_decisions.csv`: benchmark-only simulated grey-zone review decisions.
