# AI-Assisted Active Learning HITL Record Linkage

University research prototype for:

**AI-Assisted Human-in-the-Loop Record Linkage for Detecting Duplicate Patient Records in Healthcare Datasets**

Final method: **AI-assisted active learning record linkage using FEBRL4**.

This repository contains the technical artefact, reproducibility notes, and generated evidence outputs used to support the final report. Final academic report drafts are kept outside the repository.

## Research Question

To what extent can an AI-assisted human-in-the-loop record linkage system improve duplicate patient record detection compared with clerical review and AI-only matching?

## Dataset

The final prototype uses **FEBRL4** from the Python Record Linkage Toolkit. FEBRL4 provides two related benchmark datasets and known true links, so the project can evaluate duplicate record detection without creating its own duplicate labels.

FEBRL4 is fictitious benchmark data. It is not real hospital production data.

No raw dataset download is required. FEBRL4 is loaded through the `recordlinkage` Python package.

Earlier prototypes considered other data options, but Synthea duplicate generation is not part of the final method.

## Final Method

The final workflow is:

1. Load FEBRL4 records.
2. Preprocess identity fields.
3. Use blocking to reduce the number of candidate record pairs.
4. Build field-level comparison features.
5. Train lightweight ML classifiers on those comparison features.
6. Predict match probability for candidate pairs.
7. Select uncertain pairs whose predicted match probability is close to 0.5.
8. Simulate human reviewer labels using FEBRL4 ground truth for reproducible benchmarking.
9. Add reviewed labels to the training data.
10. Retrain the classifier in batches.
11. Evaluate final accuracy and review workload on a frozen test set.

Grey-zone review in the final method is based on **model uncertainty**, not manual boundary tuning.

## Final Evaluation Methods

The final report-facing evaluation contains only three methods:

1. **Human-only Clerical Review Baseline**
   Every blocked candidate pair is reviewed using FEBRL4 true links. This estimates the workload of full clerical review over the candidate set.

2. **AI-only ML Matcher**
   A supervised classifier predicts match probability from field-level comparison features without active-learning review batches.

3. **AI + HITL Active Learning Matcher**
   The proposed method. The classifier selects uncertain pairs, receives simulated reviewer labels, retrains in batches, and is evaluated on the frozen test set.

Live Streamlit review clicks are for demonstration and audit logging. They are not the default benchmark labels.

## Developer Quick Start

Recommended Python version: **Python 3.11**.

The dependency pins in `requirements.txt` are intended for Python 3.11. Avoid Python 3.13 because some pinned packages may not provide compatible wheels.

### 1. Clone And Enter The Repository

```bash
git clone <repo-url>
cd hitl-record-linkage
```

If you already have the repository:

```bash
git pull origin main
```

### 2. Create A Virtual Environment

```bash
python --version
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If installation fails, first confirm `python --version` shows Python 3.11.x.

## Run The Project

Run commands from the repository root.

### 1. Generate Candidate-Pair Outputs

```bash
python scripts/run_pipeline.py --review-mode simulate
```

This refreshes FEBRL4 preprocessing, blocking, comparison features, review queue exports, and supporting pipeline outputs.

### 2. Run Active-Learning Experiment

```bash
python scripts/run_active_learning.py
```

This trains Logistic Regression, Random Forest, and Gradient Boosting classifiers on field-level comparison features. It tunes the models with `GridSearchCV`, selects the best classifier, simulates reviewer labels for uncertain pairs, and retrains the classifier in batches.

The frozen test set is used only for evaluation.

### 3. Validate Outputs

```bash
python scripts/validate_outputs.py
```

The validation script checks that the final evaluation, active-learning outputs, figures, and reviewer-facing files match the final project contract.

### 4. Run Dashboard

```bash
streamlit run app.py
```

Presentation path:

1. Pitch Overview
2. Data & Blocking
3. Human Review Demo
4. Learning Progress
5. Final Results

Technical appendix pages:

1. Active Learning Details
2. Field Evidence
3. ML Model Selection
4. Evidence Files

The dashboard uses pre-generated outputs for presentation. Advanced controls are hidden in the sidebar.

## Review Modes

The pipeline keeps three review modes for reproducibility and demo continuity:

- `simulate`: uses FEBRL4 ground truth to simulate reviewer labels for benchmark outputs.
- `merge`: uses saved live reviewer decisions where available.
- `ignore`: runs without applying reviewer decisions.

The final active-learning benchmark uses simulated reviewer labels so results can be reproduced.

Reviewer-facing files do not expose ground-truth labels. `review_queue.csv`, `classified_pairs.csv`, and `final_decisions.csv` must not contain `is_true_link`.

## Main Evidence Outputs

Tables:

- `outputs/tables/blocking_stats.csv`
- `outputs/tables/hyperparameter_tuning.csv`
- `outputs/tables/model_comparison.csv`
- `outputs/tables/active_learning_rounds.csv`
- `outputs/tables/final_research_evaluation.csv`
- `outputs/tables/review_queue.csv`
- `outputs/tables/final_decisions.csv`
- `outputs/tables/simulated_review_decisions.csv`

Figures:

- `outputs/figures/active_learning_curve.png`
- `outputs/figures/active_learning_error_reduction.png`
- `outputs/figures/final_accuracy_comparison.png`
- `outputs/figures/final_workload_comparison.png`
- `outputs/figures/model_comparison_f1.png`

Technical summaries:

- `outputs/reports/dataset_profile.md`
- `outputs/reports/blocking_summary.md`
- `outputs/reports/active_learning_summary.md`
- `outputs/reports/hyperparameter_tuning_summary.md`
- `outputs/reports/limitations.md`

The `outputs/reports/` files are technical summaries, not final academic report drafts.

## Platform Notes

- **Windows:** Use PowerShell or Command Prompt from the repository root. If PowerShell blocks virtual environment activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same terminal.
- **macOS:** Use Python 3.11 from python.org, Homebrew, or pyenv.
- **Apple Silicon Macs:** If pip tries to build NumPy, pandas, scikit-learn, or matplotlib from source, the Python version is probably wrong or pip is outdated.
- **Linux:** Install Python 3.11 and the matching `venv` package first if your distribution does not include it.

## Common Setup Problems

- **`ModuleNotFoundError: recordlinkage`:** The virtual environment is not active, or requirements were installed into a different Python environment.
- **Build errors for NumPy, pandas, scikit-learn, or matplotlib:** Use Python 3.11 and upgrade pip before installing requirements.
- **`streamlit` command not found:** Run `python -m streamlit run app.py` while the virtual environment is active.
- **FEBRL dataset not found:** Reinstall requirements. FEBRL is provided by the `recordlinkage` package.
- **Outputs are missing:** Run `python scripts/run_pipeline.py --review-mode simulate`, then `python scripts/run_active_learning.py`.

## Limitations

- FEBRL4 is benchmark data, not real hospital data.
- Human review labels are simulated from FEBRL4 ground truth for reproducible evaluation.
- Real reviewers may make mistakes or disagree.
- Blocking can miss true links before the model or reviewer sees them.
- The prototype runs locally and is not a deployed clinical system.
