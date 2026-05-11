# AI-Assisted Active Learning HITL Record Linkage

University research prototype for:

**AI-Assisted Human-in-the-Loop Record Linkage for Detecting Duplicate Patient Records in Healthcare Datasets**

Final method: **AI-Assisted Active Learning HITL Record Linkage using FEBRL4**.

The EMPI-inspired pipeline provides the healthcare-style record linkage structure: preprocessing, blocking, field-level comparison, threshold logic, and professional-style review of uncertain cases. The Active Learning ML Matcher is the main proposed AI + HITL method. In this workflow, a classifier learns from field-level comparison features, selects uncertain pairs for review, receives simulated professional reviewer labels, retrains in batches, and is evaluated on a frozen test set.

Final academic report drafts are kept outside the repository. This repository contains the technical artefact, reproducibility documentation, and generated evidence outputs used to support the final report.

## Research Question

To what extent can an AI-assisted human-in-the-loop record linkage system improve duplicate patient record detection compared with clerical review and AI-only matching?

## Final Dataset: FEBRL4

The final prototype uses FEBRL4 because it provides two related benchmark datasets and known true links. This allows duplicate record detection to be evaluated reproducibly without relying on internally generated duplicate labels.

FEBRL4 is loaded from the Python Record Linkage Toolkit. It supports two-file linkage with dataset A, dataset B, and ground-truth links.

FEBRL is fictitious benchmark data. It is not real hospital production data.

No raw dataset download is required. FEBRL4 is loaded through the `recordlinkage` Python package.

Earlier prototypes considered other data options, but Synthea duplicate generation is not part of the final method.

## Research Claim

An AI-assisted active-learning HITL record linkage workflow can improve duplicate patient record detection by using professional review feedback to improve the matcher while reducing unnecessary manual review.

## EMPI-Inspired HITL Workflow

1. Load FEBRL4.
2. Preprocess identity fields.
3. Generate candidate pairs with multi-pass blocking.
4. Compare field-level evidence.
5. Score pairs with ECM probability and the Hybrid EMPI-style evidence score.
6. Classify pairs as Auto Match, Auto Non-match, or Needs Human Review.
7. Store reviewer decisions in an audit log.
8. Evaluate workload and accuracy.

## Method Positioning

The main system is **EMPI-inspired HITL record linkage**. It provides the healthcare-style workflow: preprocessing, blocking, field-level comparison, threshold-based decision logic, and grey-zone human review.

The **Hybrid EMPI-style score** is kept as a transparent fallback scoring method and explainability layer. It is based on field-level weights and disagreement penalties, but it is not one of the final evaluation methods.

The **Active Learning ML Matcher** is the main proposed AI + HITL method. It learns from field-level comparison features, predicts match probability, selects uncertain pairs for review, simulates reviewer labels with FEBRL ground truth for reproducible experiments, and retrains in batches.

The final report-facing evaluation focuses only on Human-only Clerical Review Baseline, AI-only ML Matcher, and AI + HITL Active Learning Matcher.

## Central Final Evaluation Methods

1. Human-only Clerical Review Baseline
   Every candidate pair is reviewed using FEBRL true links. This is a clerical benchmark, not a claim about real hospital deployment.

2. AI-only ML Matcher
   A supervised classifier predicts match probability from field-level comparison features without active-learning review batches.

3. AI + HITL Active Learning Matcher
   The main proposed method. The classifier selects uncertain pairs, receives simulated reviewer labels, retrains in batches, and evaluates on a frozen test set.

## Developer Quick Start

Recommended Python version: **Python 3.11**.

This project uses the Python scientific stack plus the Python Record Linkage Toolkit. The dependency pins in `requirements.txt` are intended for Python 3.11. Avoid Python 3.13 for this project because some pinned packages may not provide compatible wheels.

No raw dataset download is required. FEBRL4 is loaded through the `recordlinkage` package during the pipeline run.

### 1. Clone And Enter The Repository

```bash
git clone <repo-url>
cd hitl-record-linkage
```

If you already have the repository, pull the latest version first:

```bash
git pull origin main
```

### 2. Create A Virtual Environment

```bash
python --version
python -m venv .venv
```

Activate the environment for your platform.

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

If installation fails, first confirm `python --version` shows Python 3.11.x. Most dependency issues are caused by using a newer Python version than the pinned packages support.

## Run The Project

Run commands from the repository root, not from inside `src/`, `scripts/`, or `app/`.

### 1. Clean Benchmark Run

Use `simulate` for reproducible benchmark outputs. This mode uses FEBRL ground truth to simulate an ideal human reviewer for grey-zone pairs.

```bash
python scripts/run_pipeline.py --review-mode simulate
```

This creates or refreshes the main outputs under:

- `outputs/tables/`
- `outputs/reports/`
- `outputs/figures/`
- `data/review_decisions.csv` if live review decisions are saved through the dashboard

### 2. Run Threshold Sweep

```bash
python scripts/run_threshold_sweep.py
```

This evaluates multiple lower and upper threshold settings and writes threshold analysis tables and figures to `outputs/`.

### 3. Run Active-Learning Experiment

```bash
python scripts/run_active_learning.py
```

This trains lightweight classifiers on FEBRL candidate-pair comparison features, simulates reviewer labels using FEBRL ground truth, and writes active-learning evidence outputs to `outputs/`.

The active-learning run also tunes Logistic Regression, Random Forest, and Gradient Boosting with `GridSearchCV`. Tuning uses the initial active-learning seed labels only. The frozen test set is reserved for final evaluation.

Active-learning benchmark labels are simulated for reproducibility. Live Streamlit review clicks are stored for demonstration and audit logging, but they are not used as the default active-learning benchmark labels.

### 4. Validate Outputs

```bash
python scripts/validate_outputs.py
```

The validation script checks that key tables, reports, figures, and review queue files exist and follow the expected project contract.

Additional reproducibility notes are in `docs/reproducibility_checklist.md`.
Repository structure notes are in `docs/repository_structure.md`.

### 5. Run Dashboard

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal. It is usually `http://localhost:8501`.

Dashboard review modes:

- `simulate` uses FEBRL ground truth to simulate an ideal human reviewer resolving grey-zone pairs. This is best for reproducible benchmark evaluation.
- `merge` uses saved live reviewer decisions from `data/review_decisions.csv` where available. This is best for demo continuity, not the default benchmark.
- `ignore` does not apply human correction to grey-zone pairs. This is best for AI-only style inspection.

Formal benchmark metrics are generated from the evaluation pipeline. The AI + HITL result uses simulated grey-zone review based on FEBRL ground truth to represent an idealised human reviewer. Live reviewer decisions in Streamlit are stored for demonstration and audit logging, but they do not automatically overwrite formal benchmark metrics unless the pipeline is explicitly rerun in merge mode.

`review_queue.csv`, `classified_pairs.csv`, and `final_decisions.csv` are exported without ground-truth labels so reviewer-facing and presentation-facing files do not reveal the answer. Simulated benchmark review decisions are written separately to `outputs/tables/simulated_review_decisions.csv`.

Dashboard pages:

- Overview
- Dataset & Blocking
- Field Evidence
- Active Learning Workflow
- Human Review Queue
- Model Performance
- Learning Curves
- Final Evaluation
- Threshold Analysis
- Report Evidence

## Platform Notes

- **Windows:** Use PowerShell or Command Prompt from the repository root. If PowerShell blocks virtual environment activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same terminal, then activate `.venv` again.
- **macOS:** Use Python 3.11 from python.org, Homebrew, or pyenv. If `python` points to Python 2 or an older system Python, use `python3.11` when creating the virtual environment.
- **Apple Silicon Macs:** The pinned packages should install through prebuilt wheels on Python 3.11. If pip tries to build NumPy, pandas, scikit-learn, or matplotlib from source, the Python version is probably wrong or pip is outdated.
- **Linux:** Install Python 3.11 and the matching `venv` package first if your distribution does not include it by default.
- **All platforms:** Run `python -m pip install --upgrade pip` before installing requirements.

## Common Setup Problems

- **`ModuleNotFoundError: recordlinkage`:** The virtual environment is not active, or requirements were installed into a different Python environment.
- **Build errors for NumPy, pandas, scikit-learn, or matplotlib:** Use Python 3.11 and upgrade pip before installing requirements.
- **`streamlit` command not found:** Run `python -m streamlit run app.py` while the virtual environment is active.
- **FEBRL dataset not found:** Reinstall requirements. FEBRL is provided by the `recordlinkage` package and does not require a separate dataset download.
- **Outputs are missing:** Run `python scripts/run_pipeline.py --review-mode simulate` before opening the dashboard or validating outputs.

## Scoring And Learning Methods

The automated EMPI matcher blends two signals:

- `recordlinkage.ECMClassifier` probability where available.
- A Hybrid EMPI-style evidence score based on field-level agreement.

The hybrid score gives stronger weight to fields such as date of birth, surname, postcode, and address because they provide stronger identity evidence than broader or often-missing fields. Pairs above the upper threshold become Auto Match. Pairs below the lower threshold become Auto Non-match. Pairs between the thresholds enter grey-zone human review.

The Active Learning ML Matcher trains Logistic Regression, Random Forest, and Gradient Boosting classifiers on the same field-level comparison features. `GridSearchCV` tunes each classifier using seed labels only, then the tuned classifier with the strongest cross-validation F1-score is used for active-learning rounds. Hybrid EMPI remains available as a transparent fallback and development reference, but the final evaluation does not treat it as a method.

## Main Outputs

The `outputs/reports/` files are generated technical summaries, not final academic report drafts. They document what was run, key configuration, metrics, assumptions, and limitations.

- `outputs/tables/final_evaluation_comparison.csv`
- `outputs/tables/final_research_evaluation.csv`
- `outputs/tables/evaluation_metrics.csv`
- `outputs/tables/threshold_sweep.csv`
- `outputs/tables/model_comparison.csv`
- `outputs/tables/hyperparameter_tuning.csv`
- `outputs/tables/active_learning_rounds.csv`
- `outputs/tables/review_queue.csv`
- `outputs/tables/final_decisions.csv`
- `outputs/tables/simulated_review_decisions.csv`
- `outputs/reports/dataset_profile.md`
- `outputs/reports/blocking_summary.md`
- `outputs/reports/scoring_method_summary.md`
- `outputs/reports/evaluation_summary.md`
- `outputs/reports/active_learning_summary.md`
- `outputs/reports/hyperparameter_tuning_summary.md`
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
- `outputs/figures/model_comparison_f1.png`
- `outputs/figures/active_learning_curve.png`
- `outputs/figures/label_efficiency_curve.png`
- `outputs/figures/final_research_evaluation.png`

## Limitations

- FEBRL is benchmark data, not real hospital production data.
- AI + HITL uses simulated ideal review when calculating final evaluation metrics.
- Active-learning experiments also use simulated FEBRL labels for reproducibility.
- Blocking can miss true links before the matcher or reviewer sees them.
- Thresholds are selected from a small sweep and need further validation before any deployment claim.
