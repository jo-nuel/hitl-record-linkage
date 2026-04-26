from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathConfig:
    data_dir: Path = Path("data")
    processed_dir: Path = data_dir / "processed"
    review_decisions: Path = data_dir / "review_decisions.csv"

    outputs_dir: Path = Path("outputs")
    tables_dir: Path = outputs_dir / "tables"
    reports_dir: Path = outputs_dir / "reports"
    figures_dir: Path = outputs_dir / "figures"

    febrl_a: Path = processed_dir / "febrl_a.csv"
    febrl_b: Path = processed_dir / "febrl_b.csv"
    true_links: Path = processed_dir / "febrl_true_links.csv"

    classified_pairs: Path = tables_dir / "classified_pairs.csv"
    review_queue: Path = tables_dir / "review_queue.csv"
    review_decisions_export: Path = tables_dir / "review_decisions.csv"
    final_decisions: Path = tables_dir / "final_decisions.csv"
    evaluation_metrics: Path = tables_dir / "evaluation_metrics.csv"
    blocking_stats: Path = tables_dir / "blocking_stats.csv"
    benchmark_table: Path = tables_dir / "benchmark_comparison_table.csv"
    workload_table: Path = tables_dir / "workload_summary_table.csv"
    decision_counts_table: Path = tables_dir / "decision_counts_table.csv"
    threshold_sweep: Path = tables_dir / "threshold_sweep.csv"

    dataset_profile: Path = reports_dir / "dataset_profile.md"
    methodology_summary: Path = reports_dir / "methodology_summary.md"
    problem_formulation: Path = reports_dir / "problem_formulation.md"
    evaluation_summary: Path = reports_dir / "evaluation_summary.md"
    limitations: Path = reports_dir / "limitations.md"
    weekly_reflection_change_summary: Path = reports_dir / "weekly_reflection_change_summary.md"
    experiment_summary: Path = reports_dir / "experiment_summary.md"
    threshold_sweep_summary: Path = reports_dir / "threshold_sweep_summary.md"

    benchmark_figure: Path = figures_dir / "benchmark_comparison.png"
    workload_figure: Path = figures_dir / "review_workload.png"
    decision_distribution_figure: Path = figures_dir / "decision_distribution.png"
    score_distribution_figure: Path = figures_dir / "score_distribution.png"
    resolution_flow_figure: Path = figures_dir / "resolution_flow.png"
    threshold_f1_figure: Path = figures_dir / "threshold_vs_f1.png"
    threshold_workload_figure: Path = figures_dir / "threshold_vs_review_workload.png"
    recall_workload_figure: Path = figures_dir / "recall_vs_review_workload.png"


@dataclass
class DatasetConfig:
    name: str = "febrl4"


@dataclass
class MatcherConfig:
    lower_threshold: float = 0.50
    upper_threshold: float = 0.80
    ecm_weight: float = 0.25
    manual_review_seconds_per_pair: float = 15.0

    def validate(self) -> None:
        if self.lower_threshold >= self.upper_threshold:
            raise ValueError("lower_threshold must be less than upper_threshold")
        if not 0 <= self.ecm_weight <= 1:
            raise ValueError("ecm_weight must be between 0 and 1")


@dataclass
class AppConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)

    def validate(self) -> None:
        self.matcher.validate()


CONFIG = AppConfig()
