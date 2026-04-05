from dataclasses import dataclass, field
from pathlib import Path


# =========================
# PATH CONFIGURATION
# =========================


@dataclass
class PathConfig:
    base_dir: Path = Path("data")

    raw_data: Path = base_dir / "raw" / "patients.csv"
    processed_dir: Path = base_dir / "processed"
    reviewed_dir: Path = base_dir / "reviewed"
    results_dir: Path = base_dir / "results"

    base_records: Path = processed_dir / "base_records.csv"
    synthetic_records: Path = processed_dir / "synthetic_records.csv"
    ground_truth: Path = processed_dir / "ground_truth_duplicates.csv"
    labeled_pairs: Path = processed_dir / "labeled_pairs.csv"

    review_decisions: Path = reviewed_dir / "review_decisions.csv"
    generation_manifest: Path = processed_dir / "generation_manifest.json"

    classified_pairs: Path = results_dir / "classified_pairs.csv"
    review_queue: Path = results_dir / "review_queue.csv"
    review_decisions_results: Path = results_dir / "review_decisions.csv"
    final_decisions: Path = results_dir / "final_decisions.csv"
    evaluation_results: Path = results_dir / "evaluation_metrics.csv"
    experiment_summary: Path = results_dir / "experiment_summary.md"


# =========================
# DATASET COLUMN MAPPING
# =========================


@dataclass
class ColumnConfig:
    # Synthea mapping
    source_id: str = "Id"
    first_name: str = "FIRST"
    last_name: str = "LAST"
    dob: str = "BIRTHDATE"
    gender: str = "GENDER"
    address: str = "ADDRESS"
    city: str = "CITY"
    state: str = "STATE"
    postcode: str = "ZIP"


# =========================
# BLOCKING CONFIGURATION
# =========================


@dataclass
class BlockingConfig:
    use_birth_year: bool = True
    use_name_initial: bool = True
    use_postcode: bool = False  # optional, enable later if needed

    max_candidates_per_record: int = 50


# =========================
# SIMILARITY WEIGHTS
# =========================


@dataclass
class SimilarityConfig:
    weight_first_name: float = 0.20
    weight_last_name: float = 0.25
    weight_dob: float = 0.25
    weight_address: float = 0.15
    weight_postcode: float = 0.10
    weight_gender: float = 0.05

    # Sanity check: weights should sum to 1.0
    def validate(self):
        total = (
            self.weight_first_name
            + self.weight_last_name
            + self.weight_dob
            + self.weight_address
            + self.weight_postcode
            + self.weight_gender
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Similarity weights must sum to 1.0, got {total}")


# =========================
# CLASSIFICATION THRESHOLDS
# =========================


@dataclass
class ThresholdConfig:
    match_threshold: float = 0.90
    non_match_threshold: float = 0.60

    def validate(self):
        if self.non_match_threshold >= self.match_threshold:
            raise ValueError("non_match_threshold must be less than match_threshold")


# =========================
# SYNTHETIC DUPLICATE CONFIG
# =========================


@dataclass
class DuplicateConfig:
    duplicate_rate: float = 0.20
    random_seed: int = 42
    sample_size: int | None = 5000

    typo_probability: float = 0.35
    nickname_probability: float = 0.20
    missing_field_probability: float = 0.15
    postcode_mutation_probability: float = 0.20
    address_abbreviation_probability: float = 0.30
    case_variation_probability: float = 0.15
    transposition_probability: float = 0.20

    num_negative_pairs: int = 5000


# =========================
# REVIEW / EVALUATION CONFIG
# =========================


@dataclass
class ReviewConfig:
    manual_review_seconds_per_pair: float = 15.0
    allow_review_simulation: bool = True
    default_review_mode: str = "merge"


# =========================
# GLOBAL CONFIG OBJECT
# =========================


@dataclass
class AppConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    duplicates: DuplicateConfig = field(default_factory=DuplicateConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)

    def validate(self):
        self.similarity.validate()
        self.thresholds.validate()


# Singleton config
CONFIG = AppConfig()
