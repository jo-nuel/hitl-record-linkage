import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.empi import run_experiment  # noqa: E402
from src.utils.config import CONFIG  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FEBRL EMPI-inspired linkage pipeline.")
    parser.add_argument("--review-mode", choices=["merge", "simulate", "ignore"], default="merge")
    parser.add_argument("--lower-threshold", type=float, default=CONFIG.matcher.lower_threshold)
    parser.add_argument("--upper-threshold", type=float, default=CONFIG.matcher.upper_threshold)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_experiment(
        review_mode=args.review_mode,
        lower_threshold=args.lower_threshold,
        upper_threshold=args.upper_threshold,
    )
    print("FEBRL EMPI-inspired linkage pipeline complete.")
    print(f"Dataset A records: {len(outputs['df_a']):,}")
    print(f"Dataset B records: {len(outputs['df_b']):,}")
    print(f"True links: {len(outputs['true_links']):,}")
    print(f"Candidate pairs: {len(outputs['candidate_pairs']):,}")
    print(f"Review-needed pairs: {len(outputs['review_queue']):,}")
    print("Benchmark comparison:")
    print(outputs["final_evaluation_comparison"].to_string(index=False))
    print("Workload comparison:")
    print(outputs["workload_table"].to_string(index=False))
    print("Main outputs:")
    print(f"- {CONFIG.paths.final_evaluation_comparison}")
    print(f"- {CONFIG.paths.review_queue}")
    print(f"- {CONFIG.paths.final_decisions}")
    print(f"- {CONFIG.paths.dataset_profile}")
    print(f"- {CONFIG.paths.blocking_summary}")
    print(f"- {CONFIG.paths.figures_dir}")


if __name__ == "__main__":
    main()
