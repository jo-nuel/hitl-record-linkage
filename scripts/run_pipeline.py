import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import CONFIG  # noqa: E402
from src.pipeline import run_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Week 8 HITL record linkage prototype."
    )
    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Rebuild the synthetic duplicate dataset before running the experiment.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional number of source records to use for the generated experiment dataset.",
    )
    parser.add_argument(
        "--review-mode",
        choices=["merge", "simulate", "ignore"],
        default=CONFIG.review.default_review_mode,
        help="How the pipeline should handle review decisions.",
    )
    parser.add_argument(
        "--simulate-review",
        action="store_true",
        help="Compatibility flag that sets --review-mode simulate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review_mode = "simulate" if args.simulate_review else args.review_mode
    outputs = run_experiment(
        regenerate_data=args.regenerate_data,
        review_mode=review_mode,
        sample_size=args.sample_size,
    )

    metrics_df = outputs["metrics"]
    review_queue_df = outputs["review_queue"]

    print("Week 8 prototype run complete.")
    print(f"Review mode: {review_mode}")
    print(f"Candidate pairs: {len(outputs['candidate_pairs'])}")
    print(f"Review queue size: {len(review_queue_df)}")
    print("Evaluation comparison table:")
    print(metrics_df.to_string(index=False))
    print("Key outputs:")
    print(f"- {CONFIG.paths.classified_pairs}")
    print(f"- {CONFIG.paths.review_queue}")
    print(f"- {CONFIG.paths.review_decisions_results}")
    print(f"- {CONFIG.paths.final_decisions}")
    print(f"- {CONFIG.paths.evaluation_results}")


if __name__ == "__main__":
    main()
