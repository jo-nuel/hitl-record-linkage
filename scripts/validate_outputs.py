import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import CONFIG  # noqa: E402


REQUIRED_OUTPUTS = [
    CONFIG.paths.final_evaluation_comparison,
    CONFIG.paths.threshold_sweep,
    CONFIG.paths.evaluation_summary,
    CONFIG.paths.dataset_profile,
    CONFIG.paths.blocking_summary,
    CONFIG.paths.benchmark_figure,
]


def main() -> None:
    missing = [path for path in REQUIRED_OUTPUTS if not path.exists()]
    if missing:
        print("Missing required outputs:")
        for path in missing:
            print(f"- {path}")
        raise SystemExit(1)

    print("Output validation passed.")
    for path in REQUIRED_OUTPUTS:
        print(f"- {path}")


if __name__ == "__main__":
    main()
