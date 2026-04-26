import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.threshold_sweep import run_threshold_sweep  # noqa: E402
from src.utils.config import CONFIG  # noqa: E402


def main() -> None:
    sweep = run_threshold_sweep()
    print("Threshold sweep complete.")
    print(sweep.sort_values("ai_hitl_f1_score", ascending=False).head().to_string(index=False))
    print("Outputs:")
    print(f"- {CONFIG.paths.threshold_sweep}")
    print(f"- {CONFIG.paths.threshold_sweep_summary}")
    print(f"- {CONFIG.paths.figures_dir}")


if __name__ == "__main__":
    main()
