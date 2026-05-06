import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.active_learning import run_active_learning_experiment  # noqa: E402


def main() -> None:
    """Run the active-learning simulation and write presentation-ready outputs."""
    outputs = run_active_learning_experiment()
    print("Active-learning outputs generated:")
    for name, table in outputs.items():
        print(f"- {name}: {len(table):,} rows")


if __name__ == "__main__":
    main()
