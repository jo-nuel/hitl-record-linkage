from pathlib import Path

import pandas as pd


def ensure_directories_exist(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    ensure_directories_exist(path.parent)
    df.to_csv(path, index=index)
