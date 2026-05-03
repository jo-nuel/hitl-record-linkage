import pandas as pd
from recordlinkage.datasets import load_febrl1, load_febrl2, load_febrl3, load_febrl4

from src.utils.config import CONFIG
from src.utils.io import ensure_directories_exist


def _standardise_index(df: pd.DataFrame) -> pd.DataFrame:
    """Use stable string record IDs so FEBRL links compare consistently."""
    standardised = df.copy()
    standardised.index = standardised.index.astype(str)
    standardised.index.name = "record_id"
    return standardised


def _missing_summary(df: pd.DataFrame) -> pd.Series:
    return df.replace("", pd.NA).isna().sum().sort_values(ascending=False)


def _write_dataset_profile(
    dataset_name: str,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    true_links: pd.MultiIndex,
) -> None:
    ensure_directories_exist(CONFIG.paths.reports_dir)
    display_a = df_a.rename(columns={"soc_sec_id": "identifier"})
    display_b = df_b.rename(columns={"soc_sec_id": "identifier"})
    missing_a = _missing_summary(display_a)
    missing_b = _missing_summary(display_b)
    lines = [
        "# Dataset Profile",
        "",
        f"Dataset: {dataset_name.upper()}",
        "",
        "FEBRL is fictitious benchmark data for record linkage experiments. It is not real hospital production data.",
        "",
        f"Records in dataset A: {len(df_a):,}",
        f"Records in dataset B: {len(df_b):,}",
        f"Known true links: {len(true_links):,}",
        "",
        "## Available Fields",
        "",
        ", ".join(display_a.columns.astype(str)),
        "",
        "## Missing Values, Dataset A",
        "",
        "```",
        missing_a.to_string(),
        "```",
        "",
        "## Missing Values, Dataset B",
        "",
        "```",
        missing_b.to_string(),
        "```",
    ]
    CONFIG.paths.dataset_profile.write_text("\n".join(lines), encoding="utf-8")


def load_febrl_dataset(
    dataset_name: str = "febrl4",
    return_links: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.MultiIndex]:
    """Load the FEBRL benchmark dataset used for final evaluation.

    FEBRL4 is the default because it provides two related person-record
    datasets and known true links for reproducible record linkage evaluation.
    """
    dataset = dataset_name.lower().strip()
    if dataset != "febrl4":
        loaders = {
            "febrl1": load_febrl1,
            "febrl2": load_febrl2,
            "febrl3": load_febrl3,
        }
        if dataset not in loaders:
            raise ValueError("dataset_name must be one of febrl1, febrl2, febrl3, or febrl4")
        records, links = loaders[dataset](return_links=True)
        df = _standardise_index(records).fillna("")
        _write_dataset_profile(dataset, df, df, links)
        return df, df, links

    df_a, df_b, true_links = load_febrl4(return_links=return_links)
    df_a = _standardise_index(df_a).fillna("")
    df_b = _standardise_index(df_b).fillna("")
    _write_dataset_profile(dataset, df_a, df_b, true_links)
    return df_a, df_b, true_links
