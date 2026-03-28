import pandas as pd

from .config import CONFIG


def classify_pair(score: float) -> str:
    thresholds = CONFIG.thresholds
    thresholds.validate()

    if score >= thresholds.match_threshold:
        return "Match"
    if score < thresholds.non_match_threshold:
        return "Non-match"
    return "Review Needed"


def classify_pairs(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["system_decision"] = result["overall_score"].apply(classify_pair)
    return result
