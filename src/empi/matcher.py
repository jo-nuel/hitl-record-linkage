import numpy as np
import pandas as pd
import recordlinkage


FIELD_WEIGHTS = {
    "given_name_sim": 0.16,
    "surname_sim": 0.20,
    "date_of_birth_exact": 0.20,
    "address_sim": 0.16,
    "suburb_sim": 0.08,
    "state_exact": 0.05,
    "postcode_exact": 0.10,
    "sex_exact": 0.05,
}


def _hybrid_score(features: pd.DataFrame) -> pd.Series:
    available_weight = sum(weight for column, weight in FIELD_WEIGHTS.items() if column in features.columns)
    if available_weight == 0:
        raise ValueError("No supported comparison features found for EMPI scoring.")
    score = pd.Series(0.0, index=features.index)
    for column, weight in FIELD_WEIGHTS.items():
        if column in features.columns:
            score += pd.to_numeric(features[column], errors="coerce").fillna(0) * (weight / available_weight)
    return score.clip(0, 1)


def _apply_disagreement_penalties(score: pd.Series, features: pd.DataFrame) -> pd.Series:
    adjusted = score.copy()
    if "date_of_birth_exact" in features.columns:
        adjusted -= np.where(features["date_of_birth_exact"] < 0.5, 0.08, 0)
    if "surname_sim" in features.columns:
        adjusted -= np.where(features["surname_sim"] < 0.60, 0.08, 0)
    if "postcode_exact" in features.columns:
        adjusted -= np.where(features["postcode_exact"] < 0.5, 0.04, 0)
    if "sex_exact" in features.columns:
        adjusted -= np.where(features["sex_exact"] < 0.5, 0.04, 0)
    return adjusted.clip(0, 1)


def _ecm_probability(features: pd.DataFrame) -> pd.Series:
    binary = features.copy()
    for column in binary.columns:
        if column.endswith("_sim"):
            binary[column] = (binary[column] >= 0.85).astype(int)
        else:
            binary[column] = binary[column].astype(int)
    classifier = recordlinkage.ECMClassifier()
    try:
        classifier.fit(binary)
        return classifier.prob(binary).astype(float).clip(0, 1)
    except Exception:
        return pd.Series(np.nan, index=features.index, dtype=float)


def classify_pairs(
    features: pd.DataFrame,
    lower_threshold: float,
    upper_threshold: float,
    ecm_weight: float,
) -> pd.DataFrame:
    if lower_threshold >= upper_threshold:
        raise ValueError("lower_threshold must be less than upper_threshold")

    classified = features.copy()
    classified["hybrid_empi_score"] = _apply_disagreement_penalties(_hybrid_score(features), features)
    classified["ecm_probability"] = _ecm_probability(features)
    ecm = classified["ecm_probability"].fillna(classified["hybrid_empi_score"])
    classified["model_score"] = (
        ecm_weight * ecm + (1 - ecm_weight) * classified["hybrid_empi_score"]
    ).clip(0, 1)

    classified["model_decision"] = "Needs Human Review"
    classified.loc[classified["model_score"] >= upper_threshold, "model_decision"] = "Auto Match"
    classified.loc[classified["model_score"] <= lower_threshold, "model_decision"] = "Auto Non-match"
    classified["decision_reason"] = np.select(
        [
            classified["model_decision"] == "Auto Match",
            classified["model_decision"] == "Auto Non-match",
        ],
        [
            f"Score meets the auto-match threshold ({upper_threshold:.2f}).",
            f"Score meets the auto-non-match threshold ({lower_threshold:.2f}).",
        ],
        default="Sent to review because score is within the uncertainty band.",
    )
    return classified
