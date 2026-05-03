import re

import pandas as pd


FIELD_ALIASES = {
    "given_name": ["given_name", "first_name", "givenname"],
    "surname": ["surname", "last_name", "family_name"],
    "date_of_birth": ["date_of_birth", "dob", "birthdate"],
    "address": ["address", "full_address"],
    "address_1": ["address_1", "street_name"],
    "address_2": ["address_2"],
    "street_number": ["street_number"],
    "suburb": ["suburb", "place", "city"],
    "state": ["state"],
    "postcode": ["postcode", "zip"],
    "sex": ["sex", "gender"],
    "identifier": ["soc_sec_id", "identifier", "record_identifier"],
}


def _find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lookup = {column.lower(): column for column in df.columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


def _normalise_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalise_date(value: object) -> str:
    text = re.sub(r"\D", "", "" if pd.isna(value) else str(value))
    if len(text) >= 8:
        return text[:8]
    return text


def preprocess_records(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise FEBRL identity fields and create helper fields for blocking."""
    processed = pd.DataFrame(index=df.index.astype(str))
    processed.index.name = "record_id"

    for canonical, aliases in FIELD_ALIASES.items():
        source = _find_column(df, aliases)
        processed[canonical] = "" if source is None else df[source]

    for column in processed.columns:
        if column == "date_of_birth":
            processed[column] = processed[column].map(_normalise_date)
        else:
            processed[column] = processed[column].map(_normalise_text)

    if not processed["address"].str.strip().any():
        # FEBRL4 stores address components separately, so combine them for
        # comparison and reviewer display when no full address column exists.
        processed["address"] = (
            processed["street_number"] + " " + processed["address_1"] + " " + processed["address_2"]
        ).map(_normalise_text)

    processed["postcode"] = processed["postcode"].str.extract(r"(\d+)", expand=False).fillna("")
    processed["given_initial"] = processed["given_name"].str.slice(0, 1)
    processed["surname_initial"] = processed["surname"].str.slice(0, 1)
    processed["birth_year"] = processed["date_of_birth"].str.slice(0, 4)
    processed["postcode_prefix"] = processed["postcode"].str.slice(0, 2)
    return processed.fillna("")
