import argparse
import random
import re
import string
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import CONFIG
from .preprocess import preprocess_records
from .utils import ensure_directories_exist


NICKNAME_MAP = {
    "robert": "bob",
    "william": "bill",
    "james": "jim",
    "john": "jack",
    "jonathan": "jon",
    "michael": "mike",
    "david": "dave",
    "richard": "rick",
    "joseph": "joe",
    "thomas": "tom",
    "charles": "charlie",
    "daniel": "dan",
    "anthony": "tony",
    "christopher": "chris",
    "matthew": "matt",
    "elizabeth": "liz",
    "katherine": "kate",
    "jennifer": "jen",
    "alexander": "alex",
    "alexandra": "alex",
    "andrew": "andy",
    "nicholas": "nick",
    "stephanie": "steph",
}

ADDRESS_ABBREVIATIONS = {
    "street": "st",
    "road": "rd",
    "avenue": "ave",
    "boulevard": "blvd",
    "lane": "ln",
    "drive": "dr",
    "court": "ct",
    "place": "pl",
    "terrace": "ter",
    "highway": "hwy",
    "apartment": "apt",
}


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def random_case_variation(text: str) -> str:
    if not text:
        return text

    patterns = [
        text.lower(),
        text.upper(),
        text.title(),
        "".join(ch.upper() if random.random() > 0.5 else ch.lower() for ch in text),
    ]
    return random.choice(patterns)


def introduce_typo(text: str) -> str:
    if not text or len(text) < 3:
        return text

    operation = random.choice(["delete", "insert", "replace"])
    chars = list(text)
    idx = random.randint(0, len(chars) - 1)

    if operation == "delete" and len(chars) > 2:
        del chars[idx]
    elif operation == "insert":
        chars.insert(idx, random.choice(string.ascii_lowercase))
    else:
        chars[idx] = random.choice(string.ascii_lowercase)

    return "".join(chars)


def transpose_adjacent_chars(text: str) -> str:
    if not text or len(text) < 2:
        return text

    idx = random.randint(0, len(text) - 2)
    chars = list(text)
    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    return "".join(chars)


def apply_nickname(name: str) -> str:
    if not name:
        return name
    lowered = name.lower()
    if lowered in NICKNAME_MAP:
        replacement = NICKNAME_MAP[lowered]
        return replacement
    return name


def abbreviate_address(address: str) -> str:
    if not address:
        return address

    words = address.split()
    output = []
    for word in words:
        cleaned = re.sub(r"[^\w]", "", word).lower()
        output.append(ADDRESS_ABBREVIATIONS.get(cleaned, word))
    return " ".join(output)


def mutate_postcode(postcode: str) -> str:
    if not postcode:
        return postcode

    chars = list(postcode)
    digit_positions = [i for i, char in enumerate(chars) if char.isdigit()]
    if not digit_positions:
        return postcode

    idx = random.choice(digit_positions)
    chars[idx] = str(random.randint(0, 9))
    return "".join(chars)


def maybe_blank(value: str, probability: float) -> str:
    if value and random.random() < probability:
        return ""
    return value


def add_duplicate_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["source_type"] = "original"
    out["source_record_id"] = out["record_id"]
    out["duplicate_group_id"] = out["patient_id"]
    out["noise_tags"] = ""
    return out


def apply_noise_to_record(record: pd.Series) -> Tuple[Dict[str, str], List[str]]:
    cfg = CONFIG.duplicates
    noisy = record.to_dict()
    applied_noise: List[str] = []

    if noisy["first_name"] and random.random() < cfg.nickname_probability:
        updated = apply_nickname(noisy["first_name"])
        if updated != noisy["first_name"]:
            noisy["first_name"] = updated
            applied_noise.append("first_name_nickname")

    if noisy["first_name"] and random.random() < cfg.typo_probability:
        noisy["first_name"] = introduce_typo(noisy["first_name"])
        applied_noise.append("first_name_typo")

    if noisy["first_name"] and random.random() < cfg.transposition_probability:
        noisy["first_name"] = transpose_adjacent_chars(noisy["first_name"])
        applied_noise.append("first_name_transposed")

    if noisy["last_name"] and random.random() < cfg.typo_probability:
        noisy["last_name"] = introduce_typo(noisy["last_name"])
        applied_noise.append("last_name_typo")

    if noisy["last_name"] and random.random() < cfg.transposition_probability:
        noisy["last_name"] = transpose_adjacent_chars(noisy["last_name"])
        applied_noise.append("last_name_transposed")

    if noisy["address"] and random.random() < cfg.address_abbreviation_probability:
        noisy["address"] = abbreviate_address(noisy["address"])
        applied_noise.append("address_abbreviated")

    if noisy["address"] and random.random() < cfg.typo_probability:
        noisy["address"] = introduce_typo(noisy["address"])
        applied_noise.append("address_typo")

    if noisy["postcode"] and random.random() < cfg.postcode_mutation_probability:
        noisy["postcode"] = mutate_postcode(noisy["postcode"])
        applied_noise.append("postcode_mutated")

    for field_name in ["address", "city", "postcode"]:
        original_value = noisy.get(field_name, "")
        noisy[field_name] = maybe_blank(original_value, cfg.missing_field_probability)
        if original_value and not noisy[field_name]:
            applied_noise.append(f"{field_name}_missing")

    for field_name in ["first_name", "last_name", "address", "city"]:
        if noisy.get(field_name) and random.random() < cfg.case_variation_probability:
            noisy[field_name] = random_case_variation(noisy[field_name])
            applied_noise.append(f"{field_name}_case_changed")

    noisy["full_name"] = (
        f"{noisy.get('first_name', '').strip()} {noisy.get('last_name', '').strip()}".strip()
    )
    noisy["first_initial"] = noisy.get("first_name", "")[:1].lower()
    noisy["birth_year"] = (
        noisy.get("date_of_birth", "")[:4] if noisy.get("date_of_birth") else ""
    )

    return noisy, applied_noise


def generate_duplicate_records(
    base_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = CONFIG.duplicates
    num_duplicates = int(len(base_df) * cfg.duplicate_rate)

    if num_duplicates <= 0:
        raise ValueError(
            "duplicate_rate is too small and produced zero synthetic duplicates."
        )

    sampled_indices = np.random.choice(
        base_df.index, size=num_duplicates, replace=False
    )

    duplicate_rows = []
    ground_truth_rows = []

    for idx_num, idx in enumerate(sampled_indices, start=1):
        original = base_df.loc[idx]
        noisy_record, applied_noise = apply_noise_to_record(original)

        duplicate_record_id = f"{original['record_id']}_dup_{idx_num}"
        noisy_record["record_id"] = duplicate_record_id
        noisy_record["source_type"] = "synthetic_duplicate"
        noisy_record["source_record_id"] = original["record_id"]
        noisy_record["duplicate_group_id"] = original["duplicate_group_id"]
        noisy_record["noise_tags"] = "|".join(applied_noise)

        duplicate_rows.append(noisy_record)

        ground_truth_rows.append(
            {
                "record_id_a": original["record_id"],
                "record_id_b": duplicate_record_id,
                "is_duplicate": 1,
                "duplicate_group_id": original["duplicate_group_id"],
                "noise_tags": "|".join(applied_noise),
            }
        )

    duplicates_df = pd.DataFrame(duplicate_rows)
    ground_truth_df = pd.DataFrame(ground_truth_rows)
    return duplicates_df, ground_truth_df


def create_negative_pairs(
    records_df: pd.DataFrame, ground_truth_df: pd.DataFrame
) -> pd.DataFrame:
    num_negative_pairs = CONFIG.duplicates.num_negative_pairs
    all_record_ids = records_df["record_id"].tolist()

    true_pairs = {
        tuple(sorted((row["record_id_a"], row["record_id_b"])))
        for _, row in ground_truth_df.iterrows()
    }

    group_lookup = dict(zip(records_df["record_id"], records_df["duplicate_group_id"]))

    negative_pairs = set()
    attempts = 0
    max_attempts = num_negative_pairs * 25

    while len(negative_pairs) < num_negative_pairs and attempts < max_attempts:
        a, b = random.sample(all_record_ids, 2)
        pair = tuple(sorted((a, b)))

        if pair in true_pairs:
            attempts += 1
            continue

        if group_lookup.get(a) == group_lookup.get(b):
            attempts += 1
            continue

        negative_pairs.add(pair)
        attempts += 1

    return pd.DataFrame(
        [
            {
                "record_id_a": a,
                "record_id_b": b,
                "is_duplicate": 0,
                "duplicate_group_id": "",
                "noise_tags": "",
            }
            for a, b in negative_pairs
        ]
    )


def build_labeled_pairs(
    records_df: pd.DataFrame, ground_truth_df: pd.DataFrame
) -> pd.DataFrame:
    negative_df = create_negative_pairs(records_df, ground_truth_df)
    labeled_pairs_df = pd.concat([ground_truth_df, negative_df], ignore_index=True)
    labeled_pairs_df = labeled_pairs_df.sample(
        frac=1, random_state=CONFIG.duplicates.random_seed
    ).reset_index(drop=True)
    return labeled_pairs_df


def run_pipeline(input_csv: str | None = None, output_dir: str | None = None) -> None:
    CONFIG.validate()
    set_random_seed(CONFIG.duplicates.random_seed)

    input_path = Path(input_csv) if input_csv else CONFIG.paths.raw_data
    output_path = Path(output_dir) if output_dir else CONFIG.paths.processed_dir

    ensure_directories_exist(
        output_path, CONFIG.paths.reviewed_dir, CONFIG.paths.results_dir
    )

    raw_df = pd.read_csv(input_path)
    base_df = preprocess_records(raw_df)
    base_df = add_duplicate_metadata(base_df)

    duplicates_df, ground_truth_df = generate_duplicate_records(base_df)

    full_records_df = pd.concat([base_df, duplicates_df], ignore_index=True)
    full_records_df = full_records_df.sample(
        frac=1, random_state=CONFIG.duplicates.random_seed
    ).reset_index(drop=True)

    labeled_pairs_df = build_labeled_pairs(full_records_df, ground_truth_df)

    base_df.to_csv(output_path / "base_records.csv", index=False)
    full_records_df.to_csv(output_path / "synthetic_records.csv", index=False)
    ground_truth_df.to_csv(output_path / "ground_truth_duplicates.csv", index=False)
    labeled_pairs_df.to_csv(output_path / "labeled_pairs.csv", index=False)

    print("Synthetic duplicate pipeline complete.")
    print(f"Base records: {len(base_df)}")
    print(f"Synthetic duplicates: {len(duplicates_df)}")
    print(f"Total records: {len(full_records_df)}")
    print(f"True duplicate pairs: {len(ground_truth_df)}")
    print(f"Labeled pairs: {len(labeled_pairs_df)}")
    print(f"Saved to: {output_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic duplicate patient records."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to raw patients CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to save generated outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(input_csv=args.input, output_dir=args.output)
