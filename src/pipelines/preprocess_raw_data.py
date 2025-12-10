"""Preprocess raw missing person records into a normalized dataset.

This script cleans semi-structured CSV exports by:
- Standardizing column names and data types
- Imputing missing dates and ages where possible
- Normalizing Metro Manila locations to known districts
- Binning ages into categorical segments
- Enriching rows with inferred latitude and longitude values

Run as a module:
    python -m src.pipelines.preprocess_raw_data \
        --input "data/raw/My Victim List - Compiled.csv" \
        --output data/processed/missing_persons_clean.csv
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


AGE_KEYWORD_MAP: Dict[str, int] = {
    "INFANT": 1,
    "TODDLER": 3,
    "CHILD": 10,
    "YOUNG CHILD": 8,
    "PRETEEN": 12,
    "TEEN": 16,
    "YOUNG TEEN": 15,
    "YOUNG ADULT": 22,
    "ADULT": 30,
    "ADULT OLDER": 50,
    "SENIOR": 70,
    "OLDER ADULT": 60,
    "ELDERLY": 72,
}


@dataclass
class LocationRule:
    """Mapping rule for extracting normalized location metadata."""

    district: str
    city: str
    latitude: float
    longitude: float
    keywords: Tuple[str, ...]


LOCATION_RULES: List[LocationRule] = [
    LocationRule("Tondo", "Manila", 14.6195, 120.9723, ("TONDO", "GAGALANGIN", "BALUT")),
    LocationRule("Binondo", "Manila", 14.6001, 120.9772, ("BINONDO", "ONGPIN")),
    LocationRule("Santa Cruz", "Manila", 14.6106, 120.9830, ("SANTA CRUZ", "STA CRUZ", "ANDRES A BONIFACIO")),
    LocationRule("Quiapo", "Manila", 14.6004, 120.9830, ("QUIAPO", "CARIEDO")),
    LocationRule("Sampaloc", "Manila", 14.6116, 120.9895, ("SAMPALOC", "UNIVERSITY BELT")),
    LocationRule("Santa Mesa", "Manila", 14.6037, 121.0151, ("STA MESA", "SANTA MESA")),
    LocationRule("Malate", "Manila", 14.5693, 120.9887, ("MALATE", "SAN ANDRES BUKID", "LEVERIZA")),
    LocationRule("Ermita", "Manila", 14.5822, 120.9822, ("ERMITA", "INTRAMUROS", "MANILA ARENA", "RIZAL PARK")),
    LocationRule("Paco", "Manila", 14.5836, 120.9941, ("PACO", "PANDEQUINA", "PEDRO GIL")),
    LocationRule("Pandacan", "Manila", 14.5883, 121.0015, ("PANDACAN",)),
    LocationRule("Santa Ana", "Manila", 14.5845, 121.0129, ("STA ANA", "SANTA ANA")),
    LocationRule("Port Area", "Manila", 14.5893, 120.9686, ("PORT AREA",)),
    LocationRule("Navotas", "Navotas", 14.6667, 120.9547, ("NAVOTAS",)),
    LocationRule("Malabon", "Malabon", 14.6688, 120.9552, ("MALABON",)),
    LocationRule("Caloocan", "Caloocan", 14.6549, 120.9656, ("CALOOCAN", "BAGONG SILANG", "CAMARIN", "SANGANDAAN")),
    LocationRule("Valenzuela", "Valenzuela", 14.7060, 120.9772, ("VALENZUELA", "MALINTA")),
    LocationRule("Quezon City", "Quezon City", 14.6507, 121.0499, ("QUEZON CITY", "COMMONWEALTH", "NOVALICHES", "UGONG", "PROJECT", "HOLY SPIRIT", "DONA MARIAN")),
    LocationRule("San Juan", "San Juan", 14.6042, 121.0296, ("SAN JUAN", "GREENHILLS")),
    LocationRule("Mandaluyong", "Mandaluyong", 14.5794, 121.0359, ("MANDALUYONG", "SHAW", "BONIFACIO", "TIOSEJO")),
    LocationRule("Pasig", "Pasig", 14.5869, 121.0614, ("PASIG", "NAPICO", "MANGGAHAN", "ROSARIO", "KARANGALAN")),
    LocationRule("Taguig", "Taguig", 14.5176, 121.0509, ("TAGUIG", "BGC", "BAMBANG NI PELES", "STA ANA TAGUIG")),
    LocationRule("Pateros", "Pateros", 14.5450, 121.0665, ("PATEROS",)),
    LocationRule("Makati", "Makati", 14.5547, 121.0244, ("MAKATI", "PASEO DE ROXAS", "LEGASPI", "VILLAMOR", "NEWPORT")),
    LocationRule("Muntinlupa", "Muntinlupa", 14.4081, 121.0415, ("MUNTINLUPA", "ALABANG", "SUCAT")),
    LocationRule("Paranaque", "Paranaque", 14.4789, 120.9809, ("PARANAQUE", "PARANAQUE CITY", "TAMBO", "BICUTAN")),
    LocationRule("Las Pinas", "Las Pinas", 14.4505, 120.9762, ("LAS PINAS", "CAA", "PAMPLONA")),
    LocationRule("Pasay", "Pasay", 14.5378, 120.9943, ("PASAY", "ANDREWS AVE", "VILLARUEL", "BACLARAN", "VILLAMOR")),
    LocationRule("Marikina", "Marikina", 14.6507, 121.1029, ("MARIKINA", "NANGKA", "CONCEPCION", "FORTUNE")),
    LocationRule("Antipolo", "Antipolo", 14.6255, 121.1245, ("ANTIPOLO", "MAYAMOT", "BINANGONAN")),
    LocationRule("Rodriguez", "Rodriguez", 14.7324, 121.1440, ("RODRIGUEZ", "MONTALBAN")),
    LocationRule("Manila", "Manila", 14.5995, 120.9842, ("MANILA", "ESPANA", "STA CRUZ MANILA", "SANTA CRUZ MANILA")),
]


AGE_BINS = [-1, 12, 17, 24, 44, 64, 200]
AGE_LABELS = ["Child", "Teen", "Young Adult", "Adult", "Middle Age", "Senior"]


def normalise_text(value: str) -> str:
    """Return uppercase text stripped of accents and punctuation."""

    nfkd = unicodedata.normalize("NFKD", value)
    ascii_text = nfkd.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Z0-9 ]", " ", ascii_text.upper())
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_age(value: object) -> Optional[float]:
    """Convert raw age value into a numeric form when possible."""

    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return float(value)

    text = normalise_text(str(value))
    if not text:
        return None

    if text.isdigit():
        return float(text)

    for keyword, fallback_age in AGE_KEYWORD_MAP.items():
        if keyword in text:
            return float(fallback_age)

    return None


def normalize_gender(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text.startswith("m"):
        return "Male"
    if text.startswith("f"):
        return "Female"
    return text.title()


def match_location(value: object) -> Optional[LocationRule]:
    if pd.isna(value):
        return None

    text = normalise_text(str(value))
    if not text:
        return None

    # Prefer explicit district matches first.
    for rule in LOCATION_RULES:
        district_token = normalise_text(rule.district)
        if district_token and district_token in text:
            return rule

    for rule in LOCATION_RULES:
        if any(keyword in text for keyword in rule.keywords):
            return rule

    return None


def parse_time_column(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.replace(" ", "", regex=False)
    parsed = pd.to_datetime(cleaned, format="%I:%M%p", errors="coerce")
    parsed = parsed.fillna(pd.to_datetime(cleaned, format="%H:%M", errors="coerce"))
    return parsed.dt.strftime("%H:%M")


def parse_date_value(value: object) -> Optional[date]:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    normalized = text.replace("-", "/")
    normalized = normalized.replace(".", "/")
    normalized = re.sub(r"[^0-9/]", "", normalized)

    date_pattern = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", normalized)
    if date_pattern:
        part1 = int(date_pattern.group(1))
        part2 = int(date_pattern.group(2))
        year_raw = date_pattern.group(3)
        year = int(year_raw)
        if len(year_raw) == 2:
            year += 2000

        if part1 > 12 and part2 <= 12:
            day = part1
            month = part2
        elif part2 > 12 and part1 <= 12:
            month = part1
            day = part2
        else:
            month = part1
            day = part2

        try:
            return date(year, month, day)
        except ValueError:
            return None

    return None


def parse_date_column(series: pd.Series) -> pd.Series:
    parsed = series.apply(parse_date_value)
    return pd.to_datetime(parsed, errors="coerce")


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [col.strip() for col in df.columns]
    rename_map = {
        "AGE": "Age",
        "GENDER": "Gender",
        "Date Reported Missing": "Date Reported Missing",
        "Time Reported Missing": "Time Reported Missing",
        "Date Last Seen": "Date Last Seen",
        "Location Last Seen": "Location Last Seen",
        "Longtitude": "Longitude",
        "Latitude": "Latitude",
        "Post URL": "Post URL",
        "Person ID": "Person ID",
    }
    df = df.rename(columns=rename_map)

    df = df.replace({
        r"^\s*$": pd.NA,
        r"^na$": pd.NA,
        r"^n/a$": pd.NA,
        r"^none$": pd.NA,
    }, regex=True, value=pd.NA)

    if "Person ID" not in df.columns or df["Person ID"].isna().all():
        df["Person ID"] = [f"MPH-{index:05d}" for index in range(1, len(df) + 1)]

    df["Gender"] = df["Gender"].apply(normalize_gender)

    df["Age"] = df["Age"].apply(parse_age)

    df["Age Group"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS)

    date_columns = ["Date Reported Missing", "Date Last Seen"]
    for column in date_columns:
        df[column] = parse_date_column(df[column])

    df["Date Last Seen"] = df["Date Last Seen"].fillna(df["Date Reported Missing"])

    df["Date Reported Missing"] = df["Date Reported Missing"].dt.date
    df["Date Last Seen"] = df["Date Last Seen"].dt.date

    if "Time Reported Missing" in df.columns:
        df["Time Reported Missing"] = parse_time_column(df["Time Reported Missing"])

    location_matches = df["Location Last Seen"].apply(match_location)
    df["Barangay District"] = location_matches.apply(lambda rule: rule.district if rule else None)
    df["City"] = location_matches.apply(lambda rule: rule.city if rule else None)

    df["Latitude"] = pd.to_numeric(df.get("Latitude"), errors="coerce")
    df["Longitude"] = pd.to_numeric(df.get("Longitude"), errors="coerce")

    inferred_lat = location_matches.apply(lambda rule: rule.latitude if rule else None)
    inferred_lon = location_matches.apply(lambda rule: rule.longitude if rule else None)

    df["Latitude"] = df["Latitude"].fillna(inferred_lat)
    df["Longitude"] = df["Longitude"].fillna(inferred_lon)

    df.sort_values(by=["Date Reported Missing", "Person ID"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def preprocess_csv(input_path: Path, output_path: Path) -> Path:
    df = pd.read_csv(input_path)
    processed_df = preprocess_dataframe(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)

    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize raw missing person records.")
    parser.add_argument("--input", type=Path, required=True, help="Path to raw CSV file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/missing_persons_clean.csv"),
        help="Destination for the cleaned CSV",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_path = preprocess_csv(args.input, args.output)
    print(f"Saved cleaned dataset to {output_path}")


if __name__ == "__main__":
    main()
