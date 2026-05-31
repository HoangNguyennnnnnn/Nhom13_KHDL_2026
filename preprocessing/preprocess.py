"""Preprocess TGDD product and review data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from pyvi import ViTokenizer
except Exception:  # pragma: no cover
    ViTokenizer = None

try:
    from underthesea import word_tokenize
except Exception:  # pragma: no cover
    word_tokenize = None


RAW_DIR = Path("data-project/raw")
PROCESSED_DIR = Path("data-project/processed")
TEENCODE_PATH = Path("data-project/teencode_dict.json")
HOLIDAYS_MM_DD = {"01-01", "04-30", "05-01", "09-02"}


def parse_number(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).lower().replace(",", ".")
    digits = re.sub(r"[^\d.]", "", text)
    return float(digits) if digits else np.nan


def normalize_units(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        lower = col.lower()
        if any(unit in lower for unit in ["ram", "rom", "battery", "camera", "mah", "gb", "mp"]):
            df[col] = df[col].map(parse_number).astype("Int64")
    return df


def clean_text(text: str, teencode: dict[str, str]) -> str:
    text = "" if pd.isna(text) else str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(
        "[\U0001F300-\U0001FAFF\U00002700-\U000027BF]",
        " ",
        text,
    )
    text = re.sub(r"[^0-9a-zA-ZÀ-ỹ\s]", " ", text)
    tokens = text.split()
    tokens = [teencode.get(token, token) for token in tokens]
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


def segment_vi(text: str) -> str:
    if not text:
        return ""
    if ViTokenizer is not None:
        return ViTokenizer.tokenize(text)
    if word_tokenize is not None:
        return word_tokenize(text, format="text")
    return text


def preprocess_products(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["Original_Price", "Discounted_Price", "Sales_Volume", "Avg_Star_Rating", "Total_Reviews"]:
        if col in df.columns:
            df[col] = df[col].map(parse_number)
            df[col] = df[col].fillna(df[col].median())
    df = normalize_units(df)
    if {"Original_Price", "Discounted_Price"}.issubset(df.columns):
        denom = df["Original_Price"].replace(0, np.nan)
        df["Discount_Rate"] = ((df["Original_Price"] - df["Discounted_Price"]) / denom).fillna(0).clip(0, 1)
    date_col = "Inward_Date" if "Inward_Date" in df.columns else None
    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        df["Day_of_Week"] = dates.dt.dayofweek.fillna(0).astype(int)
        df["Is_Weekend"] = df["Day_of_Week"].isin([5, 6]).astype(int)
        df["Is_Holiday"] = dates.dt.strftime("%m-%d").isin(HOLIDAYS_MM_DD).astype(int)
    if "Product_ID" in df.columns:
        df = df.dropna(subset=["Product_ID"])
    object_cols = df.select_dtypes(include=["object"]).columns
    df[object_cols] = df[object_cols].fillna("unknown")
    return df


def preprocess_reviews(path: Path, teencode_path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    teencode = json.loads(teencode_path.read_text(encoding="utf-8")) if teencode_path.exists() else {}
    df["Review_Text_Clean"] = df["Review_Text"].map(lambda value: clean_text(value, teencode))
    df["Review_Text_Segmented"] = df["Review_Text_Clean"].map(segment_vi)
    if "Star_Rating" in df.columns and "label" not in df.columns:
        ratings = pd.to_numeric(df["Star_Rating"], errors="coerce")
        df["label"] = np.where(ratings >= 4, 1, np.where(ratings <= 2, 0, np.nan))
    return df.dropna(subset=["Review_Text_Segmented"])


def run() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    products_path = RAW_DIR / "products.csv"
    reviews_path = RAW_DIR / "reviews.csv"
    if products_path.exists():
        preprocess_products(products_path).to_csv(PROCESSED_DIR / "products_clean.csv", index=False)
    if reviews_path.exists():
        preprocess_reviews(reviews_path, TEENCODE_PATH).to_csv(PROCESSED_DIR / "reviews_clean.csv", index=False)
    if not products_path.exists() and not reviews_path.exists():
        raise FileNotFoundError("Expected data-project/raw/products.csv or data-project/raw/reviews.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()
