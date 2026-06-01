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
            df[col] = df[col].map(parse_number).astype("Float64")
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


def extract_specs_from_name(name: str, price: float) -> dict[str, float]:
    name_lower = str(name).lower()
    ram, rom, battery, camera = 4.0, 128.0, 5000.0, 50.0  # reasonable defaults
    
    # 1. Parse RAM & ROM from patterns like "8GB/256GB" or "8GB-256GB" or "8g/256g"
    m_pair = re.search(r"(\d+)\s*g[b]?\s*[-/]\s*(\d+)\s*g[b]?", name_lower)
    if m_pair:
        ram = float(m_pair.group(1))
        rom = float(m_pair.group(2))
    else:
        # Check single ROM pattern like "256GB" or "128GB"
        m_rom = re.search(r"(\d+)\s*g[b]?(?!\s*/)", name_lower)
        m_rom_tb = re.search(r"(\d+)\s*t[b]?", name_lower)
        if m_rom_tb:
            rom = float(m_rom_tb.group(1)) * 1024
        elif m_rom:
            rom = float(m_rom.group(1))
        
        # Heuristics for RAM if ROM is known
        if rom >= 512:
            ram = 12.0
        elif rom >= 256:
            ram = 8.0
        elif rom >= 128:
            ram = 6.0
        elif rom >= 64:
            ram = 4.0
        else:
            ram = 2.0

    # For feature phones (Nokia 105, Masstel, Fami, IZI, etc.)
    if any(k in name_lower for k in ["nokia 105", "izi", "fami", "masstel", "nokia 110"]):
        ram = 0.048  # 48MB
        rom = 0.128  # 128MB
        battery = 1000.0
        camera = 0.3
        return {"RAM": ram, "ROM": rom, "Battery": battery, "Camera_MP": camera}

    # Specific iPhone RAM heuristics
    if "iphone" in name_lower:
        if "17" in name_lower:
            ram = 8.0
        elif "16" in name_lower:
            ram = 8.0
        elif "15" in name_lower:
            ram = 8.0 if "pro" in name_lower else 6.0
        elif "14" in name_lower:
            ram = 6.0
        elif "13" in name_lower:
            ram = 4.0
        elif "12" in name_lower:
            ram = 4.0
        else:
            ram = 4.0
            
    # Battery heuristics
    if "power" in name_lower or "neo" in name_lower:
        battery = 6000.0
    elif "iphone" in name_lower:
        if "max" in name_lower or "plus" in name_lower:
            battery = 4400.0
        else:
            battery = 3200.0
    else:
        battery = 5000.0
        
    # Camera heuristics
    if "pro" in name_lower or "ultra" in name_lower:
        camera = 108.0 if "samsung" in name_lower else 50.0
    elif "iphone" in name_lower:
        camera = 48.0
    elif price > 15000000:
        camera = 50.0
    elif price > 7000000:
        camera = 50.0
    else:
        camera = 13.0
        
    return {"RAM": ram, "ROM": rom, "Battery": battery, "Camera_MP": camera}


def preprocess_products(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["Original_Price", "Discounted_Price", "Sales_Volume", "Avg_Star_Rating", "Total_Reviews"]:
        if col in df.columns:
            df[col] = df[col].map(parse_number)
            df[col] = df[col].fillna(df[col].median())
    
    # Extract specs from Name and price
    specs_list = []
    for _, row in df.iterrows():
        name = row.get("Name", "")
        price = row.get("Discounted_Price", 0)
        specs_list.append(extract_specs_from_name(name, price))
    specs_df = pd.DataFrame(specs_list)
    for col in specs_df.columns:
        df[col] = specs_df[col]
        
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
    object_cols = df.select_dtypes(include=["object", "string"]).columns
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
