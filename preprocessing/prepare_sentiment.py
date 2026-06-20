"""Prepare the sentiment dataset (data-mining stage [3] cleaning + [4] labeling).

This module turns the raw scraped ``reviews.csv`` into a clean, 3-class labeled
table ready for feature extraction. Compared with the old pipeline it fixes the
three data problems found during EDA:

  1. **System noise** — TGDD appends operational text after a ``||`` separator
     (e.g. "|| Bộ phận bán hàng đã liên hệ hỗ trợ ngày 30/05/2026"). This carries
     no sentiment and pollutes the features, so we cut everything from ``||`` on.
  2. **Dropped neutral class** — the old rule mapped 3★ to NaN and threw the rows
     away. We keep 3★ as an explicit ``neutral`` class -> a realistic 3-class task.
  3. **Empty / duplicate reviews** — removed so they don't bias the model.

Label encoding (ordinal, negative -> positive):
    0 = negative (1-2★)   1 = neutral (3★)   2 = positive (4-5★)

Output: ``data-project/processed/reviews_labeled.csv``
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import clean_text, segment_vi


# Ordinal label mapping used everywhere downstream (train + API).
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}
NAME_TO_LABEL = {v: k for k, v in LABEL_NAMES.items()}

# --- Negation handling (feature engineering, data-mining stage [5]) -----------
# Vietnamese negators. After one of these, the following words flip polarity, so
# we tag them with a "neg_" prefix -> TF-IDF treats "neg_tốt" as its own feature,
# distinct from "tốt". This teaches a bag-of-words model about negation purely
# through data, no deep learning needed.
NEGATORS = {
    "không", "khong", "ko", "k", "kg", "khg", "hổng", "hông", "hong",
    "chẳng", "chăng", "chả", "cha", "chưa", "chua", "đừng", "dung",
    "khỏi", "đéo", "đếch", "deh", "miễn",
}
# Clause-boundary words that end the scope of a negation.
NEG_STOP = {"nhưng", "nhung", "mà", "ma", "tuy", "song", "dù", "dẫu"}
NEG_WINDOW = 3  # tag at most this many tokens after a negator

# Sentiment lexicon (seed words common in phone reviews). When a word from these
# sets falls inside a negation scope we also emit ONE shared marker token, so all
# "không + <positive>" cases collapse onto a single strong feature (NEG_POS)
# instead of many rare per-word features -> learnable even on a small dataset.
POSITIVE_LEXICON = {
    "tốt", "đẹp", "ngon", "mượt", "nhanh", "hài_lòng", "ưng", "thích", "đáng",
    "rẻ", "bền", "trâu", "xịn", "ok", "ổn", "tuyệt", "hoàn_hảo", "nét",
    "sắc_nét", "mạnh", "êm", "ngọt", "chất", "ngầu", "hời", "recommend",
}
NEGATIVE_LEXICON = {
    "tệ", "kém", "xấu", "chậm", "lag", "giật", "hỏng", "lỗi", "đắt", "nóng",
    "yếu", "dở", "thất_vọng", "chán", "tồi", "gian_dối", "bực", "đơ", "rớt",
    "mờ", "nhoè", "hư", "trầy", "lừa",
}
NEG_POS_MARKER = "neg_pos_marker"  # a negated positive word -> negative sentiment
NEG_NEG_MARKER = "neg_neg_marker"  # a negated negative word -> positive sentiment


def apply_negation_tagging(segmented_text: str, window: int = NEG_WINDOW) -> str:
    """Tag words inside a negation scope and emit shared polarity markers.

    Example: "không tốt chút nào" ->
        "không neg_tốt neg_pos_marker neg_chút neg_nào".
    The per-word ``neg_`` prefix preserves detail; the shared ``neg_pos_marker`` /
    ``neg_neg_marker`` tokens concentrate the negation signal so the model learns
    it even when individual negated words are rare. Scope ends after ``window``
    tokens or at a clause boundary (e.g. "nhưng").
    """
    if not segmented_text:
        return ""
    tokens = segmented_text.split()
    out: list[str] = []
    countdown = 0
    for tok in tokens:
        if tok in NEGATORS:
            out.append(tok)
            countdown = window
            continue
        if countdown > 0:
            if tok in NEG_STOP:
                countdown = 0
                out.append(tok)
            else:
                out.append(f"neg_{tok}")
                if tok in POSITIVE_LEXICON:
                    out.append(NEG_POS_MARKER)
                elif tok in NEGATIVE_LEXICON:
                    out.append(NEG_NEG_MARKER)
                countdown -= 1
        else:
            out.append(tok)
    return " ".join(out)


TEENCODE_PATH = Path("data-project/teencode_dict.json")
# Where the raw CSV may live (root of repo first, then the canonical raw dir).
RAW_CANDIDATES = [Path("reviews.csv"), Path("data-project/raw/reviews.csv")]
OUTPUT_PATH = Path("data-project/processed/reviews_labeled.csv")


def strip_system_noise(text: str) -> str:
    """Remove TGDD operational text appended after a ``||`` separator.

    The customer's own words always come *before* the separator, so keeping the
    left side preserves the genuine review while dropping the support note.
    """
    if pd.isna(text):
        return ""
    return str(text).split("||", 1)[0].strip()


def label_from_stars(rating) -> float:
    """Map a 1-5 star rating to the 3-class ordinal label (NaN if unparsable)."""
    value = pd.to_numeric(rating, errors="coerce")
    if pd.isna(value):
        return np.nan
    if value <= 2:
        return 0.0  # negative
    if value == 3:
        return 1.0  # neutral
    return 2.0  # positive


def _find_raw(explicit: Path | None) -> Path:
    candidates = [explicit] if explicit else RAW_CANDIDATES
    for path in candidates:
        if path and path.exists():
            return path
    raise FileNotFoundError(
        f"reviews.csv not found. Looked in: {[str(c) for c in candidates]}"
    )


def prepare(input_path: Path | None = None) -> pd.DataFrame:
    raw_path = _find_raw(input_path)
    df = pd.read_csv(raw_path)
    teencode = (
        json.loads(TEENCODE_PATH.read_text(encoding="utf-8"))
        if TEENCODE_PATH.exists()
        else {}
    )

    n_start = len(df)

    # [3] Cleaning -----------------------------------------------------------
    df["Review_Text_Raw"] = df["Review_Text"]
    df["Review_Text_NoNoise"] = df["Review_Text"].map(strip_system_noise)
    df["Review_Text_Clean"] = df["Review_Text_NoNoise"].map(
        lambda value: clean_text(value, teencode)
    )
    # Word-segmented form is what the TF-IDF vectorizer consumes.
    df["Review_Text_Segmented"] = df["Review_Text_Clean"].map(segment_vi)
    # Final feature column: segmentation + negation tagging (what the model trains on).
    df["Review_Text_Final"] = df["Review_Text_Segmented"].map(apply_negation_tagging)

    # [4] Labeling -----------------------------------------------------------
    df["label"] = df["Star_Rating"].map(label_from_stars)
    df["label_name"] = df["label"].map(LABEL_NAMES)

    # Drop rows with no usable text or no label.
    df["_clean_nonempty"] = df["Review_Text_Segmented"].str.strip().astype(bool)
    df = df[df["_clean_nonempty"] & df["label"].notna()].copy()
    n_after_empty = len(df)

    # Remove exact duplicate reviews (same cleaned text + label).
    df = df.drop_duplicates(subset=["Review_Text_Clean", "label"]).copy()
    n_after_dedup = len(df)

    df["label"] = df["label"].astype(int)
    df = df.drop(columns=["_clean_nonempty"]).reset_index(drop=True)

    # Report -----------------------------------------------------------------
    print(f"[prepare] raw rows                : {n_start}")
    print(f"[prepare] after drop empty/no-label: {n_after_empty}")
    print(f"[prepare] after dedup             : {n_after_dedup}")
    print("[prepare] label distribution      :")
    dist = df["label_name"].value_counts()
    for name in ["negative", "neutral", "positive"]:
        count = int(dist.get(name, 0))
        print(f"            {name:9s}: {count:5d} ({count / len(df) * 100:5.1f}%)")

    return df


def run(input_path: Path | None = None) -> Path:
    df = prepare(input_path)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[prepare] wrote {len(df)} rows -> {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to raw reviews.csv (defaults to ./reviews.csv or data-project/raw/).",
    )
    parser.parse_args()
    run(parser.parse_args().input)
