"""Aspect-Based Sentiment Analysis script to extract Pros and Cons from product reviews.

Features:
  - Defines pre-defined keywords for each aspect (Battery, Performance, Camera, Design, Service).
  - Classifies review sentences into aspects.
  - Computes weighted positivity using Helpfulness_Count.
  - Summarizes top Pros and Cons per product.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

# Define our aspect keyword map (Vietnamse lowercase)
ASPECT_KEYWORDS = {
    "Battery/Charging": [
        "pin", "sạc", "sac", "dung lượng pin", "pin trâu", "tụt pin", "nóng máy khi sạc",
        "nhanh hết pin", "pin yếu", "sạc nhanh", "sac nhanh"
    ],
    "Performance/OS": [
        "mượt", "lag", "giật", "đơ", "chơi game", "load", "hiệu năng", "chip", "ram", "rom",
        "máy mát", "nóng máy", "loạn màn hình", "treo máy", "nhanh", "chậm", "tốc độ"
    ],
    "Camera": [
        "camera", "chụp ảnh", "quay video", "chụp hình", "sắc nét", "net", "quay phim", "selfie",
        "cam trước", "cam sau", "mờ", "xấu", "đẹp"
    ],
    "Design/Build": [
        "thiết kế", "ngoại hình", "màu sắc", "đẹp", "sang trọng", "nhẹ", "nặng", "ọp ẹp", "chắc chắn",
        "dễ xước", "vỡ", "trầy", "viền", "mặt lưng"
    ],
    "Service/Staff": [
        "nhân viên", "tư vấn", "phục vụ", "nhiệt tình", "hỗ trợ", "dễ thương", "chu đáo", "giao hàng",
        "thái độ", "đổi trả", "bảo hành"
    ]
}

# Define sentiment cues (Vietnamese lowercase)
POSITIVE_CUES = [
    "tốt", "mượt", "trâu", "đẹp", "ưng ý", "nhiệt tình", "nhanh", "dễ thương", "chu đáo",
    "ngon", "rất thích", "tuyệt vời", "sang trọng", "sắc nét", "hài lòng", "ok", "ổn"
]

NEGATIVE_CUES = [
    "tệ", "chậm", "lag", "giật", "đơ", "nóng", "ọp ẹp", "mờ", "xấu", "lỗi", "yếu", "hỏng",
    "xước", "trầy", "thất vọng", "rè", "loạn", "hơi hao", "rít", "chán", "kém"
]


def extract_aspect_sentiments(review_text: str, rating: float) -> list[dict]:
    """Split review text into phrases/sentences and extract aspect sentiments."""
    if pd.isna(review_text) or not isinstance(review_text, str):
        return []
    
    # Split text into sentences/clauses using common punctuation and conjunctions
    clauses = re.split(r"[.,;!?\n]|\band\b|\bnhưng\b|\bnhung\b", review_text.lower())
    
    extracted = []
    for clause in clauses:
        clause = clause.strip()
        if len(clause) < 3:
            continue
            
        for aspect, keywords in ASPECT_KEYWORDS.items():
            # Check if any keyword matches this clause
            if any(re.search(rf"\b{kw}\b", clause) for kw in keywords):
                # Classify strictly to prevent wrong overlaps (e.g. if 'pin tụt' is in clause, don't class as Performance just because of 'tụt')
                if aspect == "Performance/OS" and any(pb in clause for pb in ["pin", "sạc", "sac"]):
                    continue
                if aspect == "Design/Build" and any(cam in clause for cam in ["camera", "chụp", "quay"]):
                    continue
                    
                # Sentiment decision: default to rating-based, override if strong negative/positive cue found
                sentiment = 1 if rating >= 4 else (0 if rating <= 2 else 0.5)
                
                # Check for explicit cues in the specific clause
                has_pos = any(re.search(rf"\b{cue}\b", clause) for cue in POSITIVE_CUES)
                has_neg = any(re.search(rf"\b{cue}\b", clause) for cue in NEGATIVE_CUES)
                
                if has_neg and not has_pos:
                    sentiment = 0
                elif has_pos and not has_neg:
                    sentiment = 1
                
                extracted.append({
                    "Aspect": aspect,
                    "Clause": clause,
                    "Sentiment": sentiment
                })
    return extracted


def analyze_aspects(reviews_path: Path, output_path: Path) -> None:
    """Analyze reviews and output a summarized JSON with Pros and Cons per product."""
    if not reviews_path.exists():
        print(f"Reviews file not found: {reviews_path}")
        return
        
    df = pd.read_csv(reviews_path)
    
    # Handle optional Helpfulness_Count field safely
    if "Helpfulness_Count" in df.columns:
        df["Helpfulness_Count"] = pd.to_numeric(df["Helpfulness_Count"], errors="coerce").fillna(0).astype(int)
    else:
        df["Helpfulness_Count"] = 0
        
    df["Star_Rating"] = pd.to_numeric(df["Star_Rating"], errors="coerce").fillna(5)
    
    # Group reviews by product
    teencode_path = Path("data-project/teencode_dict.json")
    teencode = {}
    if teencode_path.exists():
        try:
            with open(teencode_path, "r", encoding="utf-8") as f:
                teencode = json.load(f)
        except Exception:
            pass

    # Helper function to clean and map teencode for aspect text
    def clean_text_with_teencode(t: str) -> str:
        if pd.isna(t):
            return ""
        t = str(t).lower()
        t = re.sub(r"<[^>]+>", " ", t)
        t = re.sub(r"https?://\S+|www\.\S+", " ", t)
        t = re.sub(r"[^0-9a-zA-ZÀ-ỹ\s]", " ", t)
        tokens = t.split()
        tokens = [teencode.get(tok, tok) for tok in tokens]
        return re.sub(r"\s+", " ", " ".join(tokens)).strip()

    results = {}

    for product_id, group in df.groupby("Product_ID"):
        product_name = group["Product_Name"].iloc[0] if "Product_Name" in group.columns else product_id
        
        aspect_data = {asp: {"pos_weight": 0.0, "total_weight": 0.0, "clauses": []} for asp in ASPECT_KEYWORDS}
        
        for _, row in group.iterrows():
            raw_text = row["Review_Text"]
            rating = row["Star_Rating"]
            weight = 1 + int(row["Helpfulness_Count"])  # weigh heavily liked reviews
            
            # Preprocess raw_text with teencode dictionary before extracting aspect sentiments
            clean_t = clean_text_with_teencode(raw_text)
            clause_sentiments = extract_aspect_sentiments(clean_t, rating)
            for item in clause_sentiments:
                asp = item["Aspect"]
                sentiment = item["Sentiment"]
                
                aspect_data[asp]["pos_weight"] += sentiment * weight
                aspect_data[asp]["total_weight"] += weight
                aspect_data[asp]["clauses"].append({
                    "text": item["Clause"],
                    "sentiment": sentiment,
                    "weight": weight
                })
        
        # Format the summary of Pros and Cons
        summary = {
            "Product_Name": product_name,
            "Aspects": {}
        }
        
        for asp, data in aspect_data.items():
            if data["total_weight"] == 0:
                continue
                
            pos_ratio = data["pos_weight"] / data["total_weight"]
            
            # Extract sample Pros (positive clauses) and Cons (negative clauses) sorted by weight
            pos_clauses = [c["text"] for c in data["clauses"] if c["sentiment"] == 1]
            neg_clauses = [c["text"] for c in data["clauses"] if c["sentiment"] == 0]
            
            # Deduplicate clauses while preserving order
            unique_pos = list(dict.fromkeys(pos_clauses))[:5]
            unique_neg = list(dict.fromkeys(neg_clauses))[:5]
            
            summary["Aspects"][asp] = {
                "Score": round(pos_ratio, 2),  # Score between 0 (very poor) and 1 (excellent)
                "Pros": unique_pos,
                "Cons": unique_neg
            }
            
        results[product_id] = summary
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"Aspect Analysis written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aspect-Based Sentiment Analysis")
    parser.add_argument("--input", type=Path, default=Path("data-project/raw/reviews.csv"))
    parser.add_argument("--output", type=Path, default=Path("data-project/processed/aspect_sentiment.json"))
    args = parser.parse_args()
    
    analyze_aspects(args.input, args.output)
