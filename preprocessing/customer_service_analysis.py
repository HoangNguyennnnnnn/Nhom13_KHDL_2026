"""Customer Service Analysis script to measure support impact and identify hot alerts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np


def analyze_customer_service(reviews_path: Path, output_dir: Path) -> None:
    if not reviews_path.exists():
        print(f"Reviews file not found: {reviews_path}")
        return
        
    df = pd.read_csv(reviews_path)
    
    # Standardize data fields
    df["Star_Rating"] = pd.to_numeric(df["Star_Rating"], errors="coerce").fillna(5)
    df["Helpfulness_Count"] = pd.to_numeric(df["Helpfulness_Count"], errors="coerce").fillna(0).astype(int)
    
    # 1. Aspect 3: Support Impact Statistics
    # Filter only those that actually have reviews or star ratings
    total_reviews_with_stars = len(df)
    
    support_groups = df.groupby("Support_Contacted")
    support_stats = {}
    
    for contacted, group in support_groups:
        contacted_str = "contacted" if int(contacted) == 1 else "not_contacted"
        support_stats[contacted_str] = {
            "count": int(len(group)),
            "average_star": float(round(group["Star_Rating"].mean(), 2)),
            "satisfied_ratio": float(round(len(group[group["Star_Rating"] >= 4]) / len(group), 2)) if len(group) > 0 else 0.0
        }
        
    # Calculate satisfaction uplift
    uplift = 0.0
    if "contacted" in support_stats and "not_contacted" in support_stats:
        uplift = round(support_stats["contacted"]["average_star"] - support_stats["not_contacted"]["average_star"], 2)
        
    impact_summary = {
        "overall_reviews": total_reviews_with_stars,
        "support_statistics": support_stats,
        "satisfaction_star_uplift": uplift
    }
    
    # 2. Aspect 4: Critical Negatives and Alerts (Crisis Warning)
    # Filter reviews: low rating (1-2 stars), sort by popularity (Helpfulness_Count)
    negative_reviews = df[df["Star_Rating"] <= 2].copy()
    
    # Highlight keywords that indicate absolute critical crisis
    crisis_keywords = [
        "lừa đảo", "lua dao", "gian dối", "gian doi", "tẩy chay", "tay chay", 
        "bức xúc", "buc xuc", "treo táo", "màn hình chói", "lỗi loa", "rè", 
        "mờ", "nhoè", "ọp ẹp", "sập nguồn", "hư", "đểu", "lỏ", "lỡm"
    ]
    
    def calculate_crisis_score(row) -> float:
        text = str(row["Review_Text"]).lower()
        score = float(row["Helpfulness_Count"]) * 2.0  # highly backed by community
        
        # Add score for explicit crisis words
        matches = sum(1 for kw in crisis_keywords if kw in text)
        score += matches * 15.0
        
        # If not contacted yet, prioritize it significantly
        if int(row["Support_Contacted"]) == 0:
            score += 25.0
            
        return score
        
    negative_reviews["crisis_score"] = negative_reviews.apply(calculate_crisis_score, axis=1)
    
    # Sort and take top 10 critical issues
    top_alerts = negative_reviews.sort_values(by="crisis_score", ascending=False).head(10)
    
    alerts_list = []
    for _, row in top_alerts.iterrows():
        alerts_list.append({
            "Review_ID": str(row["Review_ID"]),
            "Product_ID": str(row["Product_ID"]),
            "Product_Name": str(row["Product_Name"]),
            "Star_Rating": int(row["Star_Rating"]),
            "Review_Text": str(row["Review_Text"]),
            "Helpfulness_Count": int(row["Helpfulness_Count"]),
            "Support_Contacted": int(row["Support_Contacted"]),
            "Crisis_Score": round(float(row["crisis_score"]), 1)
        })
        
    final_output = {
        "support_impact": impact_summary,
        "critical_alerts": alerts_list
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "customer_service_analysis.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
        
    print(f"Customer Service Analysis successfully written to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Service Analysis")
    parser.add_argument("--input", type=Path, default=Path("data-project/raw/reviews.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data-project/processed"))
    args = parser.parse_args()
    
    analyze_customer_service(args.input, args.output_dir)
