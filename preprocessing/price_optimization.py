"""ML model to optimize discounts and estimate price elasticity of demand per brand/product."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def run_price_optimization(products_path: Path, output_path: Path) -> None:
    if not products_path.exists():
        print(f"Products file not found: {products_path}")
        return
        
    df = pd.read_csv(products_path)
    
    # ── Standardize required fields ──────────────────────────────────────────
    df["Original_Price"] = pd.to_numeric(df["Original_Price"], errors="coerce").fillna(0)
    df["Discounted_Price"] = pd.to_numeric(df["Discounted_Price"], errors="coerce").fillna(0)
    df["Sales_Volume"] = pd.to_numeric(df["Sales_Volume"], errors="coerce").fillna(0)
    
    # Calculate Discount Rate
    denom = df["Original_Price"].replace(0, np.nan)
    df["Discount_Rate"] = ((df["Original_Price"] - df["Discounted_Price"]) / denom).fillna(0).clip(0, 1)
    
    # Filter products that have valid pricing and sales volume
    df_valid = df[(df["Original_Price"] > 0) & (df["Sales_Volume"] > 0)].copy()
    
    if len(df_valid) < 5:
        print("Not enough product data with sales volume to calculate price elasticity.")
        return
        
    results = {}
    
    # ── Analyze Elasticity by Brand ──────────────────────────────────────────
    # Group by brand to build statistically stable models
    for brand, group in df_valid.groupby("Brand"):
        if len(group) < 2:
            # Skip brands with single product since we cannot draw a regression line
            continue
            
        X = group[["Discount_Rate"]].values
        y = group["Sales_Volume"].values
        
        # Fit linear regression model: Sales_Volume = alpha + beta * Discount_Rate
        model = LinearRegression()
        model.fit(X, y)
        
        beta = float(model.coef_[0])
        alpha = float(model.intercept_)
        r2 = float(model.score(X, y))
        
        # Mean reference points
        mean_discount = float(group["Discount_Rate"].mean())
        mean_sales = float(group["Sales_Volume"].mean())
        
        # Price Elasticity of Demand (PED proxy representing how much sales volume responds to discount changes)
        # Elasticity = % change in quantity / % change in discount rate proxy
        # Since we use Discount_Rate, PED = dQ/dD * (Mean_Discount / Mean_Sales)
        ped = 0.0
        if mean_sales > 0:
            ped = beta * (mean_discount / mean_sales) if mean_discount > 0 else beta / mean_sales
            
        # Estimate sales increase for a 5% and 10% discount rate increase
        # Delta_Q = beta * Delta_Discount
        sales_increase_5pct = max(0.0, beta * 0.05)
        sales_increase_10pct = max(0.0, beta * 0.10)
        
        pct_increase_5pct = (sales_increase_5pct / mean_sales) * 100 if mean_sales > 0 else 0.0
        pct_increase_10pct = (sales_increase_10pct / mean_sales) * 100 if mean_sales > 0 else 0.0
        
        results[brand] = {
            "Product_Count": int(len(group)),
            "Average_Original_Price": float(round(group["Original_Price"].mean(), 2)),
            "Average_Discount_Rate": float(round(mean_discount, 4)),
            "Average_Sales_Volume": float(round(mean_sales, 2)),
            "Model_R2": float(round(r2, 4)),
            "Price_Elasticity": float(round(ped, 4)),
            "Estimated_Uplift_5pct_Discount": {
                "Sales_Volume_Increase": float(round(sales_increase_5pct, 2)),
                "Percentage_Increase": float(round(pct_increase_5pct, 2))
            },
            "Estimated_Uplift_10pct_Discount": {
                "Sales_Volume_Increase": float(round(sales_increase_10pct, 2)),
                "Percentage_Increase": float(round(pct_increase_10pct, 2))
            }
        }
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"Price Elasticity and Discount Optimization successfully written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Price Elasticity & Discount Optimization")
    parser.add_argument("--input", type=Path, default=Path("data-project/processed/products_clean.csv"))
    parser.add_argument("--output", type=Path, default=Path("data-project/processed/price_elasticity.json"))
    args = parser.parse_args()
    
    run_price_optimization(args.input, args.output)
