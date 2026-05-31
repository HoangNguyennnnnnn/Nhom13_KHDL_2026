"""Cluster smartphone products into interpretable product segments."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


SEED = 42
FEATURES = ["RAM", "ROM", "Battery", "Camera_MP", "Discounted_Price"]


def cluster_name(row: pd.Series) -> str:
    if row.get("Battery", 0) >= 5000 and row.get("Discounted_Price", 0) <= row.get("_median_price", 0):
        return "Phân khúc pin trâu giá rẻ"
    if row.get("Camera_MP", 0) >= 48:
        return "Phân khúc camera cao"
    if row.get("RAM", 0) >= 8:
        return "Phân khúc hiệu năng"
    return "Phân khúc phổ thông"


def run(input_path: Path) -> None:
    df = pd.read_csv(input_path)
    feature_cols = [col for col in FEATURES if col in df.columns]
    if len(feature_cols) < 2:
        raise ValueError(f"Need at least two product feature columns from {FEATURES}")
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    candidates = range(2, min(10, len(df)))
    scores = []
    for k in candidates:
        model = KMeans(n_clusters=k, random_state=SEED, n_init=20)
        labels = model.fit_predict(X_scaled)
        scores.append({"k": k, "inertia": model.inertia_, "silhouette": silhouette_score(X_scaled, labels)})
    best_k = max(scores, key=lambda row: row["silhouette"])["k"] if scores else 2
    model = KMeans(n_clusters=best_k, random_state=SEED, n_init=20)
    df["cluster_id"] = model.fit_predict(X_scaled)
    df["_median_price"] = pd.to_numeric(df.get("Discounted_Price", 0), errors="coerce").median()
    df["cluster_name"] = df.apply(cluster_name, axis=1)
    df = df.drop(columns=["_median_price"])

    Path("models/clustering").mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "model": model, "features": feature_cols}, "models/clustering/kmeans_product.pkl")
    pd.DataFrame(scores).to_csv("models/clustering/product_cluster_k_scores.csv", index=False)
    df.to_csv("data-project/processed/products_clustered.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data-project/processed/products_clean.csv"))
    args = parser.parse_args()
    run(args.input)
