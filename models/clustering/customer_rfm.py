"""Build customer RFM segments and churn risk flags."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


SEED = 42


def run(input_path: Path) -> None:
    df = pd.read_csv(input_path)
    required = {"User_ID", "Purchase_Date", "Transaction_ID", "Amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Purchase history missing columns: {sorted(missing)}")
    df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"], errors="coerce")
    snapshot = df["Purchase_Date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("User_ID").agg(
        Recency=("Purchase_Date", lambda s: (snapshot - s.max()).days),
        Frequency=("Transaction_ID", "nunique"),
        Monetary=("Amount", "sum"),
    )
    if {"Frequency_Previous", "Monetary_Previous"}.issubset(df.columns):
        previous = df.groupby("User_ID").agg(
            Frequency_Previous=("Frequency_Previous", "mean"),
            Monetary_Previous=("Monetary_Previous", "mean"),
        )
        rfm = rfm.join(previous, how="left")
    else:
        rfm["Frequency_Previous"] = rfm["Frequency"]
        rfm["Monetary_Previous"] = rfm["Monetary"]
    features = ["Recency", "Frequency", "Monetary"]
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[features])
    k = min(4, len(rfm))
    model = KMeans(n_clusters=k, random_state=SEED, n_init=20)
    rfm["rfm_segment"] = model.fit_predict(X)
    freq_decline = rfm["Frequency"] < rfm["Frequency_Previous"]
    monetary_decline = rfm["Monetary"] < rfm["Monetary_Previous"]
    risk = (
        0.5 * np.clip(rfm["Recency"] / 60, 0, 1)
        + 0.25 * freq_decline.astype(float)
        + 0.25 * monetary_decline.astype(float)
    )
    rfm["at_risk"] = (rfm["Recency"] > 30) & freq_decline & monetary_decline
    rfm["churn_probability"] = np.where(rfm["at_risk"], np.maximum(risk, 0.7), risk).clip(0, 1)
    rfm.reset_index().to_csv("data-project/processed/customers_rfm.csv", index=False)
    joblib.dump({"scaler": scaler, "model": model, "features": features}, "models/clustering/kmeans_rfm.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data-project/raw/purchases.csv"))
    args = parser.parse_args()
    Path("models/clustering").mkdir(parents=True, exist_ok=True)
    run(args.input)
