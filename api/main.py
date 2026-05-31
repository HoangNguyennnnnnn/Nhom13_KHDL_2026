"""FastAPI backend for sentiment, clusters, churn alerts, and sales forecasts."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


STATE: dict[str, Any] = {}


class ForecastRequest(BaseModel):
    product_id: str
    current_price: float = Field(gt=0)
    discount_rate: float = Field(ge=0, le=1)
    sentiment_score_yesterday: float = Field(ge=0, le=1)


def load_pickle(path: str):
    file = Path(path)
    return joblib.load(file) if file.exists() else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["forecast"] = load_pickle("models/forecasting/xgboost_model.pkl")
    STATE["products"] = pd.read_csv("data-project/processed/products_clustered.csv") if Path("data-project/processed/products_clustered.csv").exists() else pd.DataFrame()
    STATE["customers"] = pd.read_csv("data-project/processed/customers_rfm.csv") if Path("data-project/processed/customers_rfm.csv").exists() else pd.DataFrame()
    STATE["rules"] = pd.read_csv("data-project/processed/association_rules.csv") if Path("data-project/processed/association_rules.csv").exists() else pd.DataFrame()
    yield
    STATE.clear()


app = FastAPI(title="TGDD Customer Sentiment & Sales Forecasting API", version="1.0.0", lifespan=lifespan)


@app.post("/api/v1/forecast/sales")
def forecast_sales(body: ForecastRequest):
    bundle = STATE.get("forecast")
    if not bundle:
        raise HTTPException(status_code=503, detail="Forecast model is not trained")
    model = bundle["model"]
    features = bundle["features"]
    base = {
        "Discounted_Price": body.current_price,
        "Discount_Rate": body.discount_rate,
        "Avg_Star_Rating": 4.0,
        "Sentiment_Score": body.sentiment_score_yesterday,
        "Sales_Volume_7d_mean": 0,
    }
    rows = []
    previous = None
    for i in range(1, 8):
        day = date.today() + timedelta(days=i)
        data = {
            **base,
            "Day_of_Week": day.weekday(),
            "Is_Weekend": int(day.weekday() >= 5),
            "Is_Holiday": int(day.strftime("%m-%d") in {"01-01", "04-30", "05-01", "09-02"}),
        }
        X = pd.DataFrame([{col: data.get(col, 0) for col in features}])
        pred = max(0.0, float(model.predict(X)[0]))
        rows.append(
            {
                "date": day.isoformat(),
                "predicted_volume": round(pred, 2),
                "trend": "up" if previous is not None and pred > previous else "down" if previous is not None else "flat",
            }
        )
        previous = pred
        base["Sales_Volume_7d_mean"] = pred
    return rows


@app.get("/api/v1/customers/churn_alert")
def churn_alert():
    df = STATE.get("customers", pd.DataFrame())
    if df.empty:
        return []
    cols = [col for col in ["User_ID", "email", "churn_probability", "rfm_segment"] if col in df.columns]
    return df[df["churn_probability"] > 0.7][cols].to_dict(orient="records")


@app.get("/api/v1/products/clusters")
def product_clusters():
    df = STATE.get("products", pd.DataFrame())
    if df.empty:
        return []
    cols = [col for col in ["Product_ID", "Brand", "cluster_id", "cluster_name", "RAM", "ROM", "Battery", "Camera_MP", "Discounted_Price"] if col in df.columns]
    return df[cols].to_dict(orient="records")


@app.get("/api/v1/products/{product_id}/cross_sell")
def cross_sell(product_id: str):
    rules = STATE.get("rules", pd.DataFrame())
    if rules.empty:
        return []
    matched = rules[rules["antecedent"].astype(str).str.split(",").map(lambda items: product_id in items)]
    return [
        {
            "accessory_id": row["consequent"],
            "accessory_name": row["consequent"],
            "confidence": row["confidence"],
            "lift": row["lift"],
        }
        for _, row in matched.iterrows()
    ]
