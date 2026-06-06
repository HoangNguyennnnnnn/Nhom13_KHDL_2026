"""FastAPI backend for sentiment, clusters, churn alerts, and sales forecasts."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


STATE: dict[str, Any] = {}
DEFAULT_FORECAST_FEATURES = [
    "Discounted_Price",
    "Discount_Rate",
    "Avg_Star_Rating",
    "Sentiment_Score",
    "Day_of_Week",
    "Is_Weekend",
    "Is_Holiday",
    "Sales_Volume_7d_mean",
]


class ForecastRequest(BaseModel):
    product_id: str
    current_price: float = Field(gt=0)
    discount_rate: float = Field(ge=0, le=1)
    sentiment_score_yesterday: float = Field(ge=0, le=1)


class SentimentRequest(BaseModel):
    text: str


def load_pickle(path: str):
    file = Path(path)
    return joblib.load(file) if file.exists() else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["forecast"] = load_pickle("models/forecasting/xgboost_model.pkl")
    STATE["products"] = pd.read_csv("data-project/processed/products_clustered.csv") if Path("data-project/processed/products_clustered.csv").exists() else pd.DataFrame()
    STATE["customers"] = pd.read_csv("data-project/processed/customers_rfm.csv") if Path("data-project/processed/customers_rfm.csv").exists() else pd.DataFrame()
    STATE["rules"] = pd.read_csv("data-project/processed/association_rules.csv") if Path("data-project/processed/association_rules.csv").exists() else pd.DataFrame()
    STATE["tfidf"] = load_pickle("models/sentiment/tfidf_vectorizer.pkl")
    STATE["sentiment_model"] = load_pickle("models/sentiment/rf_classifier.pkl")
    
    # Load teencode mapping
    teencode_path = Path("data-project/teencode_dict.json")
    import json
    STATE["teencode"] = json.loads(teencode_path.read_text(encoding="utf-8")) if teencode_path.exists() else {}
    yield
    STATE.clear()


app = FastAPI(title="TGDD Customer Sentiment & Sales Forecasting API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_pca_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Attach PC1/PC2 for plotting when processed file does not contain them yet."""
    out = df.copy()
    if "PC1" in out.columns and "PC2" in out.columns and out[["PC1", "PC2"]].notna().any().all():
        return out

    feature_cols = [
        c
        for c in ["RAM", "ROM", "Battery", "Camera_MP", "Discounted_Price", "Avg_Star_Rating", "Total_Reviews"]
        if c in out.columns
    ]
    if len(feature_cols) < 2 or len(out) < 2:
        out["PC1"] = 0.0
        out["PC2"] = 0.0
        return out

    X = out[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    X_scaled = StandardScaler().fit_transform(X)
    coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    out["PC1"] = coords[:, 0]
    out["PC2"] = coords[:, 1]
    return out


@app.post("/api/v1/forecast/sales")
def forecast_sales(body: ForecastRequest):
    bundle = STATE.get("forecast")
    if bundle is None:
        raise HTTPException(status_code=503, detail="Forecast model is not trained")
    if isinstance(bundle, dict):
        model = bundle.get("model")
        features = bundle.get("features") or DEFAULT_FORECAST_FEATURES
    else:
        model = bundle
        features = DEFAULT_FORECAST_FEATURES
    if model is None:
        raise HTTPException(status_code=503, detail="Forecast model is not trained")
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
    if df.empty or "churn_probability" not in df.columns:
        return []
    prob = pd.to_numeric(df["churn_probability"], errors="coerce").fillna(0)
    cols = [col for col in ["User_ID", "email", "churn_probability", "rfm_segment"] if col in df.columns]
    return df.loc[prob > 0.7, cols].to_dict(orient="records")


@app.get("/api/v1/products/clusters")
def product_clusters():
    df = STATE.get("products", pd.DataFrame())
    if df.empty:
        return []
    df = _ensure_pca_coords(df)
    cols = [
        col
        for col in [
            "Product_ID",
            "Name",
            "Brand",
            "cluster_id",
            "cluster_name",
            "RAM",
            "ROM",
            "Battery",
            "Camera_MP",
            "Discounted_Price",
            "Avg_Star_Rating",
            "Total_Reviews",
            "PC1",
            "PC2",
        ]
        if col in df.columns
    ]
    return df[cols].to_dict(orient="records")


@app.get("/api/v1/products/{product_id}/cross_sell")
def cross_sell(product_id: str):
    rules = STATE.get("rules", pd.DataFrame())
    required = {"antecedent", "consequent", "confidence", "lift"}
    if rules.empty or not required.issubset(rules.columns):
        return []
    matched = rules[rules["antecedent"].astype(str).str.split(",").map(lambda items: product_id in [item.strip() for item in items])]
    return [
        {
            "accessory_id": row["consequent"],
            "accessory_name": row["consequent"],
            "confidence": row["confidence"],
            "lift": row["lift"],
        }
        for _, row in matched.iterrows()
    ]


@app.post("/api/v1/sentiment/predict")
def predict_sentiment(body: SentimentRequest):
    tfidf = STATE.get("tfidf")
    model = STATE.get("sentiment_model")
    teencode = STATE.get("teencode", {})
    if tfidf is None or model is None:
        raise HTTPException(status_code=503, detail="Sentiment analysis model is not loaded")
        
    text = str(body.text).lower()
    import re
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^0-9a-zA-ZÀ-ỹ\s]", " ", text)
    
    words = text.split()
    cleaned_words = [teencode.get(w, w) for w in words]
    cleaned_text = " ".join(cleaned_words)
    
    # Vectorize and predict. Prefer class 1 when available; otherwise fall back safely.
    X_vec = tfidf.transform([cleaned_text])
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_vec)[0]
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            prob = float(probabilities[classes.index(1)])
        elif "1" in classes:
            prob = float(probabilities[classes.index("1")])
        elif len(probabilities) == 2:
            prob = float(probabilities[-1])
        else:
            prob = float(max(probabilities))
    else:
        pred = model.predict(X_vec)[0]
        prob = 1.0 if str(pred) in {"1", "true", "positive", "pos"} else 0.0

    # Lexicon adjustment for extremely rare/unbalanced negative words in short reviews
    NEG_WORDS = {"đểu", "tệ", "chán", "kém", "lỗi", "hỏng", "yếu", "đơ", "lag", "giật", "chậm", "tệ_hại", "quá_tệ", "hỏng_hóc", "vứt"}
    POS_WORDS = {"tốt", "mượt", "ngon", "ưng", "đẹp", "thích", "nhanh", "ổn", "ok", "tuyệt", "chuẩn", "sướng"}
    
    words_set = set(cleaned_text.split())
    has_neg = any(w in words_set for w in NEG_WORDS)
    has_pos = any(w in words_set for w in POS_WORDS)
    
    if has_neg and not has_pos:
        prob = max(0.05, min(0.35, prob - 0.5))
    
    label = 1 if prob >= 0.5 else 0
    confidence = prob if label == 1 else (1.0 - prob)
    
    return {
        "label": label,
        "confidence": confidence,
        "cleaned_text": cleaned_text
    }
