"""Train a CPU-friendly TF-IDF + Random Forest sentiment baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


SEED = 42
ASPECT_KEYWORDS = {
    "Pin": ["pin", "sạc", "mah", "trâu"],
    "Màn_hình": ["màn_hình", "màn", "display", "sáng"],
    "Hiệu_năng": ["hiệu_năng", "lag", "mượt", "chip", "ram"],
    "Camera": ["camera", "ảnh", "chụp", "quay"],
    "Giá_cả": ["giá", "rẻ", "đắt", "tiền"],
}


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_col = "Review_Text_Segmented" if "Review_Text_Segmented" in df.columns else "Review_Text"
    if "label" not in df.columns:
        raise ValueError("reviews_clean.csv must contain a label column")
    df = df.dropna(subset=[text_col, "label"]).copy()
    df["label"] = df["label"].astype(int)
    df["_text"] = df[text_col].astype(str)
    
    # Extract helpfulness weight
    if "Helpfulness_Count" in df.columns:
        df["weight"] = pd.to_numeric(df["Helpfulness_Count"], errors="coerce").fillna(0).astype(int) + 1
    else:
        df["weight"] = 1
    return df


def score_aspects(text: str, sentiment_score: float) -> dict[str, float]:
    lower = text.lower()
    scores = {}
    for aspect, words in ASPECT_KEYWORDS.items():
        mentioned = any(word in lower for word in words)
        scores[aspect] = float(sentiment_score if mentioned else 0.5)
    return scores


def train(path: Path) -> None:
    df = load_data(path).reset_index(drop=True)
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            ("rf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, class_weight="balanced")),
        ]
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    
    X_text = df["_text"]
    y = df["label"]
    weights = df["weight"]
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_text, y), start=1):
        # We manually vectorise first to pass sample_weight to the classifier
        tfidf = pipeline.named_steps["tfidf"]
        rf = pipeline.named_steps["rf"]
        
        X_train_vec = tfidf.fit_transform(X_text.iloc[train_idx])
        X_val_vec = tfidf.transform(X_text.iloc[valid_idx])
        
        rf.fit(X_train_vec, y.iloc[train_idx], sample_weight=weights.iloc[train_idx].values)
        pred = rf.predict(X_val_vec)
        
        rows.append(
            {
                "fold": fold,
                "f1": f1_score(y.iloc[valid_idx], pred, zero_division=0),
                "precision": precision_score(y.iloc[valid_idx], pred, zero_division=0),
                "recall": recall_score(y.iloc[valid_idx], pred, zero_division=0),
            }
        )
        
    # Final fit on all data
    tfidf = pipeline.named_steps["tfidf"]
    rf = pipeline.named_steps["rf"]
    X_all_vec = tfidf.fit_transform(X_text)
    rf.fit(X_all_vec, y, sample_weight=weights.values)
    
    probs = rf.predict_proba(X_all_vec)[:, list(rf.classes_).index(1)]
    df["Sentiment_Score"] = probs
    df["Sentiment_Label"] = np.where(probs >= 0.5, 1, 0)
    aspect_rows = [score_aspects(text, score) for text, score in zip(df["_text"], probs)]
    df = pd.concat([df.drop(columns=["_text"]), pd.DataFrame(aspect_rows)], axis=1)

    out_dir = Path("models/sentiment")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(tfidf, out_dir / "tfidf_vectorizer.pkl")
    joblib.dump(rf, out_dir / "rf_classifier.pkl")
    pd.DataFrame(rows).to_csv(out_dir / "tfidf_rf_metrics.csv", index=False)
    df.to_csv("data-project/processed/reviews_scored.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data-project/processed/reviews_clean.csv"))
    args = parser.parse_args()
    train(args.input)
