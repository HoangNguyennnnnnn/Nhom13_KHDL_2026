"""Train a CPU-friendly Vietnamese sentiment classifier.

Upgrade notes (still 100% CPU, no GPU required):
  - TF-IDF now combines *word* (1-2 gram) and *character* (3-5 gram) features.
    Character n-grams are robust to Vietnamese teencode / misspellings and
    usually lift accuracy noticeably over a word-only vectoriser.
  - The estimator is a calibrated Logistic Regression, which is faster and
    typically more accurate than a Random Forest on sparse TF-IDF text while
    still exposing ``predict_proba`` / ``classes_`` for the API and dashboards.
  - Cross-validation folds adapt to the data so tiny datasets no longer crash.

The artefacts keep their original filenames (``tfidf_vectorizer.pkl`` and
``rf_classifier.pkl``) so the FastAPI backend and Streamlit/Next.js apps load
them without any change.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


SEED = 42
ASPECT_KEYWORDS = {
    "Pin": ["pin", "sạc", "mah", "trâu"],
    "Màn_hình": ["màn_hình", "màn", "display", "sáng"],
    "Hiệu_năng": ["hiệu_năng", "lag", "mượt", "chip", "ram"],
    "Camera": ["camera", "ảnh", "chụp", "quay"],
    "Giá_cả": ["giá", "rẻ", "đắt", "tiền"],
}


def build_vectorizer() -> TfidfVectorizer:
    """Word-level TF-IDF vectorizer using 1-2 grams."""
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
    )


def build_classifier() -> VotingClassifier:
    """VotingClassifier ensemble of Calibrated LinearSVC and Logistic Regression."""
    lr = LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=1000)
    svc = CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=SEED))
    return VotingClassifier(
        estimators=[
            ("lr", lr),
            ("svc", svc),
        ],
        voting="soft",
    )


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_col = (
        "Review_Text_Segmented"
        if "Review_Text_Segmented" in df.columns
        else ("Review_Text_Clean" if "Review_Text_Clean" in df.columns else "Review_Text")
    )
    if "label" not in df.columns:
        raise ValueError("reviews_clean.csv must contain a label column")
    df = df.dropna(subset=[text_col, "label"]).copy()
    df["label"] = df["label"].astype(int)
    df["_text"] = df[text_col].astype(str)

    # Drop empty texts that carry no signal.
    df = df[df["_text"].str.strip().astype(bool)].reset_index(drop=True)

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


def _positive_index(classifier) -> int:
    classes = list(classifier.classes_)
    if 1 in classes:
        return classes.index(1)
    if "1" in classes:
        return classes.index("1")
    return len(classes) - 1


def train(path: Path) -> None:
    df = load_data(path).reset_index(drop=True)

    class_counts = df["label"].value_counts()
    if len(class_counts) < 2:
        raise ValueError(
            "Need at least two sentiment classes to train. "
            f"Found only: {class_counts.to_dict()}"
        )

    X_text = df["_text"]
    y = df["label"]
    weights = df["weight"]

    # Adapt the number of folds so rare classes never break StratifiedKFold.
    min_class = int(class_counts.min())
    n_splits = max(2, min(5, min_class))

    rows = []
    if min_class >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        for fold, (train_idx, valid_idx) in enumerate(skf.split(X_text, y), start=1):
            vec = build_vectorizer()
            clf = build_classifier()
            X_train_vec = vec.fit_transform(X_text.iloc[train_idx])
            X_val_vec = vec.transform(X_text.iloc[valid_idx])
            clf.fit(X_train_vec, y.iloc[train_idx], sample_weight=weights.iloc[train_idx].values)
            pred = clf.predict(X_val_vec)
            rows.append(
                {
                    "fold": fold,
                    "f1": f1_score(y.iloc[valid_idx], pred, zero_division=0),
                    "precision": precision_score(y.iloc[valid_idx], pred, zero_division=0),
                    "recall": recall_score(y.iloc[valid_idx], pred, zero_division=0),
                }
            )
    else:
        print("[warn] A class has a single sample; skipping cross-validation.")

    # Final fit on all data
    vectorizer = build_vectorizer()
    classifier = build_classifier()
    X_all_vec = vectorizer.fit_transform(X_text)
    classifier.fit(X_all_vec, y, sample_weight=weights.values)

    probs = classifier.predict_proba(X_all_vec)[:, _positive_index(classifier)]
    df["Sentiment_Score"] = probs
    df["Sentiment_Label"] = np.where(probs >= 0.5, 1, 0)
    aspect_rows = [score_aspects(text, score) for text, score in zip(df["_text"], probs)]
    df = pd.concat([df.drop(columns=["_text"]), pd.DataFrame(aspect_rows)], axis=1)

    out_dir = Path("models/sentiment")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.pkl")
    joblib.dump(classifier, out_dir / "rf_classifier.pkl")
    pd.DataFrame(rows).to_csv(out_dir / "tfidf_rf_metrics.csv", index=False)
    Path("data-project/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data-project/processed/reviews_scored.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data-project/processed/reviews_clean.csv"))
    args = parser.parse_args()
    train(args.input)
