"""Train the production 3-class sentiment model (TF-IDF + LogReg).

Review sentiment is strongly lexical ("tốt", "tệ", "lỗi") and TF-IDF bigrams
already capture common negations ("không tốt"), so the model is TF-IDF (1-2 gram)
+ LogisticRegression trained on the cleaned, 3-class data. CPU-only, no GPU needed.

Run prepare_sentiment.py first to produce reviews_labeled.csv.

Artifacts (loaded by api/main.py):
  sentiment_clf.pkl   - Pipeline(TfidfVectorizer, LogisticRegression), 3-class
  label_names.json    - {0: negative, 1: neutral, 2: positive}
  sentiment_metrics.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

SEED = 42
OUT_DIR = Path("models/sentiment")
DATA_PATH = Path("data-project/processed/reviews_labeled.csv")
TEXT_COL = "Review_Text_Final"  # segmented + negation-tagged
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}

# Smoke-test sentences (incl. negation / contrast) to prove the data fix works.
SMOKE_TESTS = [
    "máy quá tệ pin tụt nhanh dùng một ngày là hỏng",
    "sản phẩm không tốt chút nào chất lượng kém",
    "tưởng ngon ai ngờ dở tệ thất vọng",
    "giao hàng chậm nhân viên thái độ khó chịu",
    "máy đẹp pin trâu chụp ảnh nét rất hài lòng",
    "tạm ổn không có gì đặc biệt",
]


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced", max_iter=2000, random_state=SEED
                ),
            ),
        ]
    )


def load_dataset() -> tuple[list[str], np.ndarray]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run preprocessing/prepare_sentiment.py first."
        )
    df = pd.read_csv(DATA_PATH).dropna(subset=[TEXT_COL, "label"]).reset_index(drop=True)
    df = df[df[TEXT_COL].astype(str).str.strip().astype(bool)].reset_index(drop=True)
    return df[TEXT_COL].astype(str).tolist(), df["label"].astype(int).to_numpy()


def train() -> None:
    texts, y = load_dataset()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows = []
    for fold, (tr, va) in enumerate(skf.split(texts, y), start=1):
        m = build_model().fit([texts[i] for i in tr], y[tr])
        pred = m.predict([texts[i] for i in va])
        rows.append(
            {
                "fold": fold,
                "f1_macro": f1_score(y[va], pred, average="macro", zero_division=0),
                "precision_macro": precision_score(y[va], pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y[va], pred, average="macro", zero_division=0),
            }
        )
    metrics = pd.DataFrame(rows)
    print("[train] 5-fold CV (macro):")
    print(metrics.to_string(index=False))
    print(f"[train] mean F1-macro: {metrics['f1_macro'].mean():.4f}")

    oof = cross_val_predict(build_model(), texts, y, cv=skf)
    print("\n[train] out-of-fold report:")
    print(classification_report(y, oof, target_names=list(LABEL_NAMES.values()), zero_division=0))

    final = build_model().fit(texts, y)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, OUT_DIR / "sentiment_clf.pkl")
    (OUT_DIR / "label_names.json").write_text(
        json.dumps(LABEL_NAMES, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics.to_csv(OUT_DIR / "sentiment_metrics.csv", index=False)
    print(f"\n[train] saved -> {OUT_DIR / 'sentiment_clf.pkl'}")

    # Smoke test on the cleaned/segmented inputs.
    import sys

    sys.path.insert(0, str(Path("preprocessing").resolve()))
    from prepare_sentiment import apply_negation_tagging, strip_system_noise  # noqa: E402
    from preprocess import clean_text, segment_vi  # noqa: E402

    teencode_path = Path("data-project/teencode_dict.json")
    teencode = json.loads(teencode_path.read_text(encoding="utf-8")) if teencode_path.exists() else {}
    print("\n[train] smoke test:")
    for raw in SMOKE_TESTS:
        seg = apply_negation_tagging(segment_vi(clean_text(strip_system_noise(raw), teencode)))
        probs = final.predict_proba([seg])[0]
        pred = int(np.argmax(probs))
        print(f"  {LABEL_NAMES[pred]:8s} ({probs[pred]:.2f})  <- {raw}")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    train()
