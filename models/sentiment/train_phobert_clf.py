"""Train the student-owned 3-class sentiment classifier on PhoBERT features.

Pipeline position (data-mining stages [5]-[7]):
  [5] Feature  : PhoBERT (frozen) -> 768-dim embedding per review  (cached to .npy)
  [6] Model    : StandardScaler + LogisticRegression, trained by *us*  -> phobert_clf.pkl
  [7] Evaluate : StratifiedKFold (F1/precision/recall macro), out-of-fold confusion
                 matrix, and a head-to-head comparison against the old TF-IDF model.

Everything runs on CPU; Apple MPS is used automatically for the embedding pass if
available. The classifier itself trains in seconds because it sees dense vectors.

Artifacts written to ``models/sentiment/``:
  phobert_clf.pkl          - the trained classifier pipeline (the model you own)
  label_names.json         - {0: negative, 1: neutral, 2: positive}
  review_embeddings.npy    - cached PhoBERT features (skip re-embedding next time)
  phobert_metrics.csv      - per-fold macro metrics
  confusion_matrix.png     - out-of-fold confusion matrix
  model_comparison.csv     - PhoBERT-emb vs TF-IDF on identical folds
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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phobert_embedder import PhoBERTEmbedder

SEED = 42
OUT_DIR = Path("models/sentiment")
DATA_PATH = Path("data-project/processed/reviews_labeled.csv")
EMB_PATH = OUT_DIR / "review_embeddings.npy"
TEXT_COL = "Review_Text_Segmented"
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def build_classifier() -> Pipeline:
    """The model we own: standardize PhoBERT vectors then multinomial LogReg."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def load_dataset() -> tuple[pd.Series, np.ndarray]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run preprocessing/prepare_sentiment.py first."
        )
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TEXT_COL, "label"]).reset_index(drop=True)
    df = df[df[TEXT_COL].astype(str).str.strip().astype(bool)].reset_index(drop=True)
    return df[TEXT_COL].astype(str), df["label"].astype(int).to_numpy()


def get_embeddings(texts: pd.Series, use_cache: bool = True) -> np.ndarray:
    """Embed reviews with frozen PhoBERT, caching the result to disk."""
    if use_cache and EMB_PATH.exists():
        cached = np.load(EMB_PATH)
        if cached.shape[0] == len(texts):
            print(f"[train] using cached embeddings {cached.shape} from {EMB_PATH}")
            return cached
        print("[train] cache size mismatch -> re-embedding")
    embedder = PhoBERTEmbedder()
    print(f"[train] embedding {len(texts)} reviews on device='{embedder.device}' ...")
    emb = embedder.encode(texts.tolist(), batch_size=32, show_progress=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_PATH, emb)
    print(f"[train] saved embeddings {emb.shape} -> {EMB_PATH}")
    return emb


def cross_validate(X: np.ndarray, y: np.ndarray, build_fn, splitter) -> dict:
    """Return per-fold macro metrics and out-of-fold predictions."""
    rows = []
    for fold, (tr, va) in enumerate(splitter.split(X, y), start=1):
        model = build_fn()
        model.fit(X[tr], y[tr]) if isinstance(X, np.ndarray) else model.fit(
            [X[i] for i in tr], y[tr]
        )
        pred = model.predict(X[va]) if isinstance(X, np.ndarray) else model.predict(
            [X[i] for i in va]
        )
        rows.append(
            {
                "fold": fold,
                "f1_macro": f1_score(y[va], pred, average="macro", zero_division=0),
                "precision_macro": precision_score(
                    y[va], pred, average="macro", zero_division=0
                ),
                "recall_macro": recall_score(
                    y[va], pred, average="macro", zero_division=0
                ),
            }
        )
    return {"per_fold": pd.DataFrame(rows)}


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    """Save an out-of-fold confusion matrix as a PNG (best-effort)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        labels = [0, 1, 2]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(
            cm, display_labels=[LABEL_NAMES[i] for i in labels]
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("PhoBERT + LogReg — Out-of-fold Confusion Matrix")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[train] confusion matrix -> {path}")
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"[train] skipped confusion plot ({exc})")


def train(use_cache: bool = True) -> None:
    texts, y = load_dataset()
    X = get_embeddings(texts, use_cache=use_cache)

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # [7] Evaluate the PhoBERT-embedding model -------------------------------
    pho_cv = cross_validate(X, y, build_classifier, splitter)
    pho_oof = cross_val_predict(build_classifier(), X, y, cv=splitter)

    print("\n[train] PhoBERT-embedding CV (macro):")
    print(pho_cv["per_fold"].to_string(index=False))
    print("\n[train] Out-of-fold classification report:")
    print(
        classification_report(
            y, pho_oof, target_names=[LABEL_NAMES[i] for i in [0, 1, 2]], zero_division=0
        )
    )

    # Baseline: old-style TF-IDF + LogReg on identical folds ------------------
    def build_tfidf():
        return Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced", max_iter=2000, random_state=SEED
                    ),
                ),
            ]
        )

    text_list = texts.tolist()
    tfidf_rows = []
    for fold, (tr, va) in enumerate(splitter.split(text_list, y), start=1):
        m = build_tfidf()
        m.fit([text_list[i] for i in tr], y[tr])
        pred = m.predict([text_list[i] for i in va])
        tfidf_rows.append(f1_score(y[va], pred, average="macro", zero_division=0))

    pho_f1 = pho_cv["per_fold"]["f1_macro"].mean()
    tfidf_f1 = float(np.mean(tfidf_rows))
    comparison = pd.DataFrame(
        [
            {"model": "TF-IDF + LogReg (baseline)", "f1_macro_cv": round(tfidf_f1, 4)},
            {"model": "PhoBERT-emb + LogReg (ours)", "f1_macro_cv": round(pho_f1, 4)},
        ]
    )
    print("\n[train] Model comparison (5-fold F1-macro):")
    print(comparison.to_string(index=False))
    print(
        f"[train] improvement: {(pho_f1 - tfidf_f1) * 100:+.1f} F1-macro points "
        f"over TF-IDF baseline"
    )

    # [6] Fit the final model on ALL data and persist ------------------------
    final = build_classifier()
    final.fit(X, y)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, OUT_DIR / "phobert_clf.pkl")
    (OUT_DIR / "label_names.json").write_text(
        json.dumps(LABEL_NAMES, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pho_cv["per_fold"].to_csv(OUT_DIR / "phobert_metrics.csv", index=False)
    comparison.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    plot_confusion(y, pho_oof, OUT_DIR / "confusion_matrix.png")
    print(f"\n[train] saved model -> {OUT_DIR / 'phobert_clf.pkl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Recompute PhoBERT embeddings instead of using the cached .npy.",
    )
    args = parser.parse_args()
    train(use_cache=not args.no_cache)
