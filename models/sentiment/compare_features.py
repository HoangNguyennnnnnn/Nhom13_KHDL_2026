"""Quick feature-set comparison using the cached PhoBERT embeddings.

Data-mining stage [5]/[7]: empirically compare three feature representations on
identical folds so the choice is evidence-based, not assumed:

  1. TF-IDF only          (lexical / bag-of-words)
  2. PhoBERT only         (contextual sentence embedding, frozen)
  3. Hybrid TF-IDF+PhoBERT (lexical + semantic, concatenated)

The classifier is the same LogisticRegression in all three so only the features
differ. Re-uses the cached review_embeddings.npy -> runs in a few seconds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT_DIR = Path("models/sentiment")
DATA_PATH = Path("data-project/processed/reviews_labeled.csv")
EMB_PATH = OUT_DIR / "review_embeddings.npy"
TEXT_COL = "Review_Text_Segmented"


def main() -> None:
    df = pd.read_csv(DATA_PATH).dropna(subset=[TEXT_COL, "label"]).reset_index(drop=True)
    df = df[df[TEXT_COL].astype(str).str.strip().astype(bool)].reset_index(drop=True)
    texts = df[TEXT_COL].astype(str).tolist()
    y = df["label"].astype(int).to_numpy()
    emb = np.load(EMB_PATH)
    assert emb.shape[0] == len(texts), "embedding/text count mismatch"

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def clf():
        return LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=SEED
        )

    results = {"tfidf": [], "phobert": [], "hybrid": []}
    for tr, va in skf.split(texts, y):
        # Fit TF-IDF on the training fold only (no leakage).
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        Xtr_tf = vec.fit_transform([texts[i] for i in tr])
        Xva_tf = vec.transform([texts[i] for i in va])

        # Scale PhoBERT features on the training fold only.
        scaler = StandardScaler()
        Xtr_ph = scaler.fit_transform(emb[tr])
        Xva_ph = scaler.transform(emb[va])

        # 1) TF-IDF only
        m = clf().fit(Xtr_tf, y[tr])
        results["tfidf"].append(
            f1_score(y[va], m.predict(Xva_tf), average="macro", zero_division=0)
        )
        # 2) PhoBERT only
        m = clf().fit(Xtr_ph, y[tr])
        results["phobert"].append(
            f1_score(y[va], m.predict(Xva_ph), average="macro", zero_division=0)
        )
        # 3) Hybrid (concatenate sparse TF-IDF + dense PhoBERT)
        Xtr_hy = hstack([Xtr_tf, csr_matrix(Xtr_ph)]).tocsr()
        Xva_hy = hstack([Xva_tf, csr_matrix(Xva_ph)]).tocsr()
        m = clf().fit(Xtr_hy, y[tr])
        results["hybrid"].append(
            f1_score(y[va], m.predict(Xva_hy), average="macro", zero_division=0)
        )

    table = pd.DataFrame(
        [
            {"features": "TF-IDF only", "f1_macro_cv": np.mean(results["tfidf"])},
            {"features": "PhoBERT only", "f1_macro_cv": np.mean(results["phobert"])},
            {"features": "Hybrid TF-IDF+PhoBERT", "f1_macro_cv": np.mean(results["hybrid"])},
        ]
    ).sort_values("f1_macro_cv", ascending=False)
    table["f1_macro_cv"] = table["f1_macro_cv"].round(4)
    print(table.to_string(index=False))
    table.to_csv(OUT_DIR / "feature_comparison.csv", index=False)


if __name__ == "__main__":
    main()
