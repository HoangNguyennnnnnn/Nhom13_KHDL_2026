"""SO SÁNH các phương pháp phân loại cảm xúc (baseline -> ablation -> final).
5-fold StratifiedKFold, cùng split, có sample_weight. Output -> _compare_sentiment.txt
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import underthesea

SEED = 42
df = pd.read_csv("data-project/processed/reviews_clean.csv")
df = df.dropna(subset=["Review_Text_Segmented", "label"]).copy()
df["label"] = df["label"].astype(int)
df["_t"] = df["Review_Text_Segmented"].astype(str)
df = df[df["_t"].str.strip().astype(bool)].reset_index(drop=True)
df["w"] = pd.to_numeric(df.get("Helpfulness_Count", 0), errors="coerce").fillna(0).astype(int) + 1
X, y, w = df["_t"], df["label"], df["w"]
print(f"n={len(df)} pos={int((y==1).sum())} neg={int((y==0).sum())}")

print("caching underthesea feature once...")
_uts = np.array([[1.0 if underthesea.sentiment(str(t)) == "positive"
                  else 0.0 if underthesea.sentiment(str(t)) == "negative" else 0.5] for t in X])
UTS = csr_matrix(_uts)

def word_vec(): return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2)
def char_vec(): return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=20000)
def ens():
    return VotingClassifier([
        ("lr", LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=1000)),
        ("svc", CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=SEED))),
    ], voting="soft")

# (feature_mode, estimator_factory, supports_sample_weight)
VARIANTS = [
    ("0. Baseline (đoán lớp đa số)", "none", lambda: DummyClassifier(strategy="most_frequent"), False),
    ("1. TF-IDF(word) + Naive Bayes", "word", lambda: MultinomialNB(), True),
    ("2. TF-IDF(word) + Random Forest", "word", lambda: RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1, class_weight="balanced"), True),
    ("3. TF-IDF(word) + Logistic Reg", "word", lambda: LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=1000), True),
    ("4. TF-IDF(word) + LinearSVC", "word", lambda: LinearSVC(class_weight="balanced", random_state=SEED), True),
    ("5. TF-IDF(word) + Ensemble", "word", ens, True),
    ("6. TF-IDF(word+char) + Ensemble", "wordchar", ens, True),
    ("7. ⭐ TF-IDF(word)+underthesea + Ensemble (FINAL)", "word+uts", ens, True),
]

def build(mode, tr, va):
    if mode == "none":
        return np.zeros((len(tr), 1)), np.zeros((len(va), 1))
    wv = word_vec(); Xtr = wv.fit_transform(X.iloc[tr]); Xva = wv.transform(X.iloc[va])
    if mode == "wordchar":
        cv = char_vec(); Xtr = hstack([Xtr, cv.fit_transform(X.iloc[tr])]); Xva = hstack([Xva, cv.transform(X.iloc[va])])
    if mode == "word+uts":
        Xtr = hstack([Xtr, UTS[tr]]); Xva = hstack([Xva, UTS[va]])
    return Xtr, Xva

skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows = []
for name, mode, factory, sw in VARIANTS:
    oof = np.zeros(len(df), dtype=int)
    for tr, va in skf.split(X, y):
        Xtr, Xva = build(mode, tr, va)
        m = factory()
        if sw:
            try: m.fit(Xtr, y.iloc[tr], sample_weight=w.iloc[tr].values)
            except TypeError: m.fit(Xtr, y.iloc[tr])
        else:
            m.fit(Xtr, y.iloc[tr])
        oof[va] = m.predict(Xva)
    rows.append({
        "Phương pháp": name,
        "Accuracy": round(accuracy_score(y, oof), 4),
        "F1_macro": round(f1_score(y, oof, average="macro"), 4),
        "F1_pos": round(f1_score(y, oof, pos_label=1, zero_division=0), 4),
        "Recall_pos": round(recall_score(y, oof, pos_label=1, zero_division=0), 4),
        "Recall_neg": round(recall_score(y, oof, pos_label=0, zero_division=0), 4),
    })
    print("done:", name)

res = pd.DataFrame(rows)
out = "==== SO SÁNH PHƯƠNG PHÁP PHÂN LOẠI CẢM XÚC (5-fold CV, n=%d) ====\n\n" % len(df)
out += res.to_string(index=False)
Path("_compare_sentiment.txt").write_text(out, encoding="utf-8")
print("\n-> _compare_sentiment.txt")
