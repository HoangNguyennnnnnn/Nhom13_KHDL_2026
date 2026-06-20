"""Compute OVERALL sentiment metrics (accuracy + per-class) via 5-fold CV.
Faithful to models/sentiment/train_tfidf_rf.py but caches the underthesea
feature once so CV is fast. Output -> _sentiment_eval.txt
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import underthesea

SEED = 42
df = pd.read_csv("data-project/processed/reviews_clean.csv")
col = "Review_Text_Segmented"
df = df.dropna(subset=[col, "label"]).copy()
df["label"] = df["label"].astype(int)
df["_t"] = df[col].astype(str)
df = df[df["_t"].str.strip().astype(bool)].reset_index(drop=True)
if "Helpfulness_Count" in df.columns:
    df["w"] = pd.to_numeric(df["Helpfulness_Count"], errors="coerce").fillna(0).astype(int) + 1
else:
    df["w"] = 1

X_text, y, w = df["_t"], df["label"], df["w"]
print(f"n={len(df)}  pos={int((y==1).sum())}  neg={int((y==0).sum())}")

# Cache underthesea sentiment feature ONCE
print("computing underthesea feature (once)...")
uts = np.array([[1.0 if underthesea.sentiment(str(t))=="positive"
                 else 0.0 if underthesea.sentiment(str(t))=="negative" else 0.5]
                for t in X_text])
# (note: calls twice per text above for clarity; fine for one-off eval)
uts = csr_matrix(uts)

def vec(): return TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=2)
def clf():
    lr = LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=1000)
    svc = CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=SEED))
    return VotingClassifier([("lr",lr),("svc",svc)], voting="soft")

skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
oof = np.zeros(len(df), dtype=int)
for tr, va in skf.split(X_text, y):
    v = vec(); Xtr = hstack([v.fit_transform(X_text.iloc[tr]), uts[tr]])
    Xva = hstack([v.transform(X_text.iloc[va]), uts[va]])
    c = clf(); c.fit(Xtr, y.iloc[tr], sample_weight=w.iloc[tr].values)
    oof[va] = c.predict(Xva)

out = []
out.append("==== SENTIMENT — OVERALL (out-of-fold, 5-fold CV) ====")
out.append(f"n={len(df)}  pos={int((y==1).sum())}  neg={int((y==0).sum())}")
out.append(f"Accuracy           : {accuracy_score(y, oof):.4f}")
out.append(f"F1 (positive)      : {f1_score(y, oof, pos_label=1):.4f}")
out.append(f"F1 (macro)         : {f1_score(y, oof, average='macro'):.4f}")
out.append(f"F1 (weighted)      : {f1_score(y, oof, average='weighted'):.4f}")
out.append(f"Precision (pos)    : {precision_score(y, oof, pos_label=1):.4f}")
out.append(f"Recall (pos)       : {recall_score(y, oof, pos_label=1):.4f}")
out.append(f"Precision (neg)    : {precision_score(y, oof, pos_label=0):.4f}")
out.append(f"Recall (neg)       : {recall_score(y, oof, pos_label=0):.4f}")
out.append("\nConfusion matrix [rows=true 0,1 | cols=pred 0,1]:")
out.append(str(confusion_matrix(y, oof)))
out.append("\n" + classification_report(y, oof, target_names=["Tiêu cực(0)","Tích cực(1)"], digits=4))
txt = "\n".join(out)
Path("_sentiment_eval.txt").write_text(txt, encoding="utf-8")
print("done -> _sentiment_eval.txt")
