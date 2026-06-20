"""One-off: compute dataset statistics for the README. Output -> _readme_stats.txt"""
import json
import pandas as pd
import numpy as np

out = []
def p(*a):
    out.append(" ".join(str(x) for x in a))

RAW = "data-project/raw"
PROC = "data-project/processed"

# ---- Products ----
prod = pd.read_csv(f"{RAW}/products.csv")
p("==== PRODUCTS (raw) ====")
p("rows:", len(prod), "cols:", list(prod.columns))
p("\nBrand distribution:")
p(prod["Brand"].value_counts().to_string())
for col in ["Original_Price", "Discounted_Price", "Sales_Volume", "Avg_Star_Rating", "Total_Reviews"]:
    s = pd.to_numeric(prod[col], errors="coerce")
    p(f"\n{col}: min={s.min():.0f} max={s.max():.0f} mean={s.mean():.0f} median={s.median():.0f} missing={s.isna().sum()}")

# ---- Reviews ----
rev = pd.read_csv(f"{RAW}/reviews.csv")
p("\n\n==== REVIEWS (raw) ====")
p("rows:", len(rev), "cols:", list(rev.columns))
p("unique products with reviews:", rev["Product_ID"].nunique())
stars = pd.to_numeric(rev["Star_Rating"], errors="coerce")
p("\nStar rating distribution:")
p(stars.value_counts().sort_index().to_string())
p("star missing:", stars.isna().sum())
help_c = pd.to_numeric(rev["Helpfulness_Count"], errors="coerce").fillna(0)
p("helpfulness: max=", help_c.max(), "mean=", round(help_c.mean(),2), "nonzero=", (help_c>0).sum())
sup = pd.to_numeric(rev["Support_Contacted"], errors="coerce").fillna(0)
p("support_contacted=1:", int((sup==1).sum()))
emptytxt = rev["Review_Text"].fillna("").str.strip().eq("").sum()
p("empty review text:", emptytxt)

# ---- reviews_clean (labels) ----
try:
    rc = pd.read_csv(f"{PROC}/reviews_clean.csv")
    p("\n\n==== REVIEWS_CLEAN ====")
    p("rows:", len(rc))
    if "label" in rc.columns:
        p("label distribution (1=pos,0=neg, NaN=neutral 3star):")
        p(rc["label"].value_counts(dropna=False).to_string())
except Exception as e:
    p("reviews_clean error", e)

# ---- reviews_scored ----
try:
    rs = pd.read_csv(f"{PROC}/reviews_scored.csv")
    p("\n\n==== REVIEWS_SCORED (training set) ====")
    p("rows used for training:", len(rs))
    if "label" in rs.columns:
        p("train label dist:", rs["label"].value_counts().to_dict())
    if "Sentiment_Score" in rs.columns:
        p("sentiment_score mean:", round(rs["Sentiment_Score"].mean(),3))
except Exception as e:
    p("reviews_scored error", e)

# ---- products_clustered ----
try:
    pc = pd.read_csv(f"{PROC}/products_clustered.csv")
    p("\n\n==== PRODUCTS_CLUSTERED ====")
    p("rows:", len(pc))
    if "cluster_name" in pc.columns:
        p("cluster_name distribution:")
        p(pc["cluster_name"].value_counts().to_string())
    if "cluster_id" in pc.columns:
        p("cluster_id distribution:", pc["cluster_id"].value_counts().sort_index().to_dict())
    for col in ["RAM","ROM","Battery","Camera_MP"]:
        if col in pc.columns:
            s = pd.to_numeric(pc[col], errors="coerce")
            p(f"{col}: min={s.min()} max={s.max()} mean={round(s.mean(),1)}")
except Exception as e:
    p("clustered error", e)

# ---- aspect sentiment ----
try:
    asp = json.load(open(f"{PROC}/aspect_sentiment.json", encoding="utf-8"))
    p("\n\n==== ASPECT_SENTIMENT ====")
    p("products analyzed:", len(asp))
    from collections import Counter, defaultdict
    aspect_count = Counter()
    aspect_scores = defaultdict(list)
    for pid, v in asp.items():
        for aname, adata in v.get("Aspects", {}).items():
            aspect_count[aname] += 1
            aspect_scores[aname].append(adata.get("Score", 0))
    p("aspect coverage (how many products mention each aspect):")
    for a, c in aspect_count.most_common():
        p(f"  {a}: mentioned in {c} products, avg score={round(np.mean(aspect_scores[a]),3)}")
except Exception as e:
    p("aspect error", e)

# ---- price elasticity ----
try:
    pe = json.load(open(f"{PROC}/price_elasticity.json", encoding="utf-8"))
    p("\n\n==== PRICE_ELASTICITY ====")
    p("brands modeled:", len(pe))
    for b, v in sorted(pe.items(), key=lambda kv: -kv[1]["Product_Count"]):
        p(f"  {b}: n={v['Product_Count']} avgPrice={v['Average_Original_Price']:.0f} "
          f"PED={v['Price_Elasticity']} R2={v['Model_R2']} avgSales={v['Average_Sales_Volume']:.0f}")
except Exception as e:
    p("elasticity error", e)

# ---- model metrics ----
p("\n\n==== MODEL METRICS ====")
try:
    m = pd.read_csv("models/sentiment/tfidf_rf_metrics.csv")
    p("Sentiment (5-fold CV): F1 mean={:.4f} std={:.4f} | precision mean={:.4f} | recall mean={:.4f}".format(
        m["f1"].mean(), m["f1"].std(), m["precision"].mean(), m["recall"].mean()))
except Exception as e:
    p("sent metrics err", e)
try:
    m = pd.read_csv("models/forecasting/xgboost_metrics.csv")
    p("Forecast (5-fold CV): RMSE mean={:.0f} | MAE mean={:.0f} | R2 mean={:.4f}".format(
        m["rmse"].mean(), m["mae"].mean(), m["r2"].mean()))
except Exception as e:
    p("fc metrics err", e)
try:
    m = pd.read_csv("models/clustering/product_cluster_k_scores.csv")
    best = m.loc[m["silhouette"].idxmax()]
    p(f"Clustering: best k={int(best['k'])} silhouette={best['silhouette']:.4f}")
except Exception as e:
    p("clu metrics err", e)

# ---- teencode dict size ----
try:
    tc = json.load(open(f"data-project/teencode_dict.json", encoding="utf-8"))
    p("\nTeencode dictionary entries:", len(tc))
    p("sample teencode:", dict(list(tc.items())[:12]))
except Exception as e:
    p("teencode err", e)

with open("_readme_stats.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("done -> _readme_stats.txt")
