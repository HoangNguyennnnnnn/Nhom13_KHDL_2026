"""SO SÁNH các phương pháp dự báo doanh số (baseline -> XGBoost -> ablation).
5-fold KFold, cùng split. Output -> _compare_forecast.txt
"""
import sys, numpy as np, pandas as pd, joblib
from pathlib import Path
sys.path.insert(0, "models/forecasting")
from train_xgboost import build_dataset, FEATURES  # reuse exact dataset logic
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

SEED = 42
df = build_dataset(Path("data-project/processed/products_clean.csv"),
                   Path("data-project/processed/reviews_scored.csv"))
feats = [c for c in FEATURES if c in df.columns]
X, y = df[feats], df["target"]
print(f"n={len(df)} features={feats}")

# tuned params from the shipped model
try:
    tuned = joblib.load("models/forecasting/xgboost_model.pkl")["model"].get_params()
    tuned_params = {k: tuned[k] for k in ["n_estimators","max_depth","learning_rate","subsample","colsample_bytree","reg_lambda"] if k in tuned}
except Exception as e:
    tuned_params = {}
    print("could not load tuned params:", e)

def xgb(**kw): return XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=SEED, n_jobs=-1, **kw)

VARIANTS = [
    ("0. Baseline (đoán trung bình)", lambda: DummyRegressor(strategy="mean"), feats),
    ("1. Linear Regression", lambda: LinearRegression(), feats),
    ("2. Random Forest", lambda: RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1), feats),
    ("3. XGBoost (mặc định)", lambda: xgb(), feats),
    ("4. ⭐ XGBoost + Optuna (FINAL)", lambda: xgb(**tuned_params), feats),
    ("5. XGBoost+Optuna (KHÔNG có Sentiment)", lambda: xgb(**tuned_params), [f for f in feats if f != "Sentiment_Score"]),
]

kf = KFold(5, shuffle=True, random_state=SEED)
rows = []
for name, factory, fcols in VARIANTS:
    Xv = df[fcols]
    rmse, mae, r2 = [], [], []
    for tr, va in kf.split(Xv):
        m = factory(); m.fit(Xv.iloc[tr], y.iloc[tr]); p = m.predict(Xv.iloc[va])
        rmse.append(np.sqrt(mean_squared_error(y.iloc[va], p)))
        mae.append(mean_absolute_error(y.iloc[va], p))
        r2.append(r2_score(y.iloc[va], p) if len(va) >= 2 else 0.0)
    rows.append({"Phương pháp": name, "RMSE": int(np.mean(rmse)),
                 "MAE": int(np.mean(mae)), "R2": round(float(np.mean(r2)), 4)})
    print("done:", name)

res = pd.DataFrame(rows)
out = "==== SO SÁNH PHƯƠNG PHÁP DỰ BÁO DOANH SỐ (5-fold CV, n=%d) ====\n\n" % len(df)
out += res.to_string(index=False)
out += "\n\nTuned params (Optuna): " + str(tuned_params)
Path("_compare_forecast.txt").write_text(out, encoding="utf-8")
print("\n-> _compare_forecast.txt")
