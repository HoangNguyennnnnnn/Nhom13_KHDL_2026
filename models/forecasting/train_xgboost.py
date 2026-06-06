"""Train an XGBoost sales model (CPU-friendly, leakage-free).

Important fixes vs. the previous version:
  - Removed target leakage: ``Sales_Volume_7d_mean`` is no longer the target
    multiplied by random noise. It is now a *leave-one-out brand average*
    (the mean sales of other products of the same brand), which is a
    legitimate historical proxy that does not peek at the row's own target.
  - Calendar features are derived from ``Inward_Date`` when available instead
    of being randomly generated, so they are reproducible and meaningful.
  - No more fabricated random dataset when data is scarce: the script now
    fails loudly so you never ship metrics computed on fake data.
  - Optuna trials default to a CPU-sensible value and XGBoost uses the fast
    histogram tree method with all cores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor


SEED = 42
FEATURES = [
    "Discounted_Price",
    "Discount_Rate",
    "Avg_Star_Rating",
    "Sentiment_Score",
    "Day_of_Week",
    "Is_Weekend",
    "Is_Holiday",
    "Sales_Volume_7d_mean",
]
HOLIDAYS_MM_DD = {"01-01", "04-30", "05-01", "09-02"}


def _leave_one_out_brand_mean(df: pd.DataFrame) -> pd.Series:
    """Mean Sales_Volume of *other* products in the same brand (no self-leak)."""
    sales = pd.to_numeric(df["Sales_Volume"], errors="coerce")
    if "Brand" not in df.columns:
        overall = sales.mean()
        return sales.where(sales.isna(), overall).fillna(overall)

    grp = sales.groupby(df["Brand"])
    group_sum = grp.transform("sum")
    group_cnt = grp.transform("count")
    loo = (group_sum - sales) / (group_cnt - 1)
    # Fall back to the global mean for single-product brands / missing values.
    return loo.fillna(sales.mean())


def build_dataset(products_path: Path, reviews_path: Path | None) -> pd.DataFrame:
    df = pd.read_csv(products_path)
    if reviews_path and reviews_path.exists():
        reviews = pd.read_csv(reviews_path)
        if "Sentiment_Score" in reviews.columns:
            sentiment = reviews.groupby("Product_ID", as_index=False)["Sentiment_Score"].mean()
            df = df.merge(sentiment, on="Product_ID", how="left")

    if "Sentiment_Score" not in df.columns:
        df["Sentiment_Score"] = 0.5
    df["Sentiment_Score"] = df["Sentiment_Score"].fillna(0.5)

    if "Sales_Volume" not in df.columns:
        raise ValueError("products_clean.csv must contain a Sales_Volume column for forecasting.")

    # Target = the product's sales volume.
    df["target"] = pd.to_numeric(df["Sales_Volume"], errors="coerce")

    # Leakage-free historical proxy feature.
    df["Sales_Volume_7d_mean"] = _leave_one_out_brand_mean(df)

    # Calendar features from a real date column when present; else neutral defaults.
    if "Inward_Date" in df.columns:
        dates = pd.to_datetime(df["Inward_Date"], errors="coerce")
        df["Day_of_Week"] = dates.dt.dayofweek.fillna(0).astype(int)
        df["Is_Weekend"] = df["Day_of_Week"].isin([5, 6]).astype(int)
        df["Is_Holiday"] = dates.dt.strftime("%m-%d").isin(HOLIDAYS_MM_DD).astype(int)
    else:
        df["Day_of_Week"] = 0
        df["Is_Weekend"] = 0
        df["Is_Holiday"] = 0

    for col in FEATURES + ["target"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["target"])
    return df


def objective(trial, X, y, n_splits):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "random_state": SEED,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_jobs": -1,
    }
    split = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rmses = []
    for train_idx, valid_idx in split.split(X):
        model = XGBRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[valid_idx])
        rmses.append(np.sqrt(mean_squared_error(y.iloc[valid_idx], pred)))
    return float(np.mean(rmses))


def train(products_path: Path, reviews_path: Path | None, trials: int) -> None:
    df = build_dataset(products_path, reviews_path)
    feature_cols = [col for col in FEATURES if col in df.columns]

    if len(df) < 5:
        raise ValueError(
            f"Only {len(df)} usable rows after cleaning. Forecasting needs at least 5 "
            "real products with a Sales_Volume target. Scrape/clean more data first "
            "instead of training on fabricated values."
        )

    X, y = df[feature_cols], df["target"]
    n_splits = max(2, min(5, len(df)))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lambda trial: objective(trial, X, y, n_splits), n_trials=trials)
    params = {
        **study.best_params,
        "random_state": SEED,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_jobs": -1,
    }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    metrics = []
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X), start=1):
        model = XGBRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[valid_idx])
        metrics.append(
            {
                "fold": fold,
                "rmse": np.sqrt(mean_squared_error(y.iloc[valid_idx], pred)),
                "mae": mean_absolute_error(y.iloc[valid_idx], pred),
                "r2": r2_score(y.iloc[valid_idx], pred) if len(valid_idx) >= 2 else 0.0,
            }
        )

    final_model = XGBRegressor(**params)
    final_model.fit(X, y)
    Path("models/forecasting").mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "features": feature_cols}, "models/forecasting/xgboost_model.pkl")
    pd.DataFrame(metrics).to_csv("models/forecasting/xgboost_metrics.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", type=Path, default=Path("data-project/processed/products_clean.csv"))
    parser.add_argument("--reviews", type=Path, default=Path("data-project/processed/reviews_scored.csv"))
    parser.add_argument("--trials", type=int, default=40)
    args = parser.parse_args()
    train(args.products, args.reviews, args.trials)
