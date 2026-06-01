"""Streamlit demo for product clusters and sentiment inference."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="TGDD DS Demo", layout="wide")
st.title("Customer Sentiment & Smartphone Sales Forecasting")

cluster_tab, sentiment_tab = st.tabs(["Product Cluster Explorer", "Sentiment Demo"])

with cluster_tab:
    path = Path("data-project/processed/products_clustered.csv")
    if not path.exists():
        st.warning("Run product clustering first to create data-project/processed/products_clustered.csv")
    else:
        df = pd.read_csv(path)
        # Dynamically grab any available numeric columns for PCA representation
        numeric_cols = df.select_dtypes(include=["number"]).columns
        features = [col for col in ["RAM", "ROM", "Battery", "Camera_MP", "Discounted_Price", "Original_Price", "Sales_Volume", "Avg_Star_Rating", "Total_Reviews", "Discount_Rate"] if col in numeric_cols]
        if len(features) < 2:
            st.warning("Need at least two numeric product features for PCA.")
        else:
            X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
            coords = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(X))
            plot_df = df.assign(PC1=coords[:, 0], PC2=coords[:, 1])
            fig = px.scatter(
                plot_df,
                x="PC1",
                y="PC2",
                color="cluster_name" if "cluster_name" in plot_df.columns else "cluster_id",
                hover_data=[col for col in ["Product_ID", "Brand", "Discounted_Price"] if col in plot_df.columns],
            )
            st.plotly_chart(fig, use_container_width=True)

with sentiment_tab:
    vectorizer_path = Path("models/sentiment/tfidf_vectorizer.pkl")
    model_path = Path("models/sentiment/rf_classifier.pkl")
    text = st.text_area("Nhập đánh giá", height=140)
    if st.button("Predict", type="primary"):
        if not vectorizer_path.exists() or not model_path.exists():
            st.error("Train TF-IDF + Random Forest first.")
        elif not text.strip():
            st.warning("Please enter review text.")
        else:
            vectorizer = joblib.load(vectorizer_path)
            model = joblib.load(model_path)
            X = vectorizer.transform([text])
            score = float(model.predict_proba(X)[0][list(model.classes_).index(1)])
            st.metric("Label", "Positive" if score >= 0.5 else "Negative")
            st.progress(score, text=f"Confidence: {score:.2%}")
