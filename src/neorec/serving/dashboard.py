"""Streamlit dashboard — interactive exploration of recommendations + experiments.

Tabs:
    1. Recommend   — query the live API for a given user id
    2. Compare     — side-by-side outputs from two models
    3. Metrics     — MLflow-sourced comparison of offline metrics
    4. Attention   — per-user DIN attention heatmap
"""

from __future__ import annotations

import os

import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
MLFLOW_URL = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def main() -> None:
    st.set_page_config(page_title="NeoRec Dashboard", layout="wide", page_icon="🎬")
    st.title("NeoRec — Recommender System Dashboard")
    st.caption(f"API: `{API_URL}` · MLflow: `{MLFLOW_URL}`")

    tab_recommend, tab_compare, tab_metrics, tab_attention = st.tabs(
        ["Recommend", "Compare models", "Metrics", "DIN attention"]
    )

    with tab_recommend:
        st.subheader("Live recommendation")
        # TODO(W5 Day 33-34)
        st.info("TODO: form for user_id + requests.get(f'{API_URL}/recommend/...')")

    with tab_compare:
        st.subheader("Model A vs Model B")
        # TODO(W5 Day 33-34)
        st.info("TODO: side-by-side lists")

    with tab_metrics:
        st.subheader("Offline metrics (from MLflow)")
        # TODO(W5 Day 33-34)
        st.info("TODO: bar chart of Recall@10 / NDCG@10 per model")

    with tab_attention:
        st.subheader("DIN attention heatmap")
        # TODO(W5 Day 33-34)
        st.info("TODO: picker → heatmap (hist × target)")


if __name__ == "__main__":
    main()
