"""Streamlit dashboard — interactive exploration of recommendations + experiments.

Tabs:
    1. Recommend   — query the live API for a given user id
    2. Compare     — side-by-side outputs from two models
    3. Metrics     — MLflow-sourced comparison of offline metrics
    4. Attention   — per-user DIN attention heatmap
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")
MLFLOW_URL = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
ROOT = Path(__file__).resolve().parents[3]


def _get_json(url: str) -> dict:
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    return r.json()


def _load_ablation_summary() -> pd.DataFrame:
    path = ROOT / "experiments" / "ablations" / "significance.json"
    if not path.exists():
        return pd.DataFrame()
    data = json.loads(path.read_text())
    rows = []
    for name, value in data.get("point_estimates", {}).items():
        ci = data.get("bootstrap_ci_95", {}).get(name, [None, None])
        rows.append({"model": name.upper(), "Recall@10": value, "CI low": ci[0], "CI high": ci[1]})
    return pd.DataFrame(rows).sort_values("Recall@10", ascending=False)


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - depends on optional UI dep
        raise RuntimeError(
            "streamlit is required for the dashboard. Install with "
            "`pip install streamlit` or run the Docker dashboard service."
        ) from exc

    st.set_page_config(page_title="NeoRec Dashboard", layout="wide", page_icon="🎬")
    st.title("NeoRec — Recommender System Dashboard")
    st.caption(f"API: `{API_URL}` · MLflow: `{MLFLOW_URL}`")

    tab_recommend, tab_compare, tab_metrics, tab_attention = st.tabs(
        ["Recommend", "Compare models", "Metrics", "DIN attention"]
    )

    with tab_recommend:
        st.subheader("Live recommendation")
        col1, col2, col3 = st.columns([1, 1, 2])
        user_id = col1.number_input("user_id", min_value=0, value=1, step=1)
        k = col2.slider("top-K", min_value=5, max_value=20, value=10)
        diversity = col3.slider("MMR λ ()", 0.0, 1.0, 0.7, 0.05)
        if st.button("Recommend", type="primary"):
            try:
                data = _get_json(f"{API_URL}/recommend/{int(user_id)}?k={k}&diversity={diversity}")
                st.metric("Total latency (ms)", f"{data['latency_ms'].get('total', 0):.1f}")
                st.bar_chart(pd.Series(data["latency_ms"], name="latency_ms"))
                st.dataframe(pd.DataFrame(data["items"]), use_container_width=True)
            except Exception as exc:
                st.error(f"API request failed: {exc}")

    with tab_compare:
        st.subheader("Model A vs Model B")
        st.caption(" DeepFM → DIN → MMR MMR λ ")
        user_id_cmp = st.number_input("compare user_id", min_value=0, value=1, step=1)
        left, right = st.columns(2)
        for lam, col in [(0.3, left), (0.9, right)]:
            with col:
                st.markdown(f"**MMR λ={lam}**")
                try:
                    data = _get_json(f"{API_URL}/recommend/{int(user_id_cmp)}?k=10&diversity={lam}")
                    st.dataframe(pd.DataFrame(data["items"])[["item_id", "title", "score"]])
                except Exception as exc:
                    st.warning(str(exc))

    with tab_metrics:
        st.subheader("Offline metrics (from MLflow)")
        df = _load_ablation_summary()
        if df.empty:
            st.info("No significance.json found yet. Run `python scripts/build_significance.py`.")
        else:
            st.bar_chart(df.set_index("model")["Recall@10"])
            st.dataframe(df, use_container_width=True)

    with tab_attention:
        st.subheader("DIN attention heatmap")
        fig = ROOT / "experiments" / "results" / "ranking" / "din_attention_heatmap.png"
        if fig.exists():
            st.image(str(fig), caption="DIN attention heatmap")
        else:
            st.info("Attention heatmap not found. Run `python scripts/build_din_attention.py`.")


if __name__ == "__main__":
    main()
