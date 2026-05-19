"""Build README figures from cached ablation JSON results.

This script is intentionally data-only: it does not retrain models or touch
MLflow. It turns the already-recorded W4/W5 JSON artifacts into PNGs embedded
by README Section 8.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ABL = ROOT / "experiments" / "ablations"
FIG = ROOT / "experiments" / "results" / "figures"


def _load(name: str) -> dict:
    path = ABL / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the corresponding ablation first.")
    return json.loads(path.read_text())


def _setup() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 11,
    })


def build_mmr() -> None:
    rows = [
        {
            "lambda": run["lambda"],
            "recall@10": run["metrics"]["recall@10"],
            "coverage@10": run["metrics"]["coverage@10"],
            "diversity": 1.0 - run["metrics"]["ils@10"],
        }
        for run in _load("mmr_lambda.json")["runs"]
    ]
    df = pd.DataFrame(rows).sort_values("lambda")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sc = ax.scatter(df["recall@10"], df["diversity"], c=df["lambda"], s=120, cmap="viridis")
    ax.plot(df["recall@10"], df["diversity"], color="gray", alpha=0.5)
    for _, row in df.iterrows():
        ax.annotate(
            f"λ={row['lambda']:.1f}",
            (row["recall@10"], row["diversity"]),
            xytext=(6, 6),
            textcoords="offset points",
        )
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Diversity (1 - ILS@10)")
    ax.set_title("MMR Pareto Frontier")
    fig.colorbar(sc, ax=ax, label="lambda")
    fig.tight_layout()
    fig.savefig(FIG / "mmr_pareto_scatter.png", bbox_inches="tight")
    plt.close(fig)


def build_cold_start() -> None:
    df = pd.DataFrame(_load("cold_start_bucket.json")["buckets"])
    order = ["cold (<20)", "warm (20-59)", "hot (60+)"]
    df = df.set_index("bucket").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["recall@10"], 0.4, label="Recall@10", color="#1f77b4")
    ax.bar(x + 0.2, df["coverage@10"], 0.4, label="Coverage@10", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(df["bucket"])
    ax.set_title("DIN Performance by User History Bucket")
    ax.legend()
    for i, row in enumerate(df.itertuples(index=False)):
        ax.text(i - 0.2, row[2] + 0.006, f"{row[2]:.3f}", ha="center", fontsize=9)
        ax.text(i + 0.2, row[5] + 0.006, f"{row[5]:.3f}", ha="center", fontsize=9)
        ax.text(i, max(row[2], row[5]) + 0.05, f"n={row.n_users}", ha="center", color="dimgray")
    fig.tight_layout()
    fig.savefig(FIG / "cold_start_bucket.png", bbox_inches="tight")
    plt.close(fig)


def build_fusion() -> None:
    data = _load("fusion_strategy.json")
    rows = []
    for name, metrics in data["single"].items():
        rows.append({"method": name, "kind": "single", **metrics})
    for name, metrics in data["fusion"].items():
        rows.append({"method": f"merge_{name}", "kind": "fusion", **metrics})
    df = pd.DataFrame(rows).sort_values("recall@10", ascending=False)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    colors = ["#2ca02c" if kind == "fusion" else "#1f77b4" for kind in df["kind"]]
    ax.bar(df["method"], df["recall@10"], color=colors)
    ax.set_ylabel("Recall@10")
    ax.set_title("Recall Fusion vs Single Channels")
    ax.tick_params(axis="x", rotation=18)
    for i, value in enumerate(df["recall@10"]):
        ax.text(i, value + 0.0015, f"{value:.4f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fusion_strategy_bar.png", bbox_inches="tight")
    plt.close(fig)


def build_din_attention() -> None:
    data = _load("din_attention.json")
    df = pd.DataFrame([
        {"variant": "attention", **data["attn"]["metrics"]},
        {"variant": "sum-pool", **data["sum"]["metrics"]},
    ])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].bar(df["variant"], df["recall@10"], color=["#1f77b4", "#aec7e8"])
    axes[0].set_title("Recall@10")
    axes[1].bar(df["variant"], df["valid_auc"], color=["#d62728", "#ff9896"])
    axes[1].set_title("Validation AUC")
    for ax, col in zip(axes, ["recall@10", "valid_auc"], strict=True):
        for i, value in enumerate(df[col]):
            ax.text(i, value + 0.001, f"{value:.4f}", ha="center", fontsize=10)
    fig.suptitle("DIN Attention Ablation")
    fig.tight_layout()
    fig.savefig(FIG / "din_attention_ablation.png", bbox_inches="tight")
    plt.close(fig)


def build_sasrec() -> None:
    rows = [
        {"max_seq_len": run["max_seq_len"], **run["metrics"]}
        for run in _load("sasrec_seq_len.json")["runs"]
    ]
    df = pd.DataFrame(rows).sort_values("max_seq_len")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(df["max_seq_len"], df["recall@10"], "o-", linewidth=2)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("max_seq_len")
    axes[0].set_ylabel("Recall@10")
    axes[0].set_title("Accuracy")
    axes[1].bar(df["max_seq_len"].astype(str), df["fit_seconds"], color="#ff7f0e")
    axes[1].set_xlabel("max_seq_len")
    axes[1].set_ylabel("Training seconds")
    axes[1].set_title("Training Cost")
    fig.suptitle("SASRec Sequence Length")
    fig.tight_layout()
    fig.savefig(FIG / "sasrec_seq_len.png", bbox_inches="tight")
    plt.close(fig)


def build_two_tower() -> None:
    rows = [
        {"embedding_dim": run["embedding_dim"], **run["metrics"]}
        for run in _load("two_tower_neg.json")["runs"]
    ]
    df = pd.DataFrame(rows).sort_values("embedding_dim")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(df["embedding_dim"], df["recall@10"], "o-", linewidth=2)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("embedding_dim")
    axes[0].set_ylabel("Recall@10")
    axes[0].set_title("Accuracy")
    axes[1].bar(df["embedding_dim"].astype(str), df["fit_seconds"], color="#d62728")
    axes[1].set_xlabel("embedding_dim")
    axes[1].set_ylabel("Training seconds")
    axes[1].set_title("Training Cost")
    fig.suptitle("Two-Tower Capacity")
    fig.tight_layout()
    fig.savefig(FIG / "two_tower_neg.png", bbox_inches="tight")
    plt.close(fig)


def build_funnel() -> None:
    data = _load("funnel.json")
    df = pd.DataFrame(data["stages"])
    df["retention"] = df["positives"] / float(data["n_eval"])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(df["name"], df["retention"], color="#9467bd")
    ax.set_ylabel("Positive survival rate")
    ax.set_title("Conversion Funnel: Positive Survival")
    ax.tick_params(axis="x", rotation=15)
    for i, row in enumerate(df.itertuples(index=False)):
        ax.text(i, row.retention + 0.01, f"{row.positives}/{data['n_eval']}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "funnel_bars.png", bbox_inches="tight")
    plt.close(fig)


def build_significance() -> None:
    data = _load("significance.json")
    rows = []
    for model, point in data["point_estimates"].items():
        lo, hi = data["bootstrap_ci_95"][model]
        rows.append({"model": model, "recall@10": point, "lo": lo, "hi": hi})
    df = pd.DataFrame(rows).sort_values("recall@10")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(df))
    ax.barh(y, df["recall@10"], color="#1f77b4")
    ax.errorbar(
        df["recall@10"],
        y,
        xerr=[df["recall@10"] - df["lo"], df["hi"] - df["recall@10"]],
        fmt="none",
        ecolor="black",
        capsize=4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"])
    ax.set_xlabel("Recall@10")
    ax.set_title("OOF Rankers: 95% Paired Bootstrap CI")
    fig.tight_layout()
    fig.savefig(FIG / "significance_ci.png", bbox_inches="tight")
    plt.close(fig)

    mat = pd.DataFrame(data["p_paired_bootstrap"]).loc[data["rankers"], data["rankers"]]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat.values, cmap="RdYlGn_r", vmin=0, vmax=0.1)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=30)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            value = mat.values[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white" if value < 0.05 else "black")
    ax.set_title("Paired Bootstrap p-values")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG / "significance_matrix.png", bbox_inches="tight")
    plt.close(fig)


def build_dashboard_overview() -> None:
    data = _load("significance.json")
    metrics = pd.Series(data["point_estimates"]).sort_values(ascending=False)
    mmr = pd.DataFrame([
        {
            "lambda": run["lambda"],
            "recall@10": run["metrics"]["recall@10"],
            "coverage@10": run["metrics"]["coverage@10"],
        }
        for run in _load("mmr_lambda.json")["runs"]
    ]).sort_values("lambda")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(metrics.index, metrics.values, color="#1f77b4")
    axes[0].set_title("Dashboard Metrics Tab")
    axes[0].set_ylabel("Recall@10")
    axes[1].plot(mmr["lambda"], mmr["recall@10"], "o-", label="Recall@10")
    axes[1].plot(mmr["lambda"], mmr["coverage@10"], "s--", label="Coverage@10")
    axes[1].set_title("Dashboard λ Compare Tab")
    axes[1].set_xlabel("MMR lambda")
    axes[1].legend()
    fig.suptitle("NeoRec Dashboard: Offline Metrics and MMR Trade-off")
    fig.tight_layout()
    fig.savefig(FIG / "dashboard_overview.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _setup()
    build_mmr()
    build_cold_start()
    build_fusion()
    build_din_attention()
    build_sasrec()
    build_two_tower()
    build_funnel()
    build_significance()
    build_dashboard_overview()
    print(f"Wrote README figures to {FIG}")


if __name__ == "__main__":
    main()
