"""Render every figure used in README §8 and the W4 markdown from JSON.

This bypasses Jupyter so the figures regenerate on a vanilla CI machine
with only matplotlib + numpy + pandas.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ABL = ROOT / "experiments" / "ablations"
FIG = ROOT / "experiments" / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 140,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
    "font.size": 11,
})


# ---------------------------------------------------------------------------
def mmr_lambda():
    data = json.loads((ABL / "mmr_lambda.json").read_text())
    rows = [
        {"lambda": r["lambda"], **r["metrics"]} for r in data["runs"]
    ]
    df = pd.DataFrame(rows).sort_values("lambda")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(df["lambda"], df["recall@10"], "o-", color="#1f77b4", label="Recall@10")
    axes[0].plot(df["lambda"], df["ndcg@10"], "s--", color="#d62728", label="NDCG@10")
    axes[0].set_xlabel("λ (relevance weight)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy vs λ")
    axes[0].legend()

    axes[1].plot(df["lambda"], df["coverage@10"], "o-", color="#2ca02c", label="Coverage@10")
    ax2 = axes[1].twinx()
    ax2.plot(df["lambda"], df["ils@10"], "s--", color="#ff7f0e", label="ILS@10 (↓ better)")
    axes[1].set_xlabel("λ")
    axes[1].set_ylabel("Coverage@10", color="#2ca02c")
    ax2.set_ylabel("ILS@10", color="#ff7f0e")
    axes[1].set_title("Diversity vs λ")
    lines, labels = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines + lines2, labels + labels2, loc="center right")
    plt.tight_layout()
    plt.savefig(FIG / "mmr_lambda_pareto.png", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(df["recall@10"], 1 - df["ils@10"], c=df["lambda"], cmap="viridis", s=100, edgecolor="black")
    for _, row in df.iterrows():
        ax.annotate(f"λ={row['lambda']:.1f}", (row["recall@10"], 1 - row["ils@10"]),
                    xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Recall@10  →")
    ax.set_ylabel("Diversity (1 − ILS@10)  →")
    ax.set_title("MMR Pareto frontier — DIN top-100 → re-ranked top-10")
    plt.colorbar(sc, ax=ax, label="λ")
    plt.tight_layout()
    plt.savefig(FIG / "mmr_pareto_scatter.png", bbox_inches="tight")
    plt.close()
    print("  wrote mmr_lambda_pareto.png + mmr_pareto_scatter.png")


def cold_start_bucket():
    data = json.loads((ABL / "cold_start_bucket.json").read_text())
    df = pd.DataFrame(data["buckets"]).set_index("bucket").loc[
        ["cold (<20)", "warm (20-59)", "hot (60+)"]
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["recall@10"], 0.4, label="Recall@10", color="#1f77b4")
    ax.bar(x + 0.2, df["coverage@10"], 0.4, label="Coverage@10", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(df.index)
    ax.set_ylabel("Score")
    ax.set_title("DIN OOF — bucketed by training-history length")
    for i, (r, c, n) in enumerate(zip(df["recall@10"], df["coverage@10"], df["n_users"])):
        ax.text(i - 0.2, r + 0.005, f"{r:.3f}", ha="center", fontsize=9)
        ax.text(i + 0.2, c + 0.005, f"{c:.3f}", ha="center", fontsize=9)
        ax.text(i, max(r, c) + 0.06, f"n={n}", ha="center", fontsize=9, color="gray")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "cold_start_bucket.png", bbox_inches="tight")
    plt.close()
    print("  wrote cold_start_bucket.png")


def fusion_strategy():
    data = json.loads((ABL / "fusion_strategy.json").read_text())
    rows = []
    for name, m in data["single"].items():
        rows.append({"method": name, "kind": "single", **m})
    for name, m in data["fusion"].items():
        rows.append({"method": "merge_" + name, "kind": "fusion", **m})
    df = pd.DataFrame(rows).sort_values("recall@10", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#2ca02c" if k == "fusion" else "#1f77b4" for k in df["kind"]]
    ax.bar(df["method"], df["recall@10"], color=colors)
    for i, v in enumerate(df["recall@10"]):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylabel("Recall@10")
    ax.set_title("Recall fusion — fusion (green) vs single channels (blue)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(FIG / "fusion_strategy_bar.png", bbox_inches="tight")
    plt.close()
    print("  wrote fusion_strategy_bar.png")


def din_attention():
    data = json.loads((ABL / "din_attention.json").read_text())
    df = pd.DataFrame([
        {"variant": "attention", **data["attn"]["metrics"]},
        {"variant": "sum-pool", **data["sum"]["metrics"]},
    ])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(df["variant"], df["recall@10"], color=["#1f77b4", "#aec7e8"])
    for i, v in enumerate(df["recall@10"]):
        axes[0].text(i, v + 0.0005, f"{v:.4f}", ha="center", fontsize=10)
    axes[0].set_ylabel("Recall@10")
    axes[0].set_title("End-to-end Recall@10")
    axes[1].bar(df["variant"], df["valid_auc"], color=["#d62728", "#ff9896"])
    for i, v in enumerate(df["valid_auc"]):
        axes[1].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=10)
    axes[1].set_ylabel("Valid AUC")
    axes[1].set_title("Within-pool AUC")
    plt.tight_layout()
    plt.savefig(FIG / "din_attention_ablation.png", bbox_inches="tight")
    plt.close()
    print("  wrote din_attention_ablation.png")


def sasrec_seq_len():
    data = json.loads((ABL / "sasrec_seq_len.json").read_text())
    rows = []
    for r in data["runs"]:
        m = r["metrics"]
        rows.append({
            "max_seq_len": r["max_seq_len"],
            "recall@10": m.get("recall@10"),
            "fit_seconds": m.get("fit_seconds"),
        })
    df = pd.DataFrame(rows).sort_values("max_seq_len")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(df["max_seq_len"], df["recall@10"], "o-", linewidth=2)
    axes[0].set_xlabel("max_seq_len")
    axes[0].set_ylabel("Recall@10")
    axes[0].set_title("SASRec — Recall@10 vs sequence length")
    axes[0].set_xscale("log")
    for x, y in zip(df["max_seq_len"], df["recall@10"]):
        axes[0].annotate(f"{y:.4f}", (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9)
    axes[1].bar(df["max_seq_len"].astype(str), df["fit_seconds"], color="#ff7f0e")
    axes[1].set_xlabel("max_seq_len")
    axes[1].set_ylabel("Training time (s)")
    axes[1].set_title("Cost — wall-clock to fit")
    for i, v in enumerate(df["fit_seconds"]):
        axes[1].text(i, v + 5, f"{v:.0f}s", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG / "sasrec_seq_len.png", bbox_inches="tight")
    plt.close()
    print("  wrote sasrec_seq_len.png")


def two_tower_capacity():
    path = ABL / "two_tower_neg.json"
    if not path.exists():
        print("  skipping two_tower_neg.json (not yet computed)")
        return
    data = json.loads(path.read_text())
    rows = []
    for r in data["runs"]:
        m = r["metrics"]
        rows.append({
            "embedding_dim": r.get("embedding_dim", r.get("num_negatives")),
            "recall@10": m.get("recall@10"),
            "fit_seconds": m.get("fit_seconds"),
        })
    df = pd.DataFrame(rows).sort_values("embedding_dim")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(df["embedding_dim"], df["recall@10"], "o-", linewidth=2)
    axes[0].set_xlabel("embedding_dim")
    axes[0].set_ylabel("Recall@10")
    axes[0].set_xscale("log", base=2)
    axes[0].set_title("Two-Tower — Recall@10 vs embedding_dim")
    for x, y in zip(df["embedding_dim"], df["recall@10"]):
        axes[0].annotate(f"{y:.4f}", (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9)
    axes[1].bar(df["embedding_dim"].astype(str), df["fit_seconds"], color="#d62728")
    axes[1].set_xlabel("embedding_dim")
    axes[1].set_ylabel("Training time (s)")
    axes[1].set_title("Cost — wall-clock to fit")
    for i, v in enumerate(df["fit_seconds"]):
        axes[1].text(i, v + 5, f"{v:.0f}s", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG / "two_tower_neg.png", bbox_inches="tight")
    plt.close()
    print("  wrote two_tower_neg.png")


def funnel():
    path = ABL / "funnel.json"
    if not path.exists():
        print("  skipping funnel.json (not yet computed)")
        return
    data = json.loads(path.read_text())
    n_eval = data["n_eval"]
    stages = pd.DataFrame(data["stages"])
    stages["recall@stage"] = stages["positives"] / n_eval
    stages["retention"] = stages["positives"] / stages["positives"].iloc[0]

    # Bar funnel
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#4878d0", "#ee854a", "#6acc64", "#d65f5f"]
    for i, (_, row) in enumerate(stages.iterrows()):
        width = row["positives"] / n_eval
        ax.barh(0, width, left=i, color=colors[i], edgecolor="black")
        ax.text(i + width / 2, 0,
                f"{row['name']}\nsize={row['size']}\npositives={row['positives']}\nrecall@K={row['recall@stage']:.3f}",
                ha="center", va="center", fontsize=10, color="white" if i < 3 else "black")
    ax.set_xlim(0, len(stages))
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks(np.arange(len(stages)) + 0.5)
    ax.set_xticklabels(stages["name"])
    ax.set_title("Conversion funnel — positives surviving each stage")
    plt.tight_layout()
    plt.savefig(FIG / "funnel_bars.png", bbox_inches="tight")
    plt.close()

    # Retention curve
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(stages))
    ax.plot(x, stages["retention"], "o-", linewidth=3, color="#1f77b4", markersize=10)
    for i, (name, ret, pos) in enumerate(zip(stages["name"], stages["retention"], stages["positives"])):
        ax.annotate(f"{ret*100:.1f}%\n({pos} pos)", (i, ret),
                    xytext=(0, 12), textcoords="offset points", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(stages["name"])
    ax.set_ylabel("Retention of stage-0 positives")
    ax.set_ylim(0, 1.05)
    ax.set_title("Stage-by-stage retention of merge-stage positives")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "funnel_retention.png", bbox_inches="tight")
    plt.close()

    # Drop-out bars
    drops = stages[["name", "positives"]].copy()
    drops["lost"] = stages["positives"].shift(1) - stages["positives"]
    drops = drops.iloc[1:].reset_index(drop=True)
    drops["transition"] = ["→ " + n for n in drops["name"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(drops["transition"], drops["lost"].astype(int), color="#d62728")
    for i, v in enumerate(drops["lost"].astype(int)):
        ax.text(i, v + 5, str(int(v)), ha="center", fontsize=10)
    ax.set_ylabel("Test positives lost at this transition")
    ax.set_title("Funnel bottleneck — where positives disappear")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(FIG / "funnel_dropouts.png", bbox_inches="tight")
    plt.close()
    print("  wrote funnel_bars.png + funnel_retention.png + funnel_dropouts.png")


def main():
    print("Rendering figures into", FIG)
    mmr_lambda()
    cold_start_bucket()
    fusion_strategy()
    din_attention()
    sasrec_seq_len()
    two_tower_capacity()
    funnel()
    print("Done.")


if __name__ == "__main__":
    main()
