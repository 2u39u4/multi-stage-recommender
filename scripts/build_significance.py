"""Compute paired bootstrap CI + pairwise p-values for the 4 OOF rankers.

Outputs:
    experiments/ablations/significance.json   (numerical results)
    experiments/results/figures/significance_ci.png
    experiments/results/figures/significance_matrix.png
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = (REPO_ROOT / "configs").as_posix()
sys.path.insert(0, (REPO_ROOT / "src").as_posix())

ABL = REPO_ROOT / "experiments" / "ablations"
FIG = REPO_ROOT / "experiments" / "results" / "figures"
ABL.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("significance")

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 140,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
    "font.size": 11,
})


def main() -> None:
    from neorec.eval.significance import bootstrap_ci, compare_models
    from neorec.ranking.features import RankingFeaturizer
    from neorec.ranking.train import _build_merge_candidates, _instantiate, _load_data

    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=["rank=din", "data.oof_split=true"])
    OmegaConf.set_struct(cfg, False)

    processed, train_history_df, ranker_positives_df, test_df = _load_data(cfg)
    featurizer = RankingFeaturizer(processed_dir=processed, max_genres=6, max_seq_len=50)
    featurizer.build_sequences(train_history_df)

    extra_seen = defaultdict(set)
    for u, i in zip(
        ranker_positives_df["user_id"].to_numpy(),
        ranker_positives_df["item_id"].to_numpy(),
    ):
        extra_seen[int(u)].add(int(i))
    cands = _build_merge_candidates(
        cfg=cfg,
        test_user_ids=test_df["user_id"].tolist(),
        pool_size=1000,
        extra_user_seen=extra_seen,
    )

    rankers = ["lr", "gbdt", "deepfm", "din"]
    per_user_scores: dict[str, np.ndarray] = {}

    for name in rankers:
        log.info("Scoring %s …", name)
        local_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        local_cfg.rank = OmegaConf.load(REPO_ROOT / "configs" / "rank" / f"{name}.yaml")
        model = _instantiate(name, local_cfg, featurizer)
        model.load(REPO_ROOT / "artifacts" / "rank_oof" / name)

        user_ids_used: list[int] = []
        cand_lists: list[list[int]] = []
        truths: list[int] = []
        for u, gt in zip(test_df["user_id"].tolist(), test_df["item_id"].tolist()):
            if u in cands and cands[u]:
                user_ids_used.append(int(u))
                cand_lists.append(cands[u])
                truths.append(int(gt))
        res = model.predict(user_ids_used, cand_lists, k=10)
        binary = np.array(
            [int(gt in row[:10]) for gt, row in zip(truths, res.item_ids.tolist())]
        )
        per_user_scores[name] = binary
        log.info(
            "  mean Recall@10 = %.4f (hits %d / %d)",
            binary.mean(), int(binary.sum()), len(binary),
        )

    # Bootstrap CI per model
    rows = []
    for name, sc in per_user_scores.items():
        pt, lo, hi = bootstrap_ci(sc, n_boot=1000, alpha=0.05, seed=42)
        rows.append({"model": name, "recall@10": pt, "ci_lo": lo, "ci_hi": hi})
    df = pd.DataFrame(rows).sort_values("recall@10", ascending=False).reset_index(drop=True)
    log.info("CIs:\n%s", df.to_string(index=False))

    # Pairwise p-values
    p_boot = compare_models(per_user_scores, method="paired_bootstrap", n_boot=1000)
    p_t    = compare_models(per_user_scores, method="paired_t")

    out = {
        "rankers":          rankers,
        "n_users":          int(len(next(iter(per_user_scores.values())))),
        "point_estimates":  {name: float(arr.mean()) for name, arr in per_user_scores.items()},
        "bootstrap_ci_95":  {row["model"]: [row["ci_lo"], row["ci_hi"]] for _, row in df.iterrows()},
        "p_paired_bootstrap": p_boot,
        "p_paired_t":         p_t,
    }
    (ABL / "significance.json").write_text(json.dumps(out, indent=2))

    # CI plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(df))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    ax.barh(y, df["recall@10"], color=palette[: len(df)])
    ax.errorbar(
        df["recall@10"], y,
        xerr=[df["recall@10"] - df["ci_lo"], df["ci_hi"] - df["recall@10"]],
        fmt="none", ecolor="black", capsize=4, linewidth=1.5,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"])
    ax.set_xlabel("Recall@10 (95% bootstrap CI)")
    ax.set_title("OOF rankers — point estimate + bootstrap CI")
    for i, (pt, lo, hi) in enumerate(
        zip(df["recall@10"], df["ci_lo"], df["ci_hi"])
    ):
        ax.text(pt + 0.0008, i, f"{pt:.4f}  [{lo:.4f}, {hi:.4f}]", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG / "significance_ci.png", bbox_inches="tight")
    plt.close()

    # Matrix plots
    p_boot_df = pd.DataFrame(p_boot)
    p_t_df    = pd.DataFrame(p_t)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, mat, title in [
        (axes[0], p_boot_df, "Paired bootstrap p-values"),
        (axes[1], p_t_df,    "Paired t-test p-values"),
    ]:
        im = ax.imshow(mat.values, cmap="RdYlGn_r", vmin=0, vmax=0.1)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=30)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index)
        for i in range(len(mat)):
            for j in range(len(mat.columns)):
                v = mat.values[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v < 0.05 else "black", fontsize=10)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="p-value")
    plt.tight_layout()
    plt.savefig(FIG / "significance_matrix.png", bbox_inches="tight")
    plt.close()

    log.info("Wrote %s + significance_ci.png + significance_matrix.png", ABL / "significance.json")


if __name__ == "__main__":
    main()
