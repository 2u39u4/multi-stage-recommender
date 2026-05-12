"""Generate the DIN attention heatmap PNG + notebooks/03_ranking_din_attention.ipynb.

Loads the saved DIN model from ``artifacts/rank/din/``, picks 5 users with
rich history, and visualises the attention weights between each user's
history and the held-out test positive.

Run with:  python scripts/build_din_attention.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402

from neorec.ranking.din import DINRanker  # noqa: E402
from neorec.ranking.features import RankingFeaturizer  # noqa: E402

OUT_PNG = ROOT / "experiments" / "results" / "ranking" / "din_attention_heatmap.png"
OUT_NB = ROOT / "notebooks" / "03_ranking_din_attention.ipynb"
PROCESSED = ROOT / "data" / "processed" / "movielens_1m"


def load_din() -> tuple[DINRanker, RankingFeaturizer, pd.DataFrame]:
    """Re-instantiate the trained DIN model + featurizer."""
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=["rank=din"])

    featurizer = RankingFeaturizer(
        processed_dir=PROCESSED,
        max_genres=6,
        max_seq_len=50,
    )
    train_df = (
        pd.read_parquet(PROCESSED / "interactions.parquet")
        .merge(
            pd.read_parquet(PROCESSED / "split.parquet")[["user_id", "item_id", "split"]],
            on=["user_id", "item_id"],
            how="inner",
        )
        .query("split == 'train'")
        .reset_index(drop=True)
    )
    featurizer.build_sequences(train_df)

    ranker = DINRanker(cfg, featurizer)
    ranker.load(ROOT / "artifacts" / "rank" / "din")
    return ranker, featurizer, train_df


def select_users(featurizer: RankingFeaturizer, test_df: pd.DataFrame, n: int = 5) -> list[int]:
    """Pick ``n`` users whose history fills the full ``max_seq_len`` — the
    heatmap is more interpretable when no padding rows show through."""
    rng = np.random.default_rng(42)
    history_len = featurizer._user_history_mask.sum(axis=1)
    rich_users = np.where(history_len >= featurizer.max_seq_len)[0]
    test_users = set(test_df["user_id"].tolist())
    candidates = [u for u in rich_users if u in test_users]
    rng.shuffle(candidates)
    return candidates[:n]


def main() -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    OUT_NB.parent.mkdir(parents=True, exist_ok=True)

    print("Loading DIN model and featurizer…")
    ranker, featurizer, train_df = load_din()

    test_df = (
        pd.read_parquet(PROCESSED / "split.parquet")
        .query("split == 'test'")
        .reset_index(drop=True)
    )
    items = pd.read_parquet(PROCESSED / "item_features.parquet").set_index("item_id")
    titles = items["title"].to_dict()

    users = select_users(featurizer, test_df, n=5)
    test_index = test_df.set_index("user_id")["item_id"].to_dict()

    user_ids = np.array(users, dtype=np.int64)
    target_items = np.array([test_index[u] for u in users], dtype=np.int64)

    print(f"Computing attention for users {users} on their held-out positives…")
    weights, history, mask = ranker.attention_for_users(user_ids, target_items)

    # Normalise per-row to make the colour scale comparable across users.
    w_norm = np.zeros_like(weights, dtype=np.float32)
    for i in range(weights.shape[0]):
        valid = weights[i][mask[i] > 0]
        if valid.size == 0:
            continue
        v_min, v_max = float(valid.min()), float(valid.max())
        denom = max(v_max - v_min, 1e-6)
        w_norm[i] = (weights[i] - v_min) / denom
        w_norm[i][mask[i] == 0] = np.nan

    # ----------- Render heatmap (5 rows × 50 cols) -----------
    fig, ax = plt.subplots(figsize=(14, 5.5))
    im = ax.imshow(w_norm, cmap="magma", aspect="auto",
                   vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_yticks(range(len(users)))
    y_labels = []
    for u, t in zip(users, target_items):
        title = titles.get(int(t), f"item {t}")
        y_labels.append(f"u{u} → {title[:35]}")
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel("History position (older  →  newer)", fontsize=11)
    ax.set_title("DIN attention weights · target item vs user history (normalised per row)",
                 fontsize=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("relative attention weight", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    print(f"✓ Saved heatmap → {OUT_PNG.relative_to(ROOT)}")

    # ----------- Build companion notebook -----------
    cells: list[dict] = []

    def md(*lines: str) -> None:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)})

    def code(*lines: str) -> None:
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": "\n".join(lines),
        })

    md(
        "# 03 · DIN Attention Visualisation",
        "",
        "Loads the saved DIN model and shows, for 5 random users, how attention",
        "redistributes when the **same user history** is scored against their",
        "actual held-out positive.",
        "",
        "Image artefact: `experiments/results/ranking/din_attention_heatmap.png`",
        "(also embedded into README §7.2).",
    )

    md("## 1 · Setup")
    code(
        "import sys, os",
        "from pathlib import Path",
        "ROOT = Path('.').resolve()",
        "if ROOT.name == 'notebooks': ROOT = ROOT.parent",
        "sys.path.insert(0, str(ROOT / 'src'))",
        "os.chdir(ROOT)",
        "",
        "import numpy as np, pandas as pd, torch",
        "import matplotlib.pyplot as plt",
        "from hydra import compose, initialize_config_dir",
        "from neorec.ranking.din import DINRanker",
        "from neorec.ranking.features import RankingFeaturizer",
        "",
        "plt.rcParams.update({'figure.dpi': 110, 'axes.grid': False})",
    )

    md("## 2 · Re-instantiate trained DIN")
    code(
        "with initialize_config_dir(version_base='1.3', config_dir=str(ROOT / 'configs')):",
        "    cfg = compose(config_name='config', overrides=['rank=din'])",
        "",
        "PROC = ROOT / 'data' / 'processed' / 'movielens_1m'",
        "featurizer = RankingFeaturizer(processed_dir=PROC, max_genres=6, max_seq_len=50)",
        "train_df = (pd.read_parquet(PROC / 'interactions.parquet')",
        "            .merge(pd.read_parquet(PROC / 'split.parquet')[['user_id','item_id','split']],",
        "                   on=['user_id','item_id'], how='inner')",
        "            .query('split == \"train\"')",
        "            .reset_index(drop=True))",
        "featurizer.build_sequences(train_df)",
        "ranker = DINRanker(cfg, featurizer)",
        "ranker.load(ROOT / 'artifacts' / 'rank' / 'din')",
        "print('DIN ready, embedding_dim =', ranker.model.embedding_dim,",
        "      'use_attention =', ranker.model.use_attention)",
    )

    md(
        "## 3 · Pick 5 users + their held-out positives",
        "",
        "We filter to users whose history fills the full `max_seq_len=50` —",
        "the heatmap is more interpretable when no padding rows show through.",
    )
    code(
        "test_df = (pd.read_parquet(PROC / 'split.parquet')",
        "           .query('split == \"test\"').reset_index(drop=True))",
        "items = pd.read_parquet(PROC / 'item_features.parquet').set_index('item_id')",
        "titles = items['title'].to_dict()",
        "",
        "rng = np.random.default_rng(42)",
        "hist_len = featurizer._user_history_mask.sum(axis=1)",
        "rich = np.where(hist_len >= featurizer.max_seq_len)[0]",
        "test_users = set(test_df['user_id'].tolist())",
        "users = [u for u in rich if u in test_users]",
        "rng.shuffle(users)",
        "users = users[:5]",
        "test_idx = test_df.set_index('user_id')['item_id'].to_dict()",
        "targets = [test_idx[u] for u in users]",
        "print('users:', users)",
        "print('targets:', [titles.get(t,'?')[:40] for t in targets])",
    )

    md("## 4 · Compute attention weights and visualise")
    code(
        "w, history, mask = ranker.attention_for_users(",
        "    np.array(users, dtype=np.int64), np.array(targets, dtype=np.int64))",
        "",
        "w_norm = np.zeros_like(w, dtype=np.float32)",
        "for i in range(w.shape[0]):",
        "    valid = w[i][mask[i] > 0]",
        "    if valid.size == 0: continue",
        "    a, b = float(valid.min()), float(valid.max())",
        "    w_norm[i] = (w[i] - a) / max(b - a, 1e-6)",
        "    w_norm[i][mask[i] == 0] = np.nan",
        "",
        "fig, ax = plt.subplots(figsize=(14, 5.5))",
        "im = ax.imshow(w_norm, cmap='magma', aspect='auto', vmin=0., vmax=1.)",
        "ax.set_yticks(range(len(users)))",
        "ax.set_yticklabels([f'u{u} → ' + titles.get(int(t), f'i{t}')[:35]",
        "                    for u, t in zip(users, targets)], fontsize=10)",
        "ax.set_xlabel('History position (older → newer)')",
        "ax.set_title('DIN attention weights · target item vs user history')",
        "fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='relative attention')",
        "plt.tight_layout()",
        "plt.show()",
    )

    md(
        "## 5 · Discussion",
        "",
        "* **Bright cells** are history items the attention unit considers most",
        "  similar to the target. In the well-behaved cases, you'll see a few",
        "  ‘spikes’ aligned with movies of the same genre or franchise as the target.",
        "* **Even rows** indicate attention failed to discriminate — the model",
        "  fell back to sum-pooling.  This happens when the user's history is too",
        "  homogeneous (e.g. a casual viewer who rates only blockbusters).",
        "* The ablation result in §7.2 shows: full DIN gets +3.2 pts AUC over the",
        "  no-attention baseline, but its end-to-end Recall@10 drops because the",
        "  CTR head over-trusts history-target similarity on hard negatives",
        "  sampled by the recall layer.  This motivates **hard-negative mining**",
        "  in W6.",
    )

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"✓ Saved notebook → {OUT_NB.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
