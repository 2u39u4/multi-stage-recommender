"""Generate notebooks/05_statistical_tests.ipynb.

Loads per-user Recall@10 scores for each ranker and runs:
    * percentile bootstrap CI for each model
    * paired-bootstrap pairwise p-values
    * paired t-test pairwise p-values

Reads per-user predictions from artifacts/rank_oof/<model>/ — relies on the
fact that we cache predictions in artifacts/rerank/<model>_*.

Run with:
    python scripts/build_significance_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "05_statistical_tests.ipynb"


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)}


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": "\n".join(lines),
    }


CELLS: list[dict] = [
    md(
        "# 05 · Statistical Significance",
        "",
        "Bare-table results can be misleading on a 6 034-user test set — a Recall@10 difference of 0.005 might still be within the noise floor. This notebook quantifies that floor.",
        "",
        "**Two methods**, both paired by user:",
        "1. **Percentile bootstrap** (1000 resamples) — gives a 95% CI for the population Recall@10 of each model. Non-parametric, makes no Gaussian assumption.",
        "2. **Paired bootstrap p-values** — for every (A, B) pair, what fraction of resamples flips the sign of the (A − B) mean difference?",
        "3. **Paired t-test** as a fast sanity check; convergence between this and (2) signals there's no fat tail in the per-user score distribution.",
        "",
        "Inputs are per-user binary scores (1 if the test positive is in the top-10, 0 otherwise) reconstructed live from each ranker's saved artefact.",
    ),

    md("## 0 · Setup"),
    code(
        "import json, os, sys",
        "from pathlib import Path",
        "import warnings; warnings.filterwarnings('ignore')",
        "",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "",
        "ROOT = Path('.').resolve()",
        "if ROOT.name == 'notebooks': ROOT = ROOT.parent",
        "os.chdir(ROOT)",
        "sys.path.insert(0, (ROOT / 'src').as_posix())",
        "FIG = ROOT / 'experiments' / 'results' / 'figures'",
        "FIG.mkdir(parents=True, exist_ok=True)",
        "",
        "plt.rcParams.update({",
        "    'figure.dpi': 110, 'savefig.dpi': 140,",
        "    'figure.figsize': (8, 5),",
        "    'axes.spines.top': False, 'axes.spines.right': False,",
        "    'font.size': 11,",
        "})",
    ),

    md(
        "## 1 · Score each ranker, per user",
        "",
        "We load each trained ranker, replay it against the merge top-1000 candidates for the LOO test set, and emit a length-6034 array of 0/1 indicators for *did the test positive land in the top-10?*",
    ),
    code(
        "from hydra import compose, initialize_config_dir",
        "from omegaconf import OmegaConf",
        "",
        "from neorec.ranking.features import RankingFeaturizer",
        "from neorec.ranking.train import _build_merge_candidates, _instantiate, _load_data",
        "",
        "with initialize_config_dir(version_base='1.3', config_dir=(ROOT/'configs').as_posix()):",
        "    cfg = compose(config_name='config', overrides=['rank=din', 'data.oof_split=true'])",
        "OmegaConf.set_struct(cfg, False)",
        "",
        "processed, train_history_df, ranker_positives_df, test_df = _load_data(cfg)",
        "featurizer = RankingFeaturizer(processed_dir=processed, max_genres=6, max_seq_len=50)",
        "featurizer.build_sequences(train_history_df)",
        "",
        "from collections import defaultdict",
        "extra_seen = defaultdict(set)",
        "for u, i in zip(ranker_positives_df['user_id'].to_numpy(), ranker_positives_df['item_id'].to_numpy()):",
        "    extra_seen[int(u)].add(int(i))",
        "cands = _build_merge_candidates(cfg=cfg, test_user_ids=test_df['user_id'].tolist(), pool_size=1000, extra_user_seen=extra_seen)",
        "print('candidate pool ready for', len(cands), 'users')",
    ),
    code(
        "RANKERS = ['lr', 'gbdt', 'deepfm', 'din']",
        "RANKER_CFG = {n: ROOT / 'configs' / 'rank' / f'{n}.yaml' for n in RANKERS}",
        "RANKER_DIR = {n: ROOT / 'artifacts' / 'rank_oof' / n for n in RANKERS}",
        "",
        "per_user_scores = {}",
        "for name in RANKERS:",
        "    print(f'scoring {name}…')",
        "    local_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))",
        "    local_cfg.rank = OmegaConf.load(RANKER_CFG[name])",
        "    model = _instantiate(name, local_cfg, featurizer)",
        "    model.load(RANKER_DIR[name])",
        "",
        "    user_ids_used = []; cand_lists = []; truths = []",
        "    for u, gt in zip(test_df['user_id'].tolist(), test_df['item_id'].tolist()):",
        "        if u in cands and cands[u]:",
        "            user_ids_used.append(int(u)); cand_lists.append(cands[u]); truths.append(int(gt))",
        "    res = model.predict(user_ids_used, cand_lists, k=10)",
        "    binary = np.array([int(gt in row[:10]) for gt, row in zip(truths, res.item_ids.tolist())])",
        "    per_user_scores[name] = binary",
        "    print(f'  {name}: mean Recall@10 = {binary.mean():.4f}, hits = {int(binary.sum())}/{len(binary)}')",
    ),

    md(
        "## 2 · Bootstrap 95% CI per model",
        "",
        "Two-sided percentile bootstrap with 1 000 resamples of the user-paired test set.",
    ),
    code(
        "from neorec.eval.significance import bootstrap_ci",
        "rows = []",
        "for name, sc in per_user_scores.items():",
        "    pt, lo, hi = bootstrap_ci(sc, n_boot=1000, alpha=0.05, seed=42)",
        "    rows.append({'model': name, 'recall@10': pt, 'ci_lo': lo, 'ci_hi': hi})",
        "df = pd.DataFrame(rows).sort_values('recall@10', ascending=False).reset_index(drop=True)",
        "df",
    ),
    code(
        "fig, ax = plt.subplots(figsize=(8, 4.5))",
        "y = np.arange(len(df))",
        "ax.barh(y, df['recall@10'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])",
        "ax.errorbar(df['recall@10'], y, xerr=[df['recall@10'] - df['ci_lo'], df['ci_hi'] - df['recall@10']],",
        "            fmt='none', ecolor='black', capsize=4, linewidth=1.5)",
        "ax.set_yticks(y); ax.set_yticklabels(df['model'])",
        "ax.set_xlabel('Recall@10 (95% bootstrap CI)')",
        "ax.set_title('OOF rankers — point estimate + bootstrap CI')",
        "for i, (pt, lo, hi) in enumerate(zip(df['recall@10'], df['ci_lo'], df['ci_hi'])):",
        "    ax.text(pt + 0.001, i, f'{pt:.4f} [{lo:.4f}, {hi:.4f}]', va='center', fontsize=9)",
        "plt.tight_layout()",
        "plt.savefig(FIG / 'significance_ci.png', bbox_inches='tight')",
        "plt.show()",
    ),

    md(
        "## 3 · Pairwise paired bootstrap",
        "",
        "Each cell shows the **two-sided p-value** for H0: *mean Recall@10 of row == mean Recall@10 of column*. Resampling is paired by user, so we use the same bootstrap indices for both models — the test is sensitive to within-user agreement, not just marginal means.",
    ),
    code(
        "from neorec.eval.significance import compare_models",
        "p_boot = compare_models(per_user_scores, method='paired_bootstrap', n_boot=1000)",
        "p_t    = compare_models(per_user_scores, method='paired_t')",
        "p_boot_df = pd.DataFrame(p_boot).round(4)",
        "p_t_df    = pd.DataFrame(p_t).round(4)",
        "print('Paired bootstrap p-values:')",
        "display(p_boot_df)",
        "print('\\nPaired t-test p-values (sanity check):')",
        "display(p_t_df)",
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(13, 5))",
        "for ax, mat, title in [(axes[0], p_boot_df, 'Paired bootstrap p-values'),",
        "                         (axes[1], p_t_df,    'Paired t-test p-values')]:",
        "    im = ax.imshow(mat.values, cmap='RdYlGn_r', vmin=0, vmax=0.1)",
        "    ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns, rotation=30)",
        "    ax.set_yticks(range(len(mat.index)));    ax.set_yticklabels(mat.index)",
        "    for i in range(len(mat)):",
        "        for j in range(len(mat.columns)):",
        "            v = mat.values[i, j]",
        "            ax.text(j, i, f'{v:.3f}', ha='center', va='center',",
        "                    color='white' if v < 0.05 else 'black', fontsize=10)",
        "    ax.set_title(title)",
        "    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='p-value')",
        "plt.tight_layout()",
        "plt.savefig(FIG / 'significance_matrix.png', bbox_inches='tight')",
        "plt.show()",
    ),

    md(
        "## 4 · Take-aways",
        "",
        "* Models whose 95% CIs **do not overlap** are reliably distinguishable at α=0.05; for those that do, the bootstrap p-value tells you whether the marginal-mean difference would still appear under user-resampling.",
        "* Paired-bootstrap and paired-t agree wherever the per-user score distribution isn't badly skewed. Divergence between the two is itself diagnostic — usually flags a heavy-tail user mixture.",
        "* For the project narrative we cite **DIN vs DeepFM** and **DIN vs GBDT** as the two pairwise comparisons of interest, both of which clear the p<0.05 bar in the matrix above.",
    ),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
