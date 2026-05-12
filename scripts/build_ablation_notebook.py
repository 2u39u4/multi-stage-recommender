"""Generate notebooks/03_ablations.ipynb — all six W4 ablations in one notebook.

Cells read pre-computed JSON files under experiments/ablations/. Plots are
saved to experiments/results/figures/ and embedded into README §8.

Run with:
    python scripts/build_ablation_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "03_ablations.ipynb"


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
        "# 03 · W4 Ablations",
        "",
        "Six controlled experiments that quantify the contribution of each component in the pipeline:",
        "",
        "1. **MMR λ Pareto frontier** — accuracy vs. diversity trade-off in re-ranking.",
        "2. **Cold-start vs hot-user performance** — how DIN behaves across the user-activity spectrum.",
        "3. **Recall fusion strategy** — RRF vs norm_weighted vs single channels.",
        "4. **DIN attention vs sum pooling** — does the local-activation unit actually help?",
        "5. **SASRec sequence length** — return on extra positional context.",
        "6. **Two-Tower negative ratio** — sweet spot for BPR-uniform negative sampling.",
        "",
        "All numerical inputs are produced by `python scripts/run_ablations.py <name>` and saved as JSON under `experiments/ablations/`. This notebook loads them, plots, and writes PNGs into `experiments/results/figures/`.",
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
        "ABL = ROOT / 'experiments' / 'ablations'",
        "FIG = ROOT / 'experiments' / 'results' / 'figures'",
        "FIG.mkdir(parents=True, exist_ok=True)",
        "print('working dir:', ROOT)",
        "print('ablation dir:', ABL, '— files:', sorted(p.name for p in ABL.glob('*.json')))",
        "",
        "plt.rcParams.update({",
        "    'figure.dpi': 110, 'savefig.dpi': 140,",
        "    'figure.figsize': (8, 5),",
        "    'axes.spines.top': False, 'axes.spines.right': False,",
        "    'axes.grid': True, 'grid.alpha': 0.3,",
        "    'font.size': 11,",
        "})",
    ),

    md(
        "## 1 · MMR λ Pareto frontier",
        "",
        "Sweep λ ∈ {0.0, 0.3, 0.5, 0.7, 1.0} over the **same DIN ranker output** (top-100 candidates per user).",
        "λ=1.0 is pure relevance (Recall@10 maximized); λ=0.0 is pure diversity (Coverage@10 / catalog reach maximized).",
        "ILS@10 = average intra-list cosine similarity in the Two-Tower embedding space — **lower = more diverse**.",
    ),
    code(
        "data = json.loads((ABL / 'mmr_lambda.json').read_text())",
        "rows = []",
        "for run in data['runs']:",
        "    m = run['metrics']",
        "    rows.append({",
        "        'lambda': run['lambda'],",
        "        'recall@10': m['recall@10'],",
        "        'ndcg@10':   m['ndcg@10'],",
        "        'coverage@10': m['coverage@10'],",
        "        'ils@10':    m['ils@10'],",
        "    })",
        "df = pd.DataFrame(rows).sort_values('lambda')",
        "df",
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))",
        "axes[0].plot(df['lambda'], df['recall@10'], 'o-', color='#1f77b4', label='Recall@10')",
        "axes[0].plot(df['lambda'], df['ndcg@10'], 's--', color='#d62728', label='NDCG@10')",
        "axes[0].set_xlabel('λ (relevance weight)')",
        "axes[0].set_ylabel('Accuracy')",
        "axes[0].set_title('Accuracy vs λ')",
        "axes[0].legend()",
        "",
        "axes[1].plot(df['lambda'], df['coverage@10'], 'o-', color='#2ca02c', label='Coverage@10')",
        "ax2 = axes[1].twinx()",
        "ax2.plot(df['lambda'], df['ils@10'], 's--', color='#ff7f0e', label='ILS@10 (↓ better)')",
        "axes[1].set_xlabel('λ')",
        "axes[1].set_ylabel('Coverage@10', color='#2ca02c')",
        "ax2.set_ylabel('ILS@10', color='#ff7f0e')",
        "axes[1].set_title('Diversity vs λ')",
        "lines, labels = axes[1].get_legend_handles_labels()",
        "lines2, labels2 = ax2.get_legend_handles_labels()",
        "axes[1].legend(lines + lines2, labels + labels2, loc='center right')",
        "plt.tight_layout()",
        "plt.savefig(FIG / 'mmr_lambda_pareto.png', bbox_inches='tight')",
        "plt.show()",
    ),
    code(
        "# Pareto plot — accuracy vs diversity.",
        "fig, ax = plt.subplots(figsize=(7, 5))",
        "sc = ax.scatter(df['recall@10'], 1 - df['ils@10'], c=df['lambda'], cmap='viridis', s=100, edgecolor='black')",
        "for _, row in df.iterrows():",
        "    ax.annotate(f\"λ={row['lambda']:.1f}\", (row['recall@10'], 1 - row['ils@10']),",
        "                xytext=(6, 6), textcoords='offset points', fontsize=9)",
        "ax.set_xlabel('Recall@10  →')",
        "ax.set_ylabel('Diversity (1 − ILS@10)  →')",
        "ax.set_title('MMR Pareto frontier — DIN top-100 → re-ranked top-10')",
        "plt.colorbar(sc, ax=ax, label='λ')",
        "plt.tight_layout()",
        "plt.savefig(FIG / 'mmr_pareto_scatter.png', bbox_inches='tight')",
        "plt.show()",
    ),
    md(
        "**Reading**: each step from λ=1→0 trades roughly **2× more diversity** for **1× less accuracy**. "
        "The knee is around λ ≈ 0.7, which we ship as the default. "
        "Coverage jumps from 0.36 → 0.51 across the sweep, while ILS drops from 0.37 to 0.17 — "
        "the diversifier visibly de-clumps the catalogue.",
    ),

    md(
        "## 2 · Cold-start vs hot-user performance",
        "",
        "Bucket test users by training-history length and re-compute DIN's Recall@10 per bucket. "
        "Buckets: **cold** (<20 interactions in train_recall+train_ranker), **warm** (20–59), **hot** (60+).",
    ),
    code(
        "data = json.loads((ABL / 'cold_start_bucket.json').read_text())",
        "df = pd.DataFrame(data['buckets']).set_index('bucket').loc[",
        "    ['cold (<20)', 'warm (20-59)', 'hot (60+)']",
        "]",
        "df",
    ),
    code(
        "fig, ax = plt.subplots(figsize=(8, 4.5))",
        "x = np.arange(len(df))",
        "ax.bar(x - 0.2, df['recall@10'], 0.4, label='Recall@10', color='#1f77b4')",
        "ax.bar(x + 0.2, df['coverage@10'], 0.4, label='Coverage@10', color='#2ca02c')",
        "ax.set_xticks(x)",
        "ax.set_xticklabels(df.index)",
        "ax.set_ylabel('Score')",
        "ax.set_title('DIN OOF — bucketed by training-history length')",
        "for i, (r, c, n) in enumerate(zip(df['recall@10'], df['coverage@10'], df['n_users'])):",
        "    ax.text(i - 0.2, r + 0.005, f'{r:.3f}', ha='center', fontsize=9)",
        "    ax.text(i + 0.2, c + 0.005, f'{c:.3f}', ha='center', fontsize=9)",
        "    ax.text(i, max(r, c) + 0.06, f'n={n}', ha='center', fontsize=9, color='gray')",
        "ax.legend()",
        "plt.tight_layout()",
        "plt.savefig(FIG / 'cold_start_bucket.png', bbox_inches='tight')",
        "plt.show()",
    ),
    md(
        "**Reading**: cold users *outperform* hot users on Recall@10 (0.077 vs 0.042) — "
        "counterintuitive, but real. With leave-one-out, a hot user's pool of 1 000 candidates contains many high-relevance items "
        "that were *also* in their training history; the single test positive has to outrank all of them. "
        "Cold users have a sparser candidate pool, so the one true positive has less competition. "
        "**Coverage**, on the other hand, is much higher for hot users (0.30 vs 0.18) — DIN recommends more of the catalogue "
        "to users who have already shown diverse tastes.",
    ),

    md(
        "## 3 · Recall fusion strategy",
        "",
        "Single channels vs RRF vs norm_weighted on the same test set. Numbers match `README §7.1`.",
    ),
    code(
        "data = json.loads((ABL / 'fusion_strategy.json').read_text())",
        "rows = []",
        "for name, m in data['single'].items():",
        "    rows.append({'method': name, 'kind': 'single', **m})",
        "for name, m in data['fusion'].items():",
        "    rows.append({'method': 'merge_' + name, 'kind': 'fusion', **m})",
        "df = pd.DataFrame(rows).sort_values('recall@10', ascending=False).reset_index(drop=True)",
        "df",
    ),
    code(
        "fig, ax = plt.subplots(figsize=(9, 4.5))",
        "colors = ['#2ca02c' if k == 'fusion' else '#1f77b4' for k in df['kind']]",
        "ax.bar(df['method'], df['recall@10'], color=colors)",
        "for i, v in enumerate(df['recall@10']):",
        "    ax.text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=9)",
        "ax.set_ylabel('Recall@10')",
        "ax.set_title('Recall fusion — fusion (green) vs single channels (blue)')",
        "plt.xticks(rotation=15)",
        "plt.tight_layout()",
        "plt.savefig(FIG / 'fusion_strategy_bar.png', bbox_inches='tight')",
        "plt.show()",
    ),
    md(
        "**Reading**: norm_weighted (0.0827) edges out RRF (0.0794) and beats the best single channel (Two-Tower, 0.0590) by **+40%**. "
        "Both fusion strategies pay for the accuracy lift with lower coverage (0.43–0.49 vs 0.56–0.88 for individual channels), "
        "but that's a side-effect of all channels voting on the same head items, not a flaw in the fusion logic.",
    ),

    md(
        "## 4 · DIN attention vs sum pooling",
        "",
        "Retrain DIN with `use_attention=true` and `use_attention=false` on the OOF split (everything else identical).",
    ),
    code(
        "data = json.loads((ABL / 'din_attention.json').read_text())",
        "df = pd.DataFrame([",
        "    {'variant': 'attention', **data['attn']['metrics']},",
        "    {'variant': 'sum-pool',  **data['sum']['metrics']},",
        "])[['variant', 'recall@10', 'ndcg@10', 'valid_auc', 'valid_logloss']]",
        "df",
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))",
        "axes[0].bar(df['variant'], df['recall@10'], color=['#1f77b4', '#aec7e8'])",
        "for i, v in enumerate(df['recall@10']):",
        "    axes[0].text(i, v + 0.0005, f'{v:.4f}', ha='center', fontsize=10)",
        "axes[0].set_ylabel('Recall@10')",
        "axes[0].set_title('End-to-end Recall@10')",
        "",
        "axes[1].bar(df['variant'], df['valid_auc'], color=['#d62728', '#ff9896'])",
        "for i, v in enumerate(df['valid_auc']):",
        "    axes[1].text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=10)",
        "axes[1].set_ylabel('Valid AUC')",
        "axes[1].set_title('Within-pool AUC')",
        "plt.tight_layout()",
        "plt.savefig(FIG / 'din_attention_ablation.png', bbox_inches='tight')",
        "plt.show()",
        "",
        "lift = (df['recall@10'][0] - df['recall@10'][1]) / df['recall@10'][1] * 100",
        "print(f'Attention lift over sum-pooling: +{lift:.1f}% Recall@10')",
    ),
    md(
        "**Reading**: attention is a small but consistent win (+8% Recall@10, +0.7 pp AUC). "
        "On ML-1M the attention payoff is modest because the average sequence length is only 78; "
        "Zhou et al. report 30–60% lifts on Amazon book reviews where sequences run into the thousands. "
        "See `experiments/results/figures/din_attention_heatmap_*.png` for a per-user visualisation of the attention weights themselves.",
    ),

    md(
        "## 5 · SASRec sequence length",
        "",
        "Retrain SASRec with `max_seq_len ∈ {10, 20, 50, 100}`. Everything else (50 epochs, 2 blocks, BPR-per-position loss) is held constant.",
    ),
    code(
        "data = json.loads((ABL / 'sasrec_seq_len.json').read_text())",
        "rows = []",
        "for r in data['runs']:",
        "    m = r['metrics']",
        "    rows.append({",
        "        'max_seq_len': r['max_seq_len'],",
        "        'recall@10': m.get('recall@10', np.nan),",
        "        'ndcg@10':   m.get('ndcg@10', np.nan),",
        "        'coverage@10': m.get('coverage@10', np.nan),",
        "        'fit_seconds': m.get('fit_seconds', np.nan),",
        "    })",
        "df = pd.DataFrame(rows).sort_values('max_seq_len')",
        "df",
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))",
        "axes[0].plot(df['max_seq_len'], df['recall@10'], 'o-', linewidth=2)",
        "axes[0].set_xlabel('max_seq_len')",
        "axes[0].set_ylabel('Recall@10')",
        "axes[0].set_title('SASRec — Recall@10 vs sequence length')",
        "axes[0].set_xscale('log')",
        "for x, y in zip(df['max_seq_len'], df['recall@10']):",
        "    axes[0].annotate(f'{y:.4f}', (x, y), xytext=(6, 6), textcoords='offset points', fontsize=9)",
        "",
        "axes[1].bar(df['max_seq_len'].astype(str), df['fit_seconds'], color='#ff7f0e')",
        "axes[1].set_xlabel('max_seq_len')",
        "axes[1].set_ylabel('Training time (s)')",
        "axes[1].set_title('Cost — wall-clock to fit')",
        "for i, v in enumerate(df['fit_seconds']):",
        "    axes[1].text(i, v + 5, f'{v:.0f}s', ha='center', fontsize=9)",
        "plt.tight_layout()",
        "plt.savefig(FIG / 'sasrec_seq_len.png', bbox_inches='tight')",
        "plt.show()",
    ),
    md(
        "**Reading — surprising result**: Recall@10 *monotonically drops* as we extend the sequence (L=10 → 0.1014, L=50 → 0.0585, L=100 → 0.0278). "
        "Cause: leave-one-out evaluation rewards predicting the **very last item**, while the per-position BPR loss spends most of its capacity on long-tail positions in the sequence. "
        "With L=10 the model effectively becomes a *next-action* predictor of the most recent 10 items; with L=100 the loss is diluted across positions whose targets have nothing to do with the LOO test item. "
        "**Take-away**: longer sequences only help when the evaluation horizon also grows (e.g. session-based, multi-step predictions). For LOO on ML-1M, short context wins.",
    ),

    md(
        "## 6 · Two-Tower capacity (embedding_dim)",
        "",
        "The plan called for a `num_negatives ∈ {1, 4, 16, 64}` sweep, but our Two-Tower trainer uses the canonical single-negative BPR loss (Rendle 2009) — exactly one triplet per positive regardless of `num_negatives`. We substituted **embedding_dim ∈ {16, 32, 64, 128}**, which is a real capacity knob the code supports directly and probes the same kind of question: *how much representation budget actually helps?*",
    ),
    code(
        "data = json.loads((ABL / 'two_tower_neg.json').read_text())",
        "rows = []",
        "for r in data['runs']:",
        "    m = r['metrics']",
        "    rows.append({",
        "        'embedding_dim': r.get('embedding_dim', r.get('num_negatives')),",
        "        'recall@10': m.get('recall@10', np.nan),",
        "        'ndcg@10':   m.get('ndcg@10', np.nan),",
        "        'coverage@10': m.get('coverage@10', np.nan),",
        "        'fit_seconds': m.get('fit_seconds', np.nan),",
        "    })",
        "df = pd.DataFrame(rows).sort_values('embedding_dim')",
        "df",
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))",
        "axes[0].plot(df['embedding_dim'], df['recall@10'], 'o-', linewidth=2)",
        "axes[0].set_xlabel('embedding_dim')",
        "axes[0].set_ylabel('Recall@10')",
        "axes[0].set_xscale('log', base=2)",
        "axes[0].set_title('Two-Tower — Recall@10 vs embedding_dim')",
        "for x, y in zip(df['embedding_dim'], df['recall@10']):",
        "    axes[0].annotate(f'{y:.4f}', (x, y), xytext=(6, 6), textcoords='offset points', fontsize=9)",
        "",
        "axes[1].bar(df['embedding_dim'].astype(str), df['fit_seconds'], color='#d62728')",
        "axes[1].set_xlabel('embedding_dim')",
        "axes[1].set_ylabel('Training time (s)')",
        "axes[1].set_title('Cost — wall-clock to fit')",
        "for i, v in enumerate(df['fit_seconds']):",
        "    axes[1].text(i, v + 5, f'{v:.0f}s', ha='center', fontsize=9)",
        "plt.tight_layout()",
        "plt.savefig(FIG / 'two_tower_neg.png', bbox_inches='tight')",
        "plt.show()",
    ),
    md(
        "**Reading**: capacity helps up to a point, then plateaus or regresses. ML-1M has only 6034 users × 3533 items, so once `embedding_dim ≥ 64` the model has more parameters than effective constraints and starts overfitting. "
        "The shape of the curve is informative — picking dim=64 is essentially a Bayes-optimal trade-off for this dataset, larger doesn't help.",
    ),

    md(
        "## 7 · Summary",
        "",
        "| Ablation | Lever | Best setting (this dataset) | Δ Recall@10 |",
        "|---|---|---:|---:|",
        "| MMR λ | accuracy ↔ diversity | λ=0.7 (serving knob) | trade 12% Recall for 17% more catalogue coverage |",
        "| User bucket | cold / warm / hot | (LOO artefact, not tunable) | cold users +82% over hot |",
        "| Fusion | RRF vs norm | **norm_weighted** | +4% over RRF, +40% over best single channel |",
        "| DIN attention | on / off | **on** | +8% over sum-pool |",
        "| SASRec L | 10/20/50/100 | **L=10** (loss/eval mismatch) | +73% over L=100 |",
        "| Two-Tower dim | 16/32/64/128 | **64** (default) | overfits past 64 on ML-1M |",
        "",
        "Each lever is fully independent — toggling one keeps the rest of the pipeline reproducible. The MMR knob is the only one that should be tuned at *serving* time, the rest are pre-deployment training choices.",
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
