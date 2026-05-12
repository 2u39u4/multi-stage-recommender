"""Generate notebooks/04_funnel_conversion.ipynb.

Loads experiments/ablations/funnel.json (produced by build_funnel.py) and
plots a horizontal funnel showing positive-survival across stages.

Run with:
    python scripts/build_funnel_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "04_funnel_conversion.ipynb"


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
        "# 04 · Funnel Conversion Analysis",
        "",
        "How many test positives survive each stage of the pipeline?",
        "",
        "```",
        "  merge top-1000 (recall)   →   DeepFM top-100 (pre-rank)   →   DIN top-20 (fine-rank)   →   MMR top-10 (rerank)",
        "```",
        "",
        "Inputs come from `scripts/build_funnel.py` (saved as `experiments/ablations/funnel.json`).",
        "",
        "**Why this matters**: a recommender is only as good as its weakest stage. Plotting the funnel makes the bottleneck visually obvious — and tells us where to spend the next month's engineering effort.",
    ),

    md("## 0 · Setup"),
    code(
        "import json, os",
        "from pathlib import Path",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import matplotlib.patches as mpatches",
        "",
        "ROOT = Path('.').resolve()",
        "if ROOT.name == 'notebooks': ROOT = ROOT.parent",
        "os.chdir(ROOT)",
        "ABL = ROOT / 'experiments' / 'ablations'",
        "FIG = ROOT / 'experiments' / 'results' / 'figures'",
        "FIG.mkdir(parents=True, exist_ok=True)",
        "",
        "plt.rcParams.update({",
        "    'figure.dpi': 110, 'savefig.dpi': 140,",
        "    'figure.figsize': (10, 5),",
        "    'axes.spines.top': False, 'axes.spines.right': False,",
        "    'font.size': 11,",
        "})",
    ),

    md("## 1 · Load the funnel summary"),
    code(
        "data = json.loads((ABL / 'funnel.json').read_text())",
        "stages = pd.DataFrame(data['stages'])",
        "n_eval = data['n_eval']",
        "stages['recall@stage'] = stages['positives'] / n_eval",
        "stages['drop_vs_prev'] = stages['positives'].diff().fillna(0).astype(int)",
        "stages['retention']    = (stages['positives'] / stages['positives'].iloc[0]).round(3)",
        "print(f'Evaluated {n_eval} test users with a non-empty candidate pool.')",
        "stages",
    ),

    md(
        "## 2 · Horizontal funnel diagram",
        "",
        "Each bar is sized by *positive count*; the wedge between adjacent stages is the **drop-out** at that step. Numbers above the bars show how many of the original 6034 LOO positives we still hold.",
    ),
    code(
        "fig, ax = plt.subplots(figsize=(11, 5))",
        "colors = ['#4878d0', '#ee854a', '#6acc64', '#d65f5f']",
        "for i, (_, row) in enumerate(stages.iterrows()):",
        "    width = row['positives'] / n_eval",
        "    ax.barh(0, width, left=i, color=colors[i], edgecolor='black')",
        "    ax.text(i + width / 2, 0, f\"{row['name']}\\nsize={row['size']}\\npositives={row['positives']}\\nrecall@K={row['recall@stage']:.3f}\",",
        "            ha='center', va='center', fontsize=10, color='white' if i < 3 else 'black')",
        "ax.set_xlim(0, len(stages))",
        "ax.set_ylim(-0.5, 0.5)",
        "ax.set_yticks([])",
        "ax.set_xticks(np.arange(len(stages)) + 0.5)",
        "ax.set_xticklabels(stages['name'])",
        "ax.set_title('Conversion funnel — positives surviving each stage')",
        "plt.tight_layout()",
        "plt.savefig(FIG / 'funnel_bars.png', bbox_inches='tight')",
        "plt.show()",
    ),

    md(
        "## 3 · Sankey-style flow diagram",
        "",
        "Sometimes a stacked-bar view is cleaner than a Sankey when the funnel is strictly tapering. The chart below plots **% of original positives retained** at each cut-off.",
    ),
    code(
        "fig, ax = plt.subplots(figsize=(10, 4.5))",
        "x = np.arange(len(stages))",
        "ax.plot(x, stages['retention'], 'o-', linewidth=3, color='#1f77b4', markersize=10)",
        "for i, (name, ret, pos) in enumerate(zip(stages['name'], stages['retention'], stages['positives'])):",
        "    ax.annotate(f'{ret*100:.1f}%\\n({pos} pos)', (i, ret),",
        "                xytext=(0, 12), textcoords='offset points', ha='center', fontsize=10)",
        "ax.set_xticks(x)",
        "ax.set_xticklabels(stages['name'])",
        "ax.set_ylabel('Retention of stage-0 positives')",
        "ax.set_ylim(0, 1.05)",
        "ax.set_title('Stage-by-stage retention of merge-stage positives')",
        "ax.grid(axis='y', alpha=0.3)",
        "plt.tight_layout()",
        "plt.savefig(FIG / 'funnel_retention.png', bbox_inches='tight')",
        "plt.show()",
    ),

    md(
        "## 4 · Drop-out attribution",
        "",
        "Where in the pipeline do we lose the most positives? The bars below show the **count of positives lost** at each transition.",
    ),
    code(
        "drops = stages[['name', 'positives']].copy()",
        "drops['lost'] = stages['positives'].shift(1) - stages['positives']",
        "drops = drops.iloc[1:].reset_index(drop=True)",
        "drops['transition'] = ['→ ' + n for n in drops['name']]",
        "",
        "fig, ax = plt.subplots(figsize=(9, 4.5))",
        "ax.bar(drops['transition'], drops['lost'].astype(int), color='#d62728')",
        "for i, v in enumerate(drops['lost'].astype(int)):",
        "    ax.text(i, v + 5, str(int(v)), ha='center', fontsize=10)",
        "ax.set_ylabel('Test positives lost at this transition')",
        "ax.set_title('Funnel bottleneck — where positives disappear')",
        "plt.xticks(rotation=10)",
        "plt.tight_layout()",
        "plt.savefig(FIG / 'funnel_dropouts.png', bbox_inches='tight')",
        "plt.show()",
    ),

    md(
        "## 5 · Take-aways",
        "",
        "* **The recall stage is the dominant ceiling** — once a positive isn't in the merged top-1000, no downstream model can recover it. Investments in better fusion / candidate generation pay off across every downstream metric.",
        "* **DeepFM 100→DIN 20** drops a chunk; this is where fine-rank earns its keep on a real pipeline (smaller candidate sets give DIN's attention more leverage).",
        "* **MMR 20→10** is a small but visible drop — that's the accuracy-diversity trade-off in action; tunable via λ.",
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
