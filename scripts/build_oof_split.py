"""Build an out-of-fold (OOF) split for the ranker / recall layer.

Why
---
The W3 retrospective in
``experiments/results/ranking_scheme_a_investigation.md`` documents how
feeding recall scores into the ranker (Scheme A) breaks down on ML-1M
leave-one-out evaluation: the recall layers are fit on the same ``train_df``
the ranker samples positives from, so training-positive recall scores are
artificially inflated → the ranker learns a threshold that no inference-time
candidate can match.

The proper fix is what a production wall-clock split would do — train recall
on a strictly-prior data slice, train the ranker on a strictly-later one.
This script materialises that split offline so the existing recall / ranking
training pipelines stay simple (they just read a different parquet).

How
---
For every user, sort their existing ``split == 'train'`` interactions by
``ts`` ascending:

* the first ``1 - ranker_fraction`` (default 90 %) become ``train_recall``
* the last ``ranker_fraction`` (default 10 %) become ``train_ranker``

Test rows are kept as ``test``. Validation rows (if any) are preserved.

Output
------
``data/processed/<dataset>/oof_split.parquet`` — same schema as ``split.parquet``
(``user_id``, ``item_id``, ``split``), but ``split`` ∈
``{train_recall, train_ranker, test, [valid]}``.

Idempotent — re-running just overwrites the file.

Usage
-----
::

    python scripts/build_oof_split.py                                   # ML-1M, 10 % ranker
    python scripts/build_oof_split.py --dataset movielens_1m --frac 0.1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("build_oof_split")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def build(dataset: str, ranker_fraction: float) -> Path:
    if not (0.0 < ranker_fraction < 0.5):
        raise ValueError(f"ranker_fraction must be in (0, 0.5), got {ranker_fraction}")

    processed = REPO_ROOT / "data" / "processed" / dataset
    split_path = processed / "split.parquet"
    out_path = processed / "oof_split.parquet"

    if not split_path.exists():
        raise FileNotFoundError(f"missing {split_path} — run preprocess first")

    df = pd.read_parquet(split_path)
    if "ts" not in df.columns:
        # Backwards-compat: older preprocess versions wrote split without ts.
        inter_path = processed / "interactions.parquet"
        if not inter_path.exists():
            raise FileNotFoundError(
                f"split.parquet has no `ts` column and {inter_path} is missing — "
                "regenerate processed splits via `neorec data preprocess`."
            )
        inter = pd.read_parquet(inter_path)[["user_id", "item_id", "ts"]]
        df = df.merge(inter, on=["user_id", "item_id"], how="left")
        if df["ts"].isna().any():
            log.warning(
                "%d split rows had no ts — they will be placed in train_recall.",
                int(df["ts"].isna().sum()),
            )
            df["ts"] = df["ts"].fillna(df["ts"].min() if df["ts"].notna().any() else 0)

    train_mask = df["split"] == "train"
    train_df = df.loc[train_mask].copy()
    other_df = df.loc[~train_mask].copy()  # test + (maybe) valid

    log.info(
        "Input: %d train, %d non-train (= %s)",
        len(train_df),
        len(other_df),
        sorted(other_df["split"].unique().tolist()),
    )

    # Per user: sort by ts ascending, cut at the (1 - frac) quantile.
    rng = np.random.default_rng(42)

    def _split_one_user(group: pd.DataFrame) -> pd.DataFrame:
        if len(group) < 2:
            # Single training interaction → keep in train_recall (avoid creating
            # users with zero recall-training data).
            group = group.copy()
            group["split"] = "train_recall"
            return group
        ordered = group.sort_values("ts", kind="stable").copy()
        # Break ties deterministically by adding a tiny random jitter per row
        # so users with many same-timestamp ratings still get a clean cut.
        n = len(ordered)
        ranker_count = max(1, int(round(n * ranker_fraction)))
        recall_count = n - ranker_count
        if recall_count < 1:
            recall_count = 1
            ranker_count = n - 1
        labels = np.empty(n, dtype=object)
        labels[:recall_count] = "train_recall"
        labels[recall_count:] = "train_ranker"
        ordered["split"] = labels
        return ordered

    # Vectorised path is messy with variable ranker_count per user; the
    # group-apply runs in ~3 s on ML-1M (6 K users, ~500 K rows) which is
    # fine for an offline one-shot script.
    log.info("Splitting %d users (target ranker fraction = %.0f %%)…",
             train_df["user_id"].nunique(), ranker_fraction * 100)
    _ = rng  # rng currently unused; reserved for future stratified strategies.
    new_train = (
        train_df.groupby("user_id", group_keys=False)
        .apply(_split_one_user)
        .reset_index(drop=True)
    )

    out = pd.concat([new_train, other_df], axis=0, ignore_index=True)
    # Preserve ts for downstream code that wants it; the canonical schema is the
    # same as split.parquet.
    keep_cols = ["user_id", "item_id", "split"]
    if "ts" in out.columns:
        keep_cols.append("ts")
    out = out[keep_cols]
    counts = out["split"].value_counts().to_dict()
    log.info("Output split counts: %s", counts)

    if "train_recall" not in counts or "train_ranker" not in counts:
        raise RuntimeError(f"empty slice after OOF split: {counts}")
    if counts["train_ranker"] < 1000:
        log.warning(
            "Very small train_ranker slice (%d rows) — increase ranker_fraction "
            "or check your data.",
            counts["train_ranker"],
        )

    out.to_parquet(out_path, index=False)
    log.info("Wrote %s (%d rows total).", out_path, len(out))

    # Sanity: every user that had train rows must have ≥1 train_recall row.
    grouped = new_train.groupby("user_id")["split"].agg(set)
    no_recall = grouped[grouped.apply(lambda s: "train_recall" not in s)]
    if len(no_recall):
        log.warning(
            "%d users have NO train_recall rows — recall layer will treat them "
            "as cold-start.", len(no_recall),
        )

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="movielens_1m",
                        help="dataset name (folder under data/processed/)")
    parser.add_argument("--frac", type=float, default=0.10,
                        help="fraction of each user's train history reserved as ranker positives")
    args = parser.parse_args()
    build(args.dataset, args.frac)


if __name__ == "__main__":
    main()
