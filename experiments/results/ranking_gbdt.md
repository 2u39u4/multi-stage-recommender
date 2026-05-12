# Gradient-Boosted Trees — side-feature CTR baseline

**Model name**: `gbdt`
**Stage**: baseline (de-facto industrial CTR baseline for years before deep models took over)
**Implementation**: scikit-learn's `HistGradientBoostingClassifier`
(histogram-based gradient boosting, same algorithmic family as
LightGBM / XGBoost — but bundled with sklearn so we avoid the macOS `libomp`
dependency LightGBM's wheel needs.)

## Features

`HistGBDT`'s categorical splitter caps cardinality at 255, so we drop the
raw `user_id` / `item_id` columns (6 034 / 3 533 categories) and rely on:

| Feature | Cardinality | Type |
|---|---:|---|
| `gender` | 2 | categorical |
| `age_bucket` | 6 | categorical |
| `occupation` | 21 | categorical |
| `year_bucket` | 5 | categorical |
| `popularity_bucket` | 10 | ordinal |
| `genre_0/1/2` | ≤ 19 each | categorical |
| `user_hist_len` | 0 – 50 | ordinal |

This makes GBDT effectively a **side-feature-only** baseline — exactly the
kind of model that motivates moving to deep architectures with ID embeddings.

## Final config

| Hyperparameter | Value |
|---|---:|
| `num_boost_round` (max_iter) | 300 |
| `learning_rate` | 0.05 |
| `num_leaves` (max_leaf_nodes) | 63 |
| `min_data_in_leaf` (min_samples_leaf) | 100 |
| `early_stopping_rounds` | 30 |
| `random_state` | 42 |

## Results on MovieLens-1M (leave-one-out)

| Metric | Value |
|---|---:|
| **Valid AUC** | **0.8793** |
| **Valid LogLoss** | **0.3310** |
| Valid accuracy | 0.8492 |
| Recall@10 (end-to-end) | 0.0277 |
| NDCG@10  (end-to-end) | 0.0119 |
| Recall@100 (end-to-end) | 0.1951 |
| Re-rank latency / user | 1.64 ms |
| Training time | ~10 s |

## What the numbers tell us

* Even without any user/item ID features, GBDT beats LR by **+0.008 AUC**
  thanks to non-linear feature interactions.
* End-to-end Recall@K is roughly flat vs. LR — the side features carry
  enough signal to separate positives from random negatives, but not enough
  to discriminate inside the recall layer's tight pool of hard negatives.
* GBDT's strength shows up in production when *cheap* tabular features are
  added (`user_avg_rating`, `item_ctr_3d`, `user_freshness_score`); on our
  current schema it's mostly a sanity baseline.

## Reproducibility hooks

* The model is saved as `gbdt.joblib` plus a `meta.json` with the column
  order — so loading and feature alignment is fully deterministic.
* `categorical_features` are derived from the schema at fit time; no manual
  list to keep in sync.
