# Logistic Regression — sanity-floor baseline

**Model name**: `lr`
**Stage**: baseline (treated as a sanity floor, not a production candidate)

## Architecture

Pure linear model on one-hot / hashed features:

```
ϕ(u, i) = onehot ⊕ multihot                  → R^{6208}
            │
            └─ user_id_hash (4 096)  ⊕  item_id_hash (2 048)
               ⊕  gender(2) ⊕ age_bucket(6) ⊕ occupation(21)
               ⊕  year_bucket(5) ⊕ pop_bucket(10)
               ⊕  genre_multihot(19)
                       │
                       ▼
              σ(  w · ϕ(u, i)  )
```

The hashing trick keeps the parameter count bounded (≈ 6.2 K weights vs. ~10 M
if we one-hot every id) at the cost of collisions in the highest-cardinality
fields. Solver: `liblinear`, `C = 1.0`, `max_iter = 200`.

## Final config

| Hyperparameter | Value |
|---|---:|
| `hash_dim_user` | 4096 |
| `hash_dim_item` | 2048 |
| `C` | 1.0 |
| `solver` | liblinear |
| `max_iter` | 200 |

## Results on MovieLens-1M (leave-one-out, full-rank evaluation)

| Metric | Value |
|---|---:|
| **Valid AUC** | **0.8715** |
| **Valid LogLoss** | **0.3390** |
| Valid accuracy | 0.8454 |
| Recall@10 (end-to-end) | 0.0283 |
| NDCG@10  (end-to-end) | 0.0126 |
| Recall@100 (end-to-end) | 0.2083 |
| Re-rank latency / user | **0.39 ms** |
| Training time | 18 s |

## What the numbers tell us

* AUC = 0.87 says LR can already separate positives from **random** negatives
  surprisingly well — but a CTR model trained on random negatives is **not**
  the same task as re-ranking the recall layer's hard negatives.
* End-to-end Recall@10 = 0.028 confirms LR cannot beat the raw merge-channel
  output (Recall@10 = 0.0793). Without ID embeddings or feature crosses, LR
  has nothing to add to what the recall layer already encoded.
* **Treat LR as a sanity floor** — every deeper model must clear this bar on
  the binary CTR task. If a deep model can't beat AUC 0.87 we have a bug.

## When to look at this baseline

* Quick CI smoke test (trains in < 30 s, no PyTorch).
* Sanity check for the feature pipeline — if `valid_pos_rate` deviates from
  the configured `negative_ratio = 4` (i.e. ~20 %), something is wrong upstream.
