# Ranking — Scheme A investigation (recall scores as ranker features)

> **TL;DR.** We implemented "Scheme A" — passing the five recall channels'
> scores plus the merge\_rrf fused score into every ranker as numeric features
> — and an optional companion fix (hard-negative mining from `merge_rrf`
> top-200). The implementation is clean and the AUC numbers are spectacular
> (0.99+), but the **end-to-end Recall@K collapses to ~0**. We traced this to
> a **look-ahead bias**: recall layers are fit on the same `train_df` we draw
> ranker positives from, so positives get artificially inflated scores at
> training time that no candidate can match at inference time. The proper fix
> requires retraining recall layers on a strictly-prior data slice (scheduled
> for W6); the Scheme A code remains in the repo as an opt-in toggle.

---

## 1. What Scheme A does

For every candidate `(u, i)` the ranker receives **12 extra columns** from a
new :class:`RecallFeatureStore`
([`src/neorec/ranking/recall_features.py`](../../src/neorec/ranking/recall_features.py)):

| Column block | Cols | Meaning |
|---|---|---|
| `score_*`   | 6 | z-scored channel score (`als`, `two_tower`, `sasrec`, `popularity`, `cold_start`, `merge_rrf`); 0 if the item wasn't in this channel's top-`depth=500` for the user |
| `mask_*`    | 6 | 1 if the channel found the item in its top-`depth`, else 0 |

Plumbing per ranker (idiomatic for each family):

* **LR** — appended as 12 dense columns inside the sparse `hstack` block (lr.py).
* **GBDT** — appended as 12 numeric columns to the DataFrame, no extra coding (gbdt.py).
* **DeepFM** — a `nn.Linear(12 → embedding_dim)` projects them into one extra
  "field embedding" that participates in the FM 2-nd-order cross **and** the
  deep tower; a separate `nn.Linear(12 → 1)` adds a wide linear path.
* **DIN** — concatenated into the head MLP input + a wide `nn.Linear(12 → 1)`
  added to the logit.

A `RecallFeatureStore` builds once (~10 s on ML-1M) and is cached to
`artifacts/rank/recall_features/recall_features_d{depth}.npz`. Companion
hard-negative sampler lives in `build_training_pairs(..., hard_candidates=...,
hard_negative_ratio=...)`.

---

## 2. Empirical results

All numbers below are produced by `python -m neorec.cli train rank rank=<name>
<overrides>` on `data=movielens_1m`, single CPU, seed=42. The "baseline" rows
match the W3 report exactly.

### 2.1 Logistic Regression

| Config | Valid AUC | Valid LogLoss | Recall@10 | Recall@50 | Recall@100 |
|---|---:|---:|---:|---:|---:|
| baseline (1:4 random)              | 0.872 | 0.339 | 0.0283 | 0.1180 | 0.2083 |
| 1:2 rand + 1:2 hard neg            | 0.683 | 0.457 | 0.0267 | 0.0800 | 0.1310 |
| **+ Scheme A** (recall feats only) | 0.985 | 0.106 | **0.0003** | 0.0075 | 0.0212 |
| **+ Scheme A + hard neg**          | 0.992 | 0.084 | **0.0005** | 0.0066 | 0.0188 |

### 2.2 GBDT (sklearn HistGradientBoosting)

| Config | Valid AUC | Valid LogLoss | Recall@10 | Recall@50 | Recall@100 |
|---|---:|---:|---:|---:|---:|
| baseline (1:4 random)            | 0.879 | 0.331 | 0.0277 | 0.1095 | 0.1951 |
| + Scheme A + hard neg            | 0.994 | 0.075 | **0.0000** | 0.0045 | 0.0172 |

### 2.3 DeepFM

| Config | Valid AUC | Valid LogLoss | Recall@10 | Recall@50 | Recall@100 |
|---|---:|---:|---:|---:|---:|
| baseline (1:4 random)         | 0.934 | 0.253 | 0.0343 | 0.1488 | 0.2890 |
| + hard neg only               | 0.789 | 0.404 | 0.0322 | 0.1318 | 0.2325 |
| + Scheme A + hard neg         | 0.998 | 0.047 | **0.0000** | 0.0050 | 0.0189 |

### 2.4 Pattern

Across every model family the same fingerprint repeats:

* **AUC inflates** toward 0.99+ (the model finds the "look-ahead" shortcut
  effortlessly).
* **End-to-end Recall@K collapses**, often to literally zero at K=10.
* The collapse is not a bug — the model's predictions *are* internally
  consistent, they just don't transfer to the inference distribution.

---

## 3. Root cause — look-ahead bias in recall scores

The diagnostic timeline:

1. With Valid AUC = 0.998 and Recall@10 = 0.0 simultaneously, the model is
   making confident *and* useless predictions — a classic train/test
   distribution shift.
2. Looking at the recall scores per pair:
   * Training positive `(u, i_train)`: `i_train ∈ train_df[u]`, so ALS /
     Two-Tower / SASRec / cold-start were **trained** to reconstruct that
     interaction. The score is artificially high.
   * Test positive `(u, i_test)`: held out from `train_df`. Recall layers
     only got it via generalisation; the score is moderate and indistinguishable
     from the ~1 000 other plausible items in `merge_rrf` top-1000.
3. The ranker therefore learns "recall-score in inflated regime → label = 1",
   which is trivially separable (→ AUC ≈ 1) and **completely useless at
   inference time** because no candidate has scores in that regime.

Hard-negative mining alone (sampling negatives from `merge_rrf` top-200)
doesn't fix this — hard negatives are also at the *moderate* score regime, so
the model still finds the inflated-positive shortcut.

This is the same class of bug as **target leakage** in supervised learning,
specialised to a two-stage recommender. In a real production system the bug
doesn't appear because labels are CTR clicks on a previous model's slate and
the recall layer was fit on a strictly-prior wall-clock data slice — there is
no overlap between "items in recall's training data" and "items the ranker
sees positives of".

---

## 4. What to do about it

### 4.1 Today — keep Scheme A as opt-in

Code stays. Defaults in [`configs/rank/*.yaml`](../../configs/rank/) are back
to the W3 baseline (`use_recall_features=false`, `hard_negative_ratio=0`,
`negative_ratio=4`) so the §7.2 numbers in the README stay reproducible.

To enable Scheme A explicitly:

```bash
neorec train rank rank=deepfm \
    rank.input.use_recall_features=true \
    rank.input.hard_negative_ratio=2 \
    rank.input.negative_ratio=2
```

The first invocation builds `artifacts/rank/recall_features/recall_features_d500.npz`
(~10 s). Subsequent runs load from this cache.

### 4.2 Proper fix — out-of-fold recall training (W6)

Replicate what a production wall-clock split would do:

1. Sort each user's `train_df` chronologically.
2. Hold out the **most recent 10 %** as `train_ranker_df`; keep the rest as
   `train_recall_df`.
3. Re-fit all five recall channels on `train_recall_df` only (`ALS`, `two_tower`,
   `sasrec`, `popularity`, `cold_start`).
4. Build the `RecallFeatureStore` from these freshly-fit recall layers — now the
   scores it returns for `train_ranker_df` items reflect **generalisation**, not
   memorisation.
5. Use `train_ranker_df` as ranker positives, sample hard negatives from
   `merge_rrf` top-200, and Scheme A becomes a clean wide-and-deep input that
   should give the predicted +50-80 % lift in Recall@10.

Estimated effort: ~half a day for the data split + recaller re-fit; the ranker
code already supports the rest.

### 4.3 Smaller, safer subsets of Scheme A worth trying first

Two channels are **structurally immune** to the look-ahead bias because they
don't depend on a memorised `(user, item)` interaction matrix:

* **`popularity_score`** — purely item-side; gives the same value to every
  user for a given item, so memorisation can't inflate it.
* **`merge_rrf rank`** (not score) — clipped to a bounded range
  `[1, depth]`; at inference time the test positive's rank in `merge_rrf`
  top-1 000 *is* a strong signal we can legitimately use.

A "Scheme A-light" using only those two columns can be tested by editing the
`channels` list in `RecallFeatureStore.__init__`. We don't ship empirical
numbers for it in this report — adding it is a single-line config change once
the W6 out-of-fold split lands, and is more naturally tested in that joint
setting.

---

## 5. Reproduction commands

```bash
# Baseline (W3 defaults)
neorec train rank rank=lr
neorec train rank rank=gbdt
neorec train rank rank=deepfm
neorec train rank rank=din

# Scheme A + hard negatives (this report's failure mode)
neorec train rank rank=lr     rank.input.use_recall_features=true rank.input.hard_negative_ratio=2 rank.input.negative_ratio=2
neorec train rank rank=gbdt   rank.input.use_recall_features=true rank.input.hard_negative_ratio=2 rank.input.negative_ratio=2
neorec train rank rank=deepfm rank.input.use_recall_features=true rank.input.hard_negative_ratio=2 rank.input.negative_ratio=2

# Hard negatives only (no recall feats) — milder degradation
neorec train rank rank=lr     rank.input.use_recall_features=false rank.input.hard_negative_ratio=2 rank.input.negative_ratio=2
neorec train rank rank=deepfm rank.input.use_recall_features=false rank.input.hard_negative_ratio=2 rank.input.negative_ratio=2
```

All runs are logged to MLflow under the `rank.<name>` experiment with the
config snapshot attached, so the table above is fully reproducible.
