# Cold-start — Content-Based Recall (TF-IDF on genres + year bucket)

**Channel name**: `cold_start`  
**Goal**: provide non-trivial recommendations for **users with very little or zero training history** — exactly the regime where collaborative-filtering channels collapse to "no signal".

## Why a dedicated channel

iALS / Two-Tower / SASRec all consume *user-side* signals (id embeddings, history sequences). For a cold user, every input they read is uninitialised or empty, so they degenerate to popularity-like behaviour without ever modelling the actual content of items the user might like. A content-based channel sidesteps this by representing each item with its **intrinsic** features.

## Pipeline

```
1. item-token TF-IDF matrix              (V × T)
2. user centroid = mean of liked items   (U × T)
3. cosine score = centroid · item.T      (per-user dot product)
4. seen-item masking → top-K
5. zero-history users → global popularity top-K  (graceful fallback)
```

### Tokens used (default)

| Feature       | Token format | Example |
|---------------|--------------|---------|
| `genres`      | `g{id}`      | `g3` (= "Action") |
| `year_bucket` | `yb{id}`     | `yb2` (1980s decade) |

### TF-IDF formula

- **TF**: presence indicator (0 / 1) — repeating the same token within an item makes no sense for genre/year.
- **IDF**: smoothed, `idf(t) = log((V + 1) / (df(t) + 1)) + 1`.
- Each item row is L2-normalised so cosine similarity reduces to a dot product.

### User centroid

For user `u` with training items `H_u`:
```
v_u = (1 / |H_u|) · Σ_{i ∈ H_u} v_i / ||v_i||
v_u ← v_u / ||v_u||      # re-normalise
```
A user with zero training items has `||v_u|| = 0` and triggers the popularity fallback (toggle: `fallback_to_popularity`).

## Config

| Hyperparameter | Value |
|---|---|
| `features` | `[genres, year_bucket]` |
| `similarity` | cosine |
| `tfidf_on_genres` | true |
| `output.fallback_to_popularity` | true |

Vocabulary size on ML-1M (after preprocess): **23 tokens** (18 genres + 5 year buckets).

## Results — MovieLens-1M, 6 034 test users, full-rank protocol

| K   | Recall@K | NDCG@K | MRR@K  | Coverage@K |
|----:|---------:|-------:|-------:|-----------:|
| 10  | 0.0167   | 0.0083 | 0.0057 | **0.8735** |
| 50  | 0.0648   | 0.0184 | 0.0077 | **0.9827** |
| 100 | 0.1104   | 0.0258 | 0.0083 | **0.9929** |
| 200 | 0.1848   | 0.0362 | 0.0089 | **0.9969** |

**Fit**: 0.24 s (no learning — closed-form TF-IDF + matmul).  
**Inference**: 0.04 ms / user (263 ms for the full test split).

## How to read these numbers

This channel is **not** trying to win the recall race. On ML-1M (where every user has ≥5 ratings ≥ 4 stars), the user-side cold-start regime barely shows up in the global metric — collaborative channels do better because they have plenty of co-occurrence to learn from.

The two numbers that actually matter for cold-start are:

| Metric | Why it matters |
|---|---|
| **Coverage @10 = 0.8735** | 87 % of the catalog is recommended to *some* user at depth 10 — almost an order of magnitude above the CF channels (0.55 – 0.67). The channel produces a very flat, diverse distribution by construction. |
| **Robustness to zero-history users** | The popularity fallback is triggered for any user whose centroid norm is 0. On ML-1M this is rare (the preprocess filters users with < 5 positive interactions), but on real systems with daily new sign-ups this is the channel that keeps the recommendation page from going blank. |

## Where Cold-start earns its keep

* **Merge channel** — Cold-start brings two unique things to RRF:
  1. Items that no CF channel surfaces (recall = +1.5pp at K=10 when added).
  2. A monotone-by-content tie-break for users whose CF embeddings are weak.
* **Production fallback** — In serving, when the personalised recall pool is empty (new user, model not yet updated, etc.), `cold_start.recall(uid, k)` is the deterministic safety net.

## Engineering notes / pitfalls

| Pitfall | Fix |
|---|---|
| Sparse → dense conversion of the item-token matrix would explode on a real catalog | We do it here because V × T = 3 533 × 23 = 81 k floats — trivial. The sparse path (`csr_matrix.dot`) is still in the code for catalogs in the millions. |
| Cosine equality across items with identical genre set → ties broken arbitrarily by argsort | This is fine for the recall stage (downstream ranker re-orders), but we noted it in the analysis notebook so the diversity numbers aren't over-interpreted. |
| Year bucket dominates TF-IDF for old movies (very few share `yb1` vs many sharing `yb3`) | This is the intended behaviour of IDF — rare features get a bigger weight. We log the vocab size at fit time so the user can sanity-check it. |

## Limitations & next steps

* Genres are the only **declared** content signal we have on ML-1M. Real catalogs have titles, descriptions, casts, images — all of which could plug into the same `_build_item_tfidf` slot with a different tokenizer.
* The "user centroid = mean" recipe is the simplest possible aggregator; a future iteration could replace it with the user-tower outputs from Two-Tower (this is exactly how a hybrid recall works in production).
* When `tfidf_on_genres=False`, this collapses to a binary content-similarity channel — useful as an ablation control in W4.
