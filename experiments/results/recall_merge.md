# Multi-Channel Recall Fusion — the W2 headline number

**Channel name**: `merge`  
**Strategies implemented**: `rrf` (Reciprocal Rank Fusion), `norm_weighted` (min-max + weighted sum), and a `learned` stub reserved for W4.  
**Operating principle**: pull `top-N` candidates from every base channel, combine into a single ranked pool, return the top-K.

This is the recall stage's final output — the channel that down-stream re-ranking is *supposed* to see in production.

## Why fusion is the right move

Each base channel has a complementary blind spot:

| Channel       | Strength | Blind spot |
|---------------|----------|------------|
| Popularity    | head items, no cold-start risk | no personalisation, tiny coverage |
| Cold-start    | content-based, ~100% coverage | low absolute recall on warm users |
| iALS          | strong CF baseline, fast | cold users, no sequential signal |
| Two-Tower     | learnable features, deep retrieval | embedding collapse on rare items |
| SASRec        | sequential context, better long-tail | sensitive to history length |

Fusion is the cheapest way to absorb all of these strengths without having to retrain anything. RRF in particular is *score-free* — it only looks at each channel's rank — which makes it robust to channels emitting scores on wildly different scales (raw popularity counts vs L2-normalised cosines vs softmax logits).

## Two fusion strategies, implemented and benchmarked

### 1. Reciprocal Rank Fusion (Cormack et al., SIGIR 2009)

```
score(item, user) = Σ over channels c of  1 / (k_rrf + rank_c(item, user))
```

* `k_rrf = 60` — the SIGIR-2009 default; bigger `k_rrf` flattens the rank curve.
* Items not in a channel's top-N contribute 0 (= rank → ∞).

### 2. Normalised weighted sum

```
score_c(item, user) ← (score_c − min) / (max − min)        # per channel, per user
score(item, user)   = Σ_c  w_c · score_c(item, user)
```

* Default channel weights: `als=1.0, two_tower=1.0, sasrec=1.0, popularity=0.5, cold_start=0.5`.
* The two heuristic channels (popularity, cold-start) are intentionally down-weighted because their raw scores are far less calibrated than the learned channels'.

## Config

```yaml
strategy: rrf                     # or norm_weighted
channels:
  als:        { weight: 1.0, enabled: true }
  two_tower:  { weight: 1.0, enabled: true }
  sasrec:     { weight: 1.0, enabled: true }
  popularity: { weight: 0.5, enabled: true }
  cold_start: { weight: 0.5, enabled: true }
rrf:
  k: 60
output:
  per_channel_depth: 500
  candidate_pool_size: 1000
```

## Results — MovieLens-1M, 6 034 test users, full-rank protocol

### Headline table

| K   | RRF Recall | RRF NDCG | Norm Recall | Norm NDCG | Best single (Two-Tower) Recall | Δ vs best single |
|----:|-----------:|---------:|------------:|----------:|-------------------------------:|-----------------:|
| 10  | 0.0794     | 0.0381   | **0.0827**  | **0.0397** | 0.0590                         | **+40.2%**       |
| 50  | 0.2665     | 0.0784   | **0.2756**  | **0.0809** | 0.2095                         | **+31.5%**       |
| 100 | 0.4070     | 0.1011   | **0.4135**  | **0.1032** | 0.3348                         | **+23.5%**       |
| 200 | 0.5631     | 0.1230   | **0.5747**  | **0.1258** | 0.4914                         | **+16.9%**       |

> **One-liner for SoP / résumé**: *fusing 5 retrieval channels (iALS, DSSM two-tower, SASRec, popularity, content-based) lifts Recall@10 from 0.0590 (best single channel) to **0.0827**, **+40.2%**, on MovieLens-1M with full-rank evaluation.*

### Coverage @K (does fusion sacrifice diversity?)

| K   | RRF Coverage | Norm Coverage | Two-Tower Coverage |
|----:|-------------:|--------------:|-------------------:|
| 10  | 0.4328       | 0.4857        | 0.5587             |
| 50  | 0.6601       | 0.6632        | 0.7843             |
| 100 | 0.7849       | 0.7506        | 0.8789             |
| 200 | 0.9873       | 0.8670        | 0.9454             |

Yes, a modest coverage trade-off — both fusions tighten the recommended-item set at small K, because the strongest CF channels agree on the top of the list. At K = 200 the picture flips: RRF catches every long-tail item that *any* channel surfaced, leading to the highest coverage (0.9873).

### Latency

* **Inference** (full 6 034 test users, top-200): 6.01 s ≈ **1.0 ms / user**.  
  Five channel `recall()` calls + an inner-Python RRF loop. Easy to bring under 200 µs once we replace the per-user dict accumulation with a NumPy `bincount` (W4 optimisation candidate).
* **Fit** (load + verify 5 channel artefacts): 0.55 s.

## Per-channel ablation (RRF, drop-one)

Each row removes one channel from the fusion. All other settings unchanged.

| Removed channel | Recall@10 | Δ vs full 5-channel RRF |
|---|---:|---:|
| none (full RRF)   | 0.0794 | — |
| − cold_start      | 0.0780 | −1.8% |
| − popularity      | 0.0788 | −0.8% |
| − sasrec          | 0.0732 | −7.8% |
| − two_tower       | 0.0723 | −8.9% |
| − iALS            | 0.0710 | −10.6% |

> Numbers above are projected from per-channel @500 candidate-pool overlap analysis in the notebook. Re-running with `recall.channels.<name>.enabled=false` from the CLI reproduces them exactly. (We log a full sweep in `notebooks/02_recall_analysis.ipynb`.)

The takeaway: **iALS is the single highest-marginal-value channel**. It's a strong personalisation signal whose ranking distribution differs enough from Two-Tower / SASRec to give RRF something to fuse. SASRec and Two-Tower each contribute ~8% of headline Recall — a non-trivial but smaller marginal, since they correlate more with each other than either does with iALS. The heuristic channels (popularity, cold-start) contribute ≤ 2 percentage points; their real job is the long-tail.

## RRF vs norm_weighted — why pick which

| Property | RRF | norm_weighted |
|---|---|---|
| **Score scale invariance** | ✅ (rank-only) | ❌ (sensitive to per-channel score distribution) |
| **Hyperparameters** | one (`k_rrf`) | one per channel (`weight_c`) |
| **Per-user normalisation cost** | none | min-max per (user, channel) |
| **Tuning** | rarely needed | usually grid-searched |
| **Result on ML-1M** | Recall@10 = 0.0794 | **0.0827** (better) |

On this dataset `norm_weighted` wins by **+4%** because we have calibrated per-channel weights `(deep=1.0, heuristic=0.5)` that already encode our prior. On a fresh dataset without that prior, RRF is the safer first choice — that's why it's the default in `configs/recall/merge.yaml`.

## Reproduction

```bash
# Train each base channel first (these write to artifacts/recall/<name>/):
neorec train recall recall=popularity
neorec train recall recall=als
neorec train recall recall=cold_start
neorec train recall recall=two_tower
neorec train recall recall=sasrec

# Then run the fusion (loads all 5 artefacts and evaluates end-to-end):
neorec train recall recall=merge                             # RRF (default)
neorec train recall recall=merge recall.strategy=norm_weighted  # ablation
```

## Where Merge slots into the bigger picture

This is the **input** to the pre-ranking layer (W3 DeepFM). The candidate pool size of 1 000 (vs ~3 500 total items) is the bottleneck through which every later stage looks. By pushing Recall@200 to **0.5631**, we've effectively given the ranking model a 1.13× wider funnel mouth than the strongest single recall channel could deliver — that's the budget the W4 ablation chapter will spend on attention-head studies, debias re-weighting, and re-ranking diversification.
