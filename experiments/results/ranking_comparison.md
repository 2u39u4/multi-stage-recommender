# Ranking head-to-head — LR · GBDT · DeepFM · DIN

This is the W3 deliverable from §4 of the execution plan: four rankers, the
same training data, the same end-to-end evaluation harness, **one apples-to-apples
table**.

## Setup

| Component | Value |
|---|---|
| Dataset | MovieLens-1M (filtered to rating ≥ 4, ≥ 5 interactions / user) |
| Split | leave-one-out per user; 6 034 test users |
| Negative sampling (train) | uniform random, 1 : 4 |
| Train / valid split | 90 / 10 by row, shuffled |
| Candidate pool (end-to-end) | merge channel (RRF over 5 base channels), top-1 000 |
| Re-rank output | top-100 from each model |
| Evaluation | Recall / NDCG / HitRate / MRR @ {10, 50, 100} + ranker AUC / LogLoss |

## Results

| Rank | Model | Stage | Valid AUC | Valid LogLoss | Recall@10 | NDCG@10 | Recall@100 | NDCG@100 | Latency / user |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | **DIN (full)**       | fine-rank   | **0.9692** | **0.1692** | 0.0126 | 0.0055 | 0.2126 | 0.0415 | 4.33 ms |
| 2 | DIN (no-attention)   | ablation    | 0.9373 | 0.2478 | 0.0336 | 0.0153 | **0.3005** | **0.0653** | 0.94 ms |
| 3 | **DeepFM**           | pre-rank    | 0.9342 | 0.2534 | **0.0343** | **0.0151** | 0.2890 | 0.0627 | **0.51 ms** |
| 4 | GBDT (Hist)          | baseline    | 0.8793 | 0.3310 | 0.0277 | 0.0119 | 0.1951 | 0.0432 | 1.64 ms |
| 5 | LR                   | baseline    | 0.8715 | 0.3390 | 0.0283 | 0.0126 | 0.2083 | 0.0458 | 0.39 ms |

Numbers are pulled directly from MLflow runs `26dbf0e2…` (LR),
`856be6b6…` (GBDT), `21445c6a…` (DeepFM), `2fc04978…` (DIN, attention on),
and the no-attention ablation run.

## What the table actually says

### 1. AUC ranks deep > shallow, as expected.

* LR → GBDT (+0.008): non-linear feature crosses help, even without IDs.
* GBDT → DeepFM (+0.055): learning ID embeddings dominates everything else.
* DeepFM → DIN (+0.035): attention over user history adds another 3.5 pts.

In other words, the **binary CTR task** behaves like the textbook says it
should: every architectural addition (interactions → embeddings → attention)
buys real AUC.

### 2. End-to-end Recall ranks `DeepFM > DIN-no-attn > LR ≈ GBDT > DIN-full`.

This is the W3 finding that took us by surprise and is worth pausing on:

* The model with the highest AUC (DIN-full) has the **lowest** end-to-end
  Recall@10. The model with the second-highest AUC (DeepFM) has the
  **highest** end-to-end Recall@10. The two metrics tell different stories.
* The full DIN learns "history-target similarity ⇒ likely click" extremely
  well on random negatives. But the candidates at end-to-end time are
  **not random** — they're the top-1 000 from the recall layer, every one
  of which already looks similar to the user's history. The attention unit
  thus assigns high score to *many* candidates, and the ground-truth
  positive isn't always the *most* similar one.
* The no-attention ablation is forced to fall back to sum-pooling, which is
  a much *weaker* signal — paradoxically, this lets the model rely more on
  the other features (user_id, item_id, demographics, popularity) and
  generalise better to the hard-negative regime.

### 3. The DeepFM sweet spot

DeepFM sits at the inflection point: deep enough to learn ID embeddings
(good AUC), but not deep enough to over-trust history similarity (good
end-to-end). It's also the cheapest deep model at 0.5 ms / user, half the
latency of DIN-no-attention.

For W3's deliverable — a working pre-ranking model that beats the
baselines on the real ranking task — **DeepFM is what you'd ship**.

## What this implies for W4–W6

1. **Hard-negative mining** is the obvious next step (W6). Train DIN on
   negatives sampled from the recall pool, not uniformly. The literature
   (Pinterest's PinSAGE, Facebook's EBR) shows AUC drops but end-to-end
   Recall climbs significantly.
2. **Listwise / pairwise re-ranking** (W4 ablation) — bring in MMR-style
   diversification *after* DIN's pointwise score, so the model doesn't have
   to make the boring "this is similar to history" call alone.
3. **Larger embedding dim for DIN** — the d=32 we used here is on the
   small side; bumping to d=64 typically buys +0.005 AUC at ≈ 2× memory.
4. **Sequence augmentation** — DIN can be paired with a SASRec-style
   sequence encoder (instead of plain item embeddings) to give the attention
   richer history representations.

## Why we report all four models, even the bad ones

> *Reporting only the best model on the best metric is how you write a press
> release. Reporting all four with their pathologies is how you write a thesis.*

For graduate-school applications, the interesting story is **not** "I built
DIN and got AUC 0.97". The interesting story is "I built four CTR models,
discovered that the model with the best CTR-AUC was the *worst* at
end-to-end Recall, traced it to the recall–rank training mismatch, and
documented the path to fix it." That's W3.
