# DIN — Deep Interest Network for fine-ranking (project headline)

**Model name**: `din`
**Stage**: fine-ranking (100 → 20 in production; in §7.2 we evaluate on 1 000 → 100 for an apples-to-apples comparison with the other rankers)
**Reference**: Zhou, Zhu, Song et al. *Deep Interest Network for Click-Through Rate Prediction.* KDD 2018.

## Architecture

DIN's core idea is **target-aware user representation**: when scoring movie X, only the parts of the user's history that look like X should contribute to the user vector. The mechanism is a local attention unit that takes the target embedding and each history embedding as input:

```
                 ┌──────────── target item embedding t ───────────┐
                 │                                                │
   ┌──── history (B, L, d) ────┐    ┌──── concat per position ────┘
   │ h_1, h_2, …, h_L (left-pad) │
   └─────┬─────────────────────┘
         │
         ▼
   [ h_i ;  t ;  h_i - t ;  h_i ⊙ t ]   →   MLP(4d → 64 → 32 → 1)  →  a_i
                                                                       │
   Σ_i  a_i · h_i  (masked over real positions)  ────────────────►  u_repr
         │
         ▼
   concat( user_id_emb, target_emb, u_repr, side_features, genre_pooled )
                                  │
                                  ▼
                       MLP[200, 80] + PReLU + Dropout → logit
```

Key choices:

* **No softmax on the attention scores** — the paper explicitly argues that
  normalising attention to a probability distribution destroys the
  *magnitude* of user interest. Two history items each with weight 0.6 carry
  more total signal than two items each with weight 0.05; softmax forces
  them to sum to 1.
* **Padding mask** — sequences are left-padded with `SEQ_PAD_VALUE = -1`;
  we clamp to 0 only for the embedding lookup and zero out the row via the
  mask immediately after.
* **Ablation switch** — `cfg.rank.model.use_attention=false` degrades the
  attention unit to sum pooling. Directly quantifies the attention lift.

## Final config

| Hyperparameter | Value | Note |
|---|---:|---|
| `embedding_dim` | 32 | shared item table for target + history |
| `attention_hidden` | (64, 32) | PReLU |
| `dnn_hidden` | (200, 80) | PReLU + dropout 0.3 |
| `max_seq_len` | 50 | left-padded, mask-aware |
| `epochs` | 6 | early-stop patience 2 |
| `batch_size` | 2048 | grad-clip 5.0 |
| `lr` | 1e-3 | Adam, weight_decay 1e-5 |
| `negative_ratio` | 4 | uniform random |
| `use_attention` | true | flip for ablation |

## Results on MovieLens-1M (leave-one-out, 1 000 → 100 re-rank)

| Metric | DIN (full) | DIN-no-attn (ablation) | Δ from attention |
|---|---:|---:|---:|
| **Valid AUC** | **0.9692** | 0.9373 | **+0.0319** |
| Valid LogLoss | 0.1692 | 0.2478 | −0.0786 (better) |
| Valid accuracy | 0.9282 | 0.8914 | +0.037 |
| Recall@10 (end-to-end) | 0.0126 | 0.0336 | **−0.021** (worse!) |
| NDCG@10  (end-to-end) | 0.0055 | 0.0153 | −0.010 |
| Recall@100 (end-to-end) | 0.2126 | **0.3005** | −0.088 |
| Re-rank latency / user | 4.33 ms | 0.94 ms | +3.4 ms |
| Training time | 264 s | 102 s | +162 s |

## The interesting finding: high AUC ≠ high end-to-end Recall

DIN-full has the highest AUC of all four rankers, and the lowest LogLoss by a
wide margin — but its **end-to-end Recall@10 is the worst**. The
no-attention DIN does the opposite: lower AUC, but the best Recall@100.

**Why?** The CTR loss is trained on *random* negatives — items the user has
never seen. The attention unit learns "this target looks similar to recent
history" and assigns high CTR. That makes it excellent at distinguishing
positives from random unseen items.

But at end-to-end evaluation the candidates are not random — they're the
**hard negatives** that the recall layer (ALS + Two-Tower + SASRec) already
ranked highly because they *also* look similar to the user's history. DIN's
attention can't tell the test positive apart from these hard negatives any
more than it can tell two equally-good action movies apart.

This is a well-documented gap in the literature — see e.g. *"Hard negative
mining is all you need"* (CIKM 2022). The fix is **train-time hard-negative
mining**: sample negatives from the recall pool instead of uniformly. We
defer this to W6 as part of the ranking-system optimisation pass.

## What attention *does* help with — the heatmap

`scripts/build_din_attention.py` produces
`experiments/results/ranking/din_attention_heatmap.png` (also embedded in
README §7.2): for 5 users with full 50-item histories, scored against their
real held-out positive, you can see attention concentrates on the handful of
history items that share genre / franchise with the target. That's the
*qualitative* signal we wanted attention to capture, and the heatmap is the
direct visualisation of it.

## When DIN is the right choice

* **Personalised re-ranking inside an already-tight shortlist** (top-20 → top-5).
  At that stage every candidate has been pre-filtered to be relevant, and
  the model only needs to pick the best of equals — exactly the regime
  attention works in.
* **Sequence-aware short-term interest modelling** (e.g. session-based or
  next-item) — DIN's per-target reweighting can capture the user's current
  intent without giving up the demographic / item-side features.
* **Explainability requirement** — the attention weights give you a free
  "why did the model rank X first?" attribution for every prediction.
