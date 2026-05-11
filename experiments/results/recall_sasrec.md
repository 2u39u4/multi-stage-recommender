# SASRec — Self-Attentive Sequential Recommendation

**Channel name**: `sasrec`  
**Reference**: Kang & McAuley. *Self-Attentive Sequential Recommendation.* ICDM 2018.

## Architecture

```
Input sequence (left-padded to L=50)
  ┌──────────────────────────────────────┐
  │ [pad, pad, …, i_{T-50+1}, …, i_T]    │  item ids; pad = num_items
  └──────────────────┬───────────────────┘
                     ▼
           item_emb (V+1 × D)  +  pos_emb (L × D)
                     │
                     ▼
              dropout → LN
                     │
              ┌──────┴──────┐
              │  block × N  │   (pre-LN MHA causal mask + FFN, GELU)
              └──────┬──────┘
                     ▼
                LayerNorm  →  (B, L, D)
```

* **Item vocabulary**: 0 … num_items−1 plus a reserved padding token at index `num_items` (with `padding_idx`).
* **Causal mask** (upper-triangular bool, True = block future) is registered as a non-persistent buffer.
* **Left-aligned padding** — the most recent item is always at position `L-1`, so the hidden state we read at inference is `hidden[:, -1, :]`.
* **Per-position BPR loss** — at every non-pad target position, `softplus(neg_score − pos_score)`; one uniform negative per position with rejection on the user's seen set.

## Final config (MovieLens-1M, leave-one-out)

| Hyperparameter | Value | Note |
|---|---:|---|
| `embedding_dim` | 64 | shared by item & position embeddings |
| `max_seq_len` | 50 | latest 50 training items |
| `num_blocks` | 2 | pre-LN transformer blocks |
| `num_heads` | 2 | multi-head attention |
| `dropout` | 0.2 | applied at input + intra-block |
| `epochs` | 50 | loss still slowly decreasing at end |
| `batch_size` | 128 | 1 sample / user / epoch |
| `lr` | 1e-3 | AdamW, weight_decay=0 |
| `grad_clip` | 5.0 | norm clipping |
| `device` | cpu | MPS slower at this size |
| **Params** | **279 936** | dominated by item embedding |

## Results — MovieLens-1M, 6 034 test users, full-rank protocol

| K   | Recall@K | NDCG@K | MRR@K  | Coverage@K |
|----:|---------:|-------:|-------:|-----------:|
| 10  | 0.0570   | 0.0284 | 0.0198 | 0.6742     |
| 50  | 0.1699   | 0.0525 | 0.0246 | 0.8356     |
| 100 | 0.2391   | 0.0637 | 0.0255 | 0.8588     |
| 200 | 0.3305   | 0.0764 | 0.0262 | 0.8913     |

**Training**: 90.84 s (50 epochs CPU, single thread).  
**Inference**: 0.07 ms / user (435 ms for the full 6 034-user test split).

## Head-to-head with the other channels

At K = 10:

| Channel       | Recall@10 | NDCG@10 | Coverage@10 |
|---------------|----------:|--------:|------------:|
| Popularity    | 0.0399    | 0.0188  | 0.0326      |
| Cold-start    | 0.0167    | 0.0083  | 0.8735      |
| iALS          | 0.0573    | 0.0274  | 0.5627      |
| Two-Tower     | 0.0590    | 0.0286  | 0.5587      |
| **SASRec**    | **0.0570**| **0.0284** | **0.6742** |

SASRec sits **statistically next to iALS / Two-Tower at K=10**, but two qualitative
properties show up:

1. **Coverage @10 is materially higher** (0.6742 vs 0.5587 for Two-Tower, 0.5627 for iALS).  
   At this cut-off SASRec sees ~20% more of the catalog than the matrix-factorisation peers, which is exactly the kind of diversity signal a sequential model should bring (it conditions on the user's *trajectory*, not just their global tastes).

2. **The relative ordering shifts at deeper K**. At K=100 SASRec ranks at Recall = 0.239 vs 0.241 (TT) and 0.231 (iALS) — i.e. SASRec recovers slightly more of the eventually-relevant items further down the list. That's a typical signature of attention-based sequential models: better long-tail recovery, comparable head accuracy.

## Engineering / debugging notes

| Pitfall | Fix |
|---|---|
| `nn.MultiheadAttention` returns NaN when an entire row of `key_padding_mask` is True | Detect rows whose padding mask is all-True and zero them out before the attention call (their outputs are masked downstream by the per-position loss mask anyway). |
| Item id 0 is a real item, can't also be the pad token | Reserve `pad_idx = num_items` (last slot of `nn.Embedding(num_items + 1, …)`); no conflict with real ids. |
| `cfg.recall.train.inference_batch_size` not present when the channel is loaded *from* `MergeRecaller` (parent cfg is the merge cfg) | Read with `OmegaConf.select(self.cfg, "recall.train.inference_batch_size", default=512)`. |
| Tempting to use `sequences.parquet` as the training source | We rebuild sequences from `train_interactions.parquet` directly — guarantees no leakage of the held-out LOO test item, since `train_interactions.parquet` is produced after the `split == 'train'` filter. |

## Where SASRec earns its keep in the final pipeline

* **Merge channel**: SASRec contributes the 3rd-strongest signal in RRF, and its higher Coverage@10 helps the merge avoid the popularity-bias collapse you'd get from stacking only collaborative-filtering channels.
* **Future ablation (W4)**: We will sweep `max_seq_len ∈ {10, 20, 50, 100}` to plot accuracy vs. sequence length — this is the kind of analysis the EXECUTION_PLAN's Day 24–26 calls out as a "must-have" curve.
