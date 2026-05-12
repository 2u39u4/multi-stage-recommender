# DeepFM — factorisation-machine + deep tower for pre-ranking

**Model name**: `deepfm`
**Stage**: pre-ranking (1 000 → 100)
**Reference**: Guo, Tang, Ye, Li, He. *DeepFM: A Factorization-Machine based
Neural Network for CTR Prediction.* IJCAI 2017.

## Architecture

```
   ┌─ user_id ─┐    ┌─ item_id ─┐  ┌─ gender ─┐  ┌─ age_bucket ─┐ …
   │ embedding │    │ embedding │  │ embedding│  │ embedding    │ …
   └────┬──────┘    └────┬──────┘  └────┬─────┘  └─────┬────────┘
        ▼                ▼              ▼              ▼
        concat → (B, K=8, d=16)  ←— K = #sparse fields + 1 (genre mean)
        │
        ├── FM 1st-order   (Σ scalar lookups + bias)
        ├── FM 2nd-order   ½ · Σ_d ((Σ_i e_i^d)² − Σ_i (e_i^d)²)        ← O(K·d)
        └── Deep tower     concat(K·d) → MLP[256, 128, 64] → 1
                                ▲
                                └─ shares the same embeddings as FM

   logit = bias + order1 + order2 + dnn_out
   p     = σ(logit)
```

Key design choices:

* **Pure PyTorch implementation** — we own the optimiser, can run an FM-only
  or Deep-only ablation by flipping `cfg.rank.model.use_fm` / `use_deep`, and
  ship without `deepctr-torch` as a dependency.
* **Shared embeddings** — the FM 2nd-order block and the Deep tower read
  from the *same* `nn.Embedding` tables. This is what DeepFM's paper
  emphasises as the model's main advantage over Wide & Deep.
* **Genre multi-hot → mean pooling** — items have 1–6 genre tags; we
  mean-pool their embeddings into a single (`B`, `d`) vector before adding
  it to the FM/DNN inputs.
* **BCE loss** with random negatives (1 : 4) — same training data as every
  other ranker in §7.2 for an apples-to-apples comparison.

## Final config

| Hyperparameter | Value | Note |
|---|---:|---|
| `embedding_dim` | 16 | shared by FM + DNN |
| `dnn_hidden` | (256, 128, 64) | ReLU + dropout 0.3 |
| `use_fm` / `use_deep` | true / true | both branches active |
| `epochs` | 8 | early-stop patience 2 |
| `batch_size` | 4096 | inference batch 16 384 |
| `lr` | 1e-3 | Adam, weight_decay 1e-5 |
| `negative_ratio` | 4 | uniform random |

## Results on MovieLens-1M (leave-one-out)

| Metric | Value |
|---|---:|
| **Valid AUC** | **0.9342** |
| **Valid LogLoss** | **0.2534** |
| Valid accuracy | 0.8885 |
| Recall@10 (end-to-end) | **0.0343** |
| NDCG@10  (end-to-end) | **0.0151** |
| Recall@100 (end-to-end) | **0.2890** |
| Re-rank latency / user | 0.51 ms |
| Training time | ~120 s on CPU |

## What the numbers tell us

* **+5.5 pts AUC over GBDT** confirms the value of learning ID embeddings:
  by giving each user and item its own learned representation, DeepFM can
  do CF interactions that no tabular tree could fake from side features.
* Among the four rankers it is the **best end-to-end model**: Recall@10
  0.0343 / Recall@100 0.289 are both ≈ 25 % above the closest baseline.
  This is the sweet spot for pre-ranking: it expresses CF, runs in 0.5 ms,
  and saves to a 1.2 MB file.
* End-to-end Recall@10 is still below the raw recall-layer output (0.079)
  because the model is trained on random negatives, not hard negatives from
  the recall pool. See the comparison report for the full discussion.

## Failure-mode taxonomy (debugging hooks)

* If `valid_pos_rate` deviates from `1 / (1+negative_ratio) = 0.20`, the
  upstream negative sampler has a bug — re-run `build_training_pairs` with
  a fixed seed and dump the head.
* If `tr_loss` plateaus at ≈ 0.50 / 0.45, the embeddings are stuck at init
  scale; increase `init_std` from 1e-2 to 1e-1.
* If `va_loss` drops below `tr_loss` after epoch 3 the dropout is too low —
  bump `dnn_dropout` from 0.3 to 0.5.

## Production hooks

* `cfg.rank.model.device=cuda` is honoured if a GPU is available; the rest
  of the training loop is GPU-clean (no tensors stuck on CPU).
* `score(user_ids, item_ids)` is fully vectorised — re-ranking 1 000
  candidates for 6 K users takes 3 s total (≈ 0.5 ms / user).
