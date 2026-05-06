# Recall — Two-Tower (DSSM, BPR loss)

| field | value |
|---|---|
| Dataset | MovieLens-1M |
| Split | leave-one-out (last interaction per user → test) |
| Implicit threshold | rating ≥ 4.0 (label = 1) |
| Train interactions | 563 204 |
| Test users | 6 034 |
| Catalog size | 3 533 items |
| Architecture | shared item embedding · user_id_emb ⊕ seq_mean · item_id_emb ⊕ genre_mean · per-item bias |
| Embedding dim | 64 |
| Sequence length | up to 50 (left-padded, masked-mean pooled) |
| Loss | BPR — `softplus(s_neg − s_pos)` |
| Negative sampling | uniform over items, rejection on user-seen set |
| Optimizer | AdamW (lr = 1e-3, wd = 1e-5, grad-clip 5.0) |
| Batch size | 2 048 |
| Epochs | 80 |
| Total params | 617 037 |
| Fit time | 776 s (≈ 13 min, single-thread CPU) |
| Inference | 0.04 ms / user |
| Device | CPU (MPS slower for tiny ops on this size) |
| MLflow run id | `5b3d66a66a11480fbe3594683d1896a7` |
| Reproduced on | 2026-05-06 |

## Metrics

| K | Recall | NDCG | HitRate | MRR | Coverage |
|---|---|---|---|---|---|
| 10  | **0.0590** | **0.0286** | 0.0590 | 0.0195 | 0.5587 |
| 50  | 0.2095 | 0.0607 | 0.2095 | 0.0258 | 0.7843 |
| 100 | 0.3348 | 0.0809 | 0.3348 | 0.0276 | 0.8789 |
| 200 | 0.4914 | 0.1027 | 0.4914 | 0.0287 | 0.9454 |

## Lift over iALS baseline

| Metric | iALS | Two-Tower | Δ |
|---|---|---|---|
| Recall@10 | 0.0573 | 0.0590 | **+2.9 %** |
| NDCG@10 | 0.0274 | 0.0286 | **+4.5 %** |
| MRR@10 | 0.0184 | 0.0195 | **+6.2 %** |
| Recall@50 | 0.2040 | 0.2095 | **+2.7 %** |
| NDCG@50 | 0.0584 | 0.0607 | **+3.9 %** |
| Recall@100 | 0.3286 | 0.3348 | **+1.9 %** |
| Recall@200 | 0.4997 | 0.4914 | −1.7 % |

> **Headline**: Two-Tower beats iALS at the K values that matter for top-K
> serving (10/50/100). At K=200 they are statistically tied — iALS slightly
> ahead, which is consistent with classical CF being very strong at large
> recall budgets.

## How to reproduce

```bash
make download
make preprocess
neorec train recall recall=two_tower
# or override:
# neorec train recall recall=two_tower recall.train.epochs=80
```

## Engineering notes

This was the first deep-learning model in the project and we hit a
*classic* set of "two-tower won't train" issues before landing on the
BPR + uniform-negative recipe. Things tried (and discarded), in order:

1. **In-batch sampled-softmax + L2-norm + cosine + τ=0.05** — never
   escaped uniform-random outputs. Loss dropped only ≈0.3 nat in 5 ep.
   Gradient through `F.normalize` is small at random init; with B=512 and
   cold τ, the softmax was effectively a delta on noise.
2. **Higher LR + larger batch + τ=0.3** — exploded (epoch-2 loss = 9 e27).
   Adam can't keep up with the 1/τ amplification when logits are
   unbounded.
3. **Item bias term + raw dot product + in-batch CE** — model
   collapsed to *pure popularity*: `b_i` learned the marginal distribution
   and `u·v` ≈ 0 for all pairs. Coverage@10 stayed at 99 % (uniform).
4. **BPR with explicit uniform negatives**  ← landed. Loss drops
   monotonically (0.63 → 0.14 over 80 ep), no instability, beats both
   popularity and iALS.

Two minor optimizations brought epoch time from 17 s on MPS to 10 s on
CPU:

* All sequence / genre lookup tables pre-built as `np.ndarray` so
  `__getitem__` is constant-time slicing — no per-call list filtering.
* Single `encode_user` call per batch shared between positive and
  negative scoring (was being computed twice in early drafts of the BPR
  loop).

## Notes

* Recall@K and HitRate@K are mathematically identical under leave-one-out
  (1 held-out positive per user) — both shown for cross-comparability.
* All results are reproducible end-to-end via `make`. The 80-epoch BPR
  run is deterministic up to numpy / pytorch seed (`cfg.seed=42`).
* The checkpoint and pre-computed user / item vectors live in
  `artifacts/recall/two_tower/` — total 4.6 MB.

## Next steps

* **Hard negative mining** (cf. Yi et al. 2019) — sample negatives from
  the top-K predicted but unwatched items, expected lift another ~3–5 %.
* **SASRec sequence model** — replace masked-mean pooling with a
  causal Transformer encoder; literature suggests this should give
  another +50–100 % over Two-Tower.
* **Multi-channel fusion** — combine ALS / Popularity / Two-Tower via
  RRF or learned weights; gains tend to compound.
