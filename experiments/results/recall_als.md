# Recall — iALS Baseline

| field | value |
|---|---|
| Dataset | MovieLens-1M |
| Split | leave-one-out (last interaction per user → test) |
| Implicit threshold | rating ≥ 4.0 (label = 1) |
| Train interactions | 563 204 |
| Test users | 6 034 |
| Catalog size | 3 533 items |
| Model | `implicit.als.AlternatingLeastSquares` |
| Factors | 64 |
| Regularization | 0.01 |
| Confidence α | 40.0 |
| Iterations | 20 |
| Seed | 42 |
| Fit time | 2.09 s (single thread) |
| Inference | 0.05 ms / user |
| MLflow run id | `cee3c12efc5548edbd6d4552a2f76ebc` |
| Reproduced on | 2026-04-25 |

## Metrics

| K | Recall | NDCG | HitRate | MRR | Coverage |
|---|---|---|---|---|---|
| 10  | 0.0573 | 0.0274 | 0.0573 | 0.0184 | 0.5627 |
| 50  | 0.2040 | 0.0584 | 0.2040 | 0.0244 | 0.6920 |
| 100 | 0.3286 | 0.0785 | 0.3286 | 0.0262 | 0.7472 |
| 200 | 0.4997 | 0.1025 | 0.4997 | 0.0274 | 0.8242 |

## How to reproduce

```bash
make download
make preprocess
make train-als
make mlflow-ui   # then open http://localhost:5000
```

Or, with explicit overrides:

```bash
neorec train recall recall=als recall.model.factors=64 recall.model.alpha=40 recall.model.iterations=20
```

## Notes

* Recall@K equals HitRate@K because each user has exactly one held-out
  positive in leave-one-out evaluation — they collapse mathematically.
  This is expected behavior, not a bug.
* As a single classical CF baseline, iALS already covers ~56 % of the
  catalog at K=10, which is a strong sanity check that the implicit
  feedback signal is being learned.
* Next steps will benchmark Two-Tower (DSSM) and SASRec; their lift over
  this baseline is the headline number for the Week-2 milestone.
