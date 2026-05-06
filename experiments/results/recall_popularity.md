# Recall — Popularity (heuristic baseline)

The "no learning" baseline: rank items by their training-set interaction
count, return the same global top-K to every user (with already-seen items
filtered out — making the lists slightly different per user).

| field | value |
|---|---|
| Dataset | MovieLens-1M |
| Split | leave-one-out |
| Variant | raw count (`time_decay = false`) |
| Train interactions | 563 204 |
| Test users | 6 034 |
| Top-5 popular items | id 2526 (2 808 hits), 245, 1041, 1043, 563 |
| Fit time | 0.07 s |
| Inference | 0.02 ms / user |
| MLflow run id | `8ac8b55d546f40d89a36ee3b70414cb5` |
| Reproduced on | 2026-05-06 |

## Metrics

| K | Recall | NDCG | HitRate | MRR | Coverage |
|---|---|---|---|---|---|
| 10  | 0.0399 | 0.0188 | 0.0399 | 0.0126 | **0.0326** |
| 50  | 0.1372 | 0.0397 | 0.1372 | 0.0168 | 0.0886 |
| 100 | 0.2367 | 0.0558 | 0.2367 | 0.0182 | 0.1378 |
| 200 | 0.3543 | 0.0722 | 0.3543 | 0.0190 | 0.2131 |

## Why include this

Two reasons that justify the cost of running it:

1. **Sanity floor.** Any learned channel that does not beat popularity by a
   meaningful margin is broken. During the Two-Tower debugging session, the
   model spent ~20 epochs collapsed to popularity-only behavior — having
   the ground-truth popularity numbers in MLflow let us catch this in 30 s
   instead of an hour of guessing.
2. **Cold-start fallback.** When a user has zero training history (a brand
   new account), every learned channel is undefined. Popularity returns
   the *unconditional* top-K and degrades gracefully. The serving pipeline
   uses Popularity as the last-resort merger input.

## The Coverage story

Coverage@10 = **3.3 %** is the most striking number here. Popularity puts
the *same 10 items* in front of (almost) every user, so only 115 of the 3
533 items in the catalog ever get recommended. By contrast iALS reaches
56 % and Two-Tower 56 %. **This is precisely why production systems do not
ship pure popularity** — it kills the long tail.

## How to reproduce

```bash
make preprocess
neorec train recall recall=popularity
# time-decayed variant:
# neorec train recall recall=popularity recall.model.time_decay=true
```
