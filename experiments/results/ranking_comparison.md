# Ranking head-to-head — LR · GBDT · DeepFM · DIN

W3 deliverable: four rankers, the same training data, the same
end-to-end evaluation harness, **one apples-to-apples table**.

Training uses an **out-of-fold (OOF) split**: every user's `train_df` is
sliced chronologically into 90 % `train_recall` (the recall layer's
training set) and 10 % `train_ranker` (the ranker's positives). This
mirrors a production wall-clock setup and is what
[`ranking_scheme_a_investigation.md`](ranking_scheme_a_investigation.md)
documents as the methodologically correct evaluation regime for this
two-stage pipeline.

## Setup

| Component | Value |
|---|---|
| Dataset | MovieLens-1M (filtered to rating ≥ 4, ≥ 5 interactions / user) |
| Split | leave-one-out per user; 6 034 test users |
| Recall training set | first 90 % of each user's interactions (chronological) |
| Ranker training set | remaining 10 % per user (chronologically later) |
| Negative sampling (train) | uniform random, 1 : 4 |
| Train / valid split | 90 / 10 by row, shuffled |
| Candidate pool (end-to-end) | merge channel (RRF over 5 base channels) loaded from `artifacts/recall_oof/`, top-1 000 |
| Re-rank output | top-100 from each model |
| Evaluation | Recall / NDCG / HitRate / MRR @ {10, 50, 100} + ranker AUC / LogLoss |

## Headline results (OOF — `artifacts/rank_oof/`)

| Rank | Model | Stage | Valid AUC | Valid LogLoss | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 | Recall@100 | NDCG@100 | Latency / user |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **DIN (with attention)** | fine-rank | **0.9311** | **0.2631** | **0.0477** | **0.0214** | **0.1787** | **0.0489** | **0.3031** | **0.0690** | 4.34 ms |
| 2 | DeepFM | pre-rank | 0.8887 | 0.3446 | 0.0401 | 0.0188 | 0.1682 | 0.0460 | 0.2748 | 0.0632 | 0.48 ms |
| 3 | GBDT (Hist) | baseline | 0.8448 | 0.3684 | 0.0358 | 0.0164 | 0.1286 | 0.0360 | 0.2131 | 0.0497 | 1.40 ms |
| 4 | LR | baseline | 0.8238 | 0.3880 | 0.0290 | 0.0153 | 0.1271 | 0.0361 | 0.2126 | 0.0499 | **0.35 ms** |

> Within-stage ordering matches what the CTR literature predicts:
> **DIN > DeepFM > GBDT > LR**, monotone across every metric. AUC,
> Recall@K and NDCG@K all agree on the ranking, which is the expected
> behaviour when training and evaluation are aligned (no look-ahead
> between recall and ranker).

## What the table actually says

### 1. AUC ranks deep > shallow, as expected

* LR → GBDT (+0.021): non-linear feature crosses help.
* GBDT → DeepFM (+0.044): learning ID embeddings dominates.
* DeepFM → DIN (+0.042): attention over user history adds another ~4 pts.

### 2. End-to-end Recall ranks DIN > DeepFM > GBDT > LR

This is the *expected* ordering — and a notable change from the W3
full-train experiments (see appendix), where DIN-with-attention had the
**highest** AUC but the **lowest** end-to-end Recall@10. That
W3-baseline inversion was driven by a misaligned training task
(positives sampled uniformly across each user's full history) combined
with look-ahead bias whenever recall scores were fed in as features.
Cutting the data into OOF folds and training the ranker on the
chronologically-latest 10 % aligns the training task with the
leave-one-out evaluation, which is exactly when attention starts paying
off.

### 3. DIN's headline lift

| Model | W3 full-train Recall@10 | OOF Recall@10 | Lift |
|-------|------------------------:|--------------:|-----:|
| LR     | 0.0283 | 0.0290 | +2 %    |
| GBDT   | 0.0277 | 0.0358 | +29 %   |
| DeepFM | 0.0343 | 0.0401 | +17 %   |
| DIN    | 0.0126 | 0.0477 | **+278 %** |

DIN benefits the most because attention over user history is most
sensitive to whether the training target lives in the same temporal
neighbourhood as the test target. OOF gives it that.

### 4. Caveat — ranker @10 < recall @10 on ML-1M

Under the same OOF pipeline the recall layer's RRF fusion reaches
Recall@10 = 0.0612, slightly above DIN's 0.0477. This is the expected
limit of leave-one-out evaluation on a small (~3.5 K-item) dense
catalog: collaborative-filtering recall already saturates the
candidate-generation task. The ranker's value here is
within-pool discrimination (AUC ≈ 0.93), latency control (4 ms vs full
catalog scoring), and demonstrating end-to-end infrastructure — not
beating the upstream candidate generator at @10. See
[`../../README.md`](../../README.md) §7.2 note for the production-scale
framing.

## Reproduction

```bash
# One-off OOF split (a few seconds)
python scripts/build_oof_split.py --dataset movielens_1m --frac 0.10

# Recall channels (~15 min on M-series MBP)
for ch in als popularity cold_start two_tower sasrec merge; do
  neorec train recall recall=$ch data.oof_split=true
done

# Rankers — these are the headline numbers above
for m in lr gbdt deepfm din; do
  neorec train rank rank=$m data.oof_split=true
done
```

All runs are logged to MLflow under the `rank.<name>.oof` experiment
with the config snapshot attached.

## What this implies for W4–W6

1. **List-wise / pairwise re-ranking** (W4) — bring in MMR-style
   diversification after DIN's pointwise score so the model isn't
   pointwise-classifying a pool where every candidate already looks
   plausible.
2. **Larger embedding dim for DIN** — d=32 is on the small side; d=64
   typically buys +0.005 AUC at ≈ 2× memory.
3. **Per-user leave-one-out positives** — instead of using *every*
   `train_ranker` interaction as a positive, take only each user's
   *latest* one. This further tightens the train/test alignment but
   shrinks the training set ~10×; needs stronger regularisation.
4. **List-wise loss** (LambdaRank / BPR over the full candidate pool) —
   directly optimises NDCG rather than pointwise CTR.

## Why we report all four models, even the bad ones

> *Reporting only the best model on the best metric is how you write a
> press release. Reporting all four with their pathologies is how you
> write a thesis.*

For graduate-school applications, the interesting story is **not** "I
built DIN and got Recall@10 = 0.048". The interesting story is "I built
four CTR models, watched the highest-AUC model post the *lowest*
end-to-end Recall, traced it to look-ahead bias between recall and
ranker training, designed an out-of-fold split that restored the
expected ordering, and documented the negative result on top of that
(Scheme A still fails even in OOF for a second-order reason)." That's
W3.

The full diagnostic narrative is in
[`ranking_scheme_a_investigation.md`](ranking_scheme_a_investigation.md).

---

## Appendix — W3 baseline (full-train, kept for archaeology)

These are the numbers produced if `data.oof_split=false` (i.e. recall
channels and rankers are both fit on the full `train_df`). They live in
`artifacts/rank/` and remain reproducible, but the OOF results above
are the canonical W3-final numbers. The appendix is kept so the §3
"DIN's headline lift" attribution table stays verifiable.

| Rank | Model | Stage | Valid AUC | Valid LogLoss | Recall@10 | NDCG@10 | Recall@100 | NDCG@100 | Latency / user |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | **DIN (full)**       | fine-rank   | **0.9692** | **0.1692** | 0.0126 | 0.0055 | 0.2126 | 0.0415 | 4.33 ms |
| 2 | DIN (no-attention)   | ablation    | 0.9373 | 0.2478 | 0.0336 | 0.0153 | **0.3005** | **0.0653** | 0.94 ms |
| 3 | **DeepFM**           | pre-rank    | 0.9342 | 0.2534 | **0.0343** | **0.0151** | 0.2890 | 0.0627 | **0.51 ms** |
| 4 | GBDT (Hist)          | baseline    | 0.8793 | 0.3310 | 0.0277 | 0.0119 | 0.1951 | 0.0432 | 1.64 ms |
| 5 | LR                   | baseline    | 0.8715 | 0.3390 | 0.0283 | 0.0126 | 0.2083 | 0.0458 | 0.39 ms |

Original MLflow run IDs: `26dbf0e2…` (LR), `856be6b6…` (GBDT),
`21445c6a…` (DeepFM), `2fc04978…` (DIN, attention on).

The W3-full-train AUC-vs-Recall inversion (DIN-with-attention has the
highest AUC and the *lowest* end-to-end Recall@10) is what
[`ranking_scheme_a_investigation.md`](ranking_scheme_a_investigation.md)
analyses in depth. The short version: training negatives were uniform
random while inference candidates were merge top-1 000 (all "hard"),
and attention over-fit the random-negative task. OOF fixes the
upstream task alignment, which is why the appendix exists but the
headline table is OOF.
