"""Driver for the six W4 ablations.

Usage
-----
    python scripts/run_ablations.py mmr_lambda
    python scripts/run_ablations.py cold_start_bucket
    python scripts/run_ablations.py fusion_strategy
    python scripts/run_ablations.py din_attention
    python scripts/run_ablations.py sasrec_seq_len
    python scripts/run_ablations.py two_tower_neg
    python scripts/run_ablations.py all

Each ablation writes a JSON results file under ``experiments/ablations/``
that notebooks read for plotting. Heavy retraining (SASRec, Two-Tower,
DIN-no-attention) is gated behind explicit subcommands so the cheap
sweeps remain fast to iterate on.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = (REPO_ROOT / "configs").as_posix()
OUT_DIR = REPO_ROOT / "experiments" / "ablations"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, (REPO_ROOT / "src").as_posix())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("ablations")


def _compose(overrides: list[str]):
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        return compose(config_name="config", overrides=overrides)


# ===========================================================================
# A) MMR λ sweep — fast, reuses trained DIN
# ===========================================================================
def ablation_mmr_lambda(lambdas: list[float] | None = None) -> dict:
    from neorec.rerank.pipeline import run as rerank_run

    lambdas = lambdas or [0.0, 0.3, 0.5, 0.7, 1.0]
    out = {"lambdas": lambdas, "runs": []}
    for lam in lambdas:
        log.info("=== MMR λ=%.2f ===", lam)
        cfg = _compose([
            "rank=din",
            "rerank=mmr",
            "data.oof_split=true",
            "rerank.strategy=mmr",
            f"rerank.mmr.lambda={lam}",
            "rerank.output.top_k_final=10",
        ])
        metrics = rerank_run(cfg)
        out["runs"].append({"lambda": lam, "metrics": metrics})
    (OUT_DIR / "mmr_lambda.json").write_text(json.dumps(out, indent=2))
    log.info("Saved MMR λ sweep to %s", OUT_DIR / "mmr_lambda.json")
    return out


# ===========================================================================
# B) Cold-start bucket — re-bucket existing OOF DIN predictions
# ===========================================================================
def ablation_cold_start_bucket() -> dict:
    """Bucket test users by training-history length and compute Recall@10
    per bucket using the cached DIN OOF rerank output.

    Bucket boundaries (interactions in train_recall + train_ranker):
        [5, 20)  cold
        [20, 60) warm
        [60, ∞)  hot
    """
    from neorec.eval.metrics import (
        coverage,
        mean_reciprocal_rank,
        ndcg_at_k,
        recall_at_k,
    )

    processed = REPO_ROOT / "data" / "processed" / "movielens_1m"
    split = pd.read_parquet(processed / "oof_split.parquet")
    train = split[split["split"].isin(["train_recall", "train_ranker"])]
    counts = train.groupby("user_id").size().rename("n_train")

    rerank_dir = REPO_ROOT / "artifacts" / "rerank" / "din_mmr"
    if not (rerank_dir / "predictions.parquet").exists():
        raise FileNotFoundError(
            f"Need {rerank_dir/'predictions.parquet'} — run 'mmr_lambda' first "
            "(λ=0.7 is fine) or `neorec rerank rank=din rerank=mmr ...`."
        )
    preds = pd.read_parquet(rerank_dir / "predictions.parquet")
    preds = preds.merge(counts, on="user_id", how="left").fillna(0)

    def bucket_of(n: int) -> str:
        if n < 20:
            return "cold (<20)"
        if n < 60:
            return "warm (20-59)"
        return "hot (60+)"

    preds["bucket"] = preds["n_train"].astype(int).map(bucket_of)
    rows = []
    for bk, group in preds.groupby("bucket"):
        y_true = [[int(t)] for t in group["truth"].tolist()]
        y_pred = [json.loads(s) for s in group["rerank_topk"].tolist()]
        rows.append({
            "bucket":     bk,
            "n_users":    int(len(group)),
            "recall@10":  recall_at_k(y_true, y_pred, k=10),
            "ndcg@10":    ndcg_at_k(y_true, y_pred, k=10),
            "mrr@10":     mean_reciprocal_rank(y_true, y_pred, k=10),
            "coverage@10": coverage(y_pred, catalog_size=3533, k=10),
        })
    out = {"buckets": rows}
    (OUT_DIR / "cold_start_bucket.json").write_text(json.dumps(out, indent=2))
    log.info("Saved cold-start bucket analysis to %s", OUT_DIR / "cold_start_bucket.json")
    return out


# ===========================================================================
# C) Recall fusion strategy — RRF vs norm_weighted vs single channels
# ===========================================================================
def ablation_fusion_strategy() -> dict:
    """Evaluate single channels + 2 fusion strategies on the full-train recall
    set (matches §7.1 in the README). Numbers come straight from the merge
    recaller's RecallResult vs the held-out test set.
    """
    from neorec.eval.metrics import (
        coverage,
        mean_reciprocal_rank,
        ndcg_at_k,
        recall_at_k,
    )
    from neorec.recall.merge import MergeRecaller, merge_norm_weighted, merge_rrf

    repo = REPO_ROOT
    processed = repo / "data" / "processed" / "movielens_1m"
    split = pd.read_parquet(processed / "split.parquet")
    test = split.query("split == 'test'").reset_index(drop=True)
    user_ids = test["user_id"].tolist()
    truths = [[int(x)] for x in test["item_id"].tolist()]

    # Compose a recall=merge cfg in full-train mode so single-channel
    # numbers stay comparable to README §7.1.
    cfg = _compose(["recall=merge", "data.oof_split=false"])
    OmegaConf.set_struct(cfg, False)

    recaller = MergeRecaller(cfg)
    recaller.fit("")

    # First, harvest each base channel's top-1000 by calling them through
    # MergeRecaller.recall() with strategies that disable cross-channel mixing.
    # Easier path: temporarily replace cfg.recall.channels with one enabled.
    enabled_names = [n for n, ch in cfg.recall.channels.items() if bool(ch.enabled)]
    single_results: dict[str, dict] = {}
    for name in enabled_names:
        rec = recaller._channels.get(name)
        if rec is None:
            continue
        log.info("Evaluating single channel: %s …", name)
        res = rec.recall(user_ids, k=1000)
        y_pred = [
            [int(x) for x in row if x >= 0]
            for row in res.item_ids.tolist()
        ]
        single_results[name] = {
            "recall@10":   recall_at_k(truths, y_pred, k=10),
            "ndcg@10":     ndcg_at_k(truths, y_pred, k=10),
            "mrr@10":      mean_reciprocal_rank(truths, y_pred, k=10),
            "coverage@10": coverage(y_pred, catalog_size=3533, k=10),
        }

    # Fusion strategies (use top-K per channel = 500 to match production).
    per_channel = []
    for name in enabled_names:
        per_channel.append(recaller._channels[name].recall(user_ids, k=500))

    out: dict[str, dict] = {"single": single_results, "fusion": {}}
    for strat, kwargs in [
        ("rrf", {"k_rrf": 60}),
        ("norm_weighted", {"weights": {"als": 1.0, "two_tower": 1.0, "sasrec": 1.0, "popularity": 0.5, "cold_start": 0.5}}),
    ]:
        log.info("Evaluating fusion strategy: %s …", strat)
        fuser = merge_rrf if strat == "rrf" else merge_norm_weighted
        fused = fuser(per_channel, candidate_pool_size=1000, **kwargs)
        y_pred = [
            [int(x) for x in row if x >= 0]
            for row in fused.item_ids.tolist()
        ]
        out["fusion"][strat] = {
            "recall@10":   recall_at_k(truths, y_pred, k=10),
            "ndcg@10":     ndcg_at_k(truths, y_pred, k=10),
            "mrr@10":      mean_reciprocal_rank(truths, y_pred, k=10),
            "coverage@10": coverage(y_pred, catalog_size=3533, k=10),
        }

    (OUT_DIR / "fusion_strategy.json").write_text(json.dumps(out, indent=2))
    log.info("Saved fusion strategy comparison to %s", OUT_DIR / "fusion_strategy.json")
    return out


# ===========================================================================
# Helpers for shelling out to `neorec train …`
# ===========================================================================
def _shell_train(overrides: list[str], log_name: str) -> dict:
    """Invoke neorec train ... in a subprocess and parse the printed metrics."""
    log_path = OUT_DIR / f"{log_name}.log"
    log.info("Running: neorec train %s (log: %s)", " ".join(overrides), log_path)
    t0 = time.time()
    env = os.environ.copy()
    env.setdefault("PROJECT_ROOT", REPO_ROOT.as_posix())
    cmd = [sys.executable, "-m", "neorec.cli", "train", *overrides[:1], *overrides[1:]]
    # neorec CLI groups train.recall vs train.rank — overrides[0] is the subcommand.
    with open(log_path, "w") as f:
        result = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=REPO_ROOT.as_posix(),
        )
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(f"Training failed (rc={result.returncode}); see {log_path}")
    log.info("  → completed in %.1f s", elapsed)
    return {"elapsed_sec": elapsed, "log": str(log_path)}


def _scrape_final_metrics(log_path: Path, marker: str = "End-to-end metrics") -> dict | None:
    """Pull the dict-shaped metrics line out of a train log."""
    text = log_path.read_text()
    for line in reversed(text.splitlines()):
        if marker in line:
            try:
                start = line.index("{")
                return json.loads(line[start:].replace("'", '"'))
            except (ValueError, json.JSONDecodeError):
                return None
    return None


# ===========================================================================
# D) DIN attention vs sum-pool (OOF)
# ===========================================================================
def _train_din_in_process(use_attention: bool) -> dict:
    """Drive ranking.train.run() programmatically to get metrics directly."""
    from neorec.ranking.train import run as rank_run

    cfg = _compose([
        "rank=din",
        "data.oof_split=true",
        f"rank.model.use_attention={'true' if use_attention else 'false'}",
    ])
    return rank_run(cfg)


def ablation_din_attention() -> dict:
    """Retrain DIN with use_attention={true, false} on OOF.

    Backs the W3 final ``artifacts/rank_oof/din`` aside before each training
    run, captures its end-to-end metrics, and restores the attention=true
    artefact at the end so the rest of the project sees the canonical model.
    """
    din_dir = REPO_ROOT / "artifacts" / "rank_oof" / "din"
    backup = REPO_ROOT / "artifacts" / "rank_oof" / "din.w3_final.bak"
    if din_dir.exists():
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(din_dir, backup)
        log.info("Backed up W3 DIN artefacts to %s", backup)

    results = {}
    try:
        for use_attn in [True, False]:
            tag = "attn" if use_attn else "sum"
            log.info("=== DIN ablation: use_attention=%s ===", use_attn)
            metrics = _train_din_in_process(use_attention=use_attn)
            results[tag] = {"metrics": metrics}
            # Snapshot the trained artefact so subsequent rerank notebooks can
            # use either flavour by pointing to the right directory.
            snap = REPO_ROOT / "artifacts" / "rank_oof" / f"din_{tag}"
            if snap.exists():
                shutil.rmtree(snap)
            shutil.copytree(din_dir, snap)
            log.info("Snapshotted DIN(%s) → %s", tag, snap)
    finally:
        if backup.exists():
            if din_dir.exists():
                shutil.rmtree(din_dir)
            shutil.copytree(backup, din_dir)
            log.info("Restored W3 DIN artefacts from %s", backup)

    (OUT_DIR / "din_attention.json").write_text(json.dumps(results, indent=2))
    log.info("Saved DIN attention ablation to %s", OUT_DIR / "din_attention.json")
    return results


# ===========================================================================
# Helpers — backup / restore artefact directory, then run recall.train in-proc
# ===========================================================================
def _backup_dir(src: Path, label: str) -> Path | None:
    if not src.exists():
        return None
    bak = src.parent / f"{src.name}.{label}.bak"
    if bak.exists():
        shutil.rmtree(bak)
    shutil.copytree(src, bak)
    log.info("Backed up %s → %s", src, bak)
    return bak


def _restore_dir(src: Path, bak: Path | None) -> None:
    if bak is None:
        return
    if src.exists():
        shutil.rmtree(src)
    shutil.copytree(bak, src)
    log.info("Restored %s ← %s", src, bak)


def _run_recall_in_process(overrides: list[str]) -> dict:
    from neorec.recall.train import run as recall_run

    cfg = _compose(overrides)
    return recall_run(cfg)


# ===========================================================================
# E) SASRec sequence length sweep
# ===========================================================================
def ablation_sasrec_seq_len(lengths: list[int] | None = None) -> dict:
    lengths = lengths or [10, 20, 50, 100]
    art = REPO_ROOT / "artifacts" / "recall" / "sasrec"
    bak = _backup_dir(art, "w2_final")
    runs = []
    try:
        for L in lengths:
            log.info("=== SASRec max_seq_len=%d ===", L)
            metrics = _run_recall_in_process([
                "recall=sasrec",
                "data.oof_split=false",
                f"recall.model.max_seq_len={L}",
            ])
            runs.append({"max_seq_len": L, "metrics": metrics})
            snap = art.parent / f"sasrec_L{L}"
            if snap.exists():
                shutil.rmtree(snap)
            shutil.copytree(art, snap)
    finally:
        _restore_dir(art, bak)
    out = {"runs": runs}
    (OUT_DIR / "sasrec_seq_len.json").write_text(json.dumps(out, indent=2))
    log.info("Saved SASRec seq-len sweep to %s", OUT_DIR / "sasrec_seq_len.json")
    return out


# ===========================================================================
# F) Two-Tower capacity sweep
# ---------------------------------------------------------------------------
# Plan asked for a num_negatives sweep, but the Two-Tower trainer uses the
# canonical single-negative BPR loss (Rendle 2009): one triplet per positive
# regardless of `num_negatives`. We swap in **embedding_dim** instead — a real
# capacity knob the code supports directly. The lesson is the same shape:
# return-on-capacity, with diminishing or negative returns past a threshold.
# Notebook + W4 docs label this clearly.
# ===========================================================================
def ablation_two_tower_neg(
    dims: list[int] | None = None,
    epochs: int = 40,
) -> dict:
    dims = dims or [16, 32, 64, 128]
    art = REPO_ROOT / "artifacts" / "recall" / "two_tower"
    bak = _backup_dir(art, "w2_final")
    runs = []
    try:
        for d in dims:
            log.info("=== Two-Tower embedding_dim=%d ===", d)
            metrics = _run_recall_in_process([
                "recall=two_tower",
                "data.oof_split=false",
                f"recall.model.embedding_dim={d}",
                f"recall.train.epochs={epochs}",
            ])
            runs.append({"embedding_dim": d, "metrics": metrics})
            snap = art.parent / f"two_tower_d{d}"
            if snap.exists():
                shutil.rmtree(snap)
            shutil.copytree(art, snap)
    finally:
        _restore_dir(art, bak)
    out = {"runs": runs, "note": "embedding_dim substituted for num_negatives — see W4 report"}
    (OUT_DIR / "two_tower_neg.json").write_text(json.dumps(out, indent=2))
    log.info("Saved Two-Tower capacity sweep to %s", OUT_DIR / "two_tower_neg.json")
    return out


# ===========================================================================
# Main
# ===========================================================================
ALL = {
    "mmr_lambda":         ablation_mmr_lambda,
    "cold_start_bucket":  ablation_cold_start_bucket,
    "fusion_strategy":    ablation_fusion_strategy,
    "din_attention":      ablation_din_attention,
    "sasrec_seq_len":     ablation_sasrec_seq_len,
    "two_tower_neg":      ablation_two_tower_neg,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ablation", choices=list(ALL) + ["all"])
    args = p.parse_args()
    targets = list(ALL) if args.ablation == "all" else [args.ablation]
    for name in targets:
        log.info(">>> Running ablation: %s", name)
        ALL[name]()


if __name__ == "__main__":
    main()
