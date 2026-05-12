"""Build the recall → pre-rank → fine-rank → rerank conversion funnel.

For every test user (LOO), tracks whether the test positive item survives
each stage of the pipeline. The output JSON is consumed by
`notebooks/04_funnel_conversion.ipynb` to draw the Sankey diagram.

Stages (OOF artefacts unless noted):
    Stage 0 — merge top-1000  (recall fusion)
    Stage 1 — DeepFM top-100  (pre-rank)
    Stage 2 — DIN top-20      (fine-rank, OOF)
    Stage 3 — MMR top-10      (rerank with λ=0.7, default)

Run:
    python scripts/build_funnel.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = (REPO_ROOT / "configs").as_posix()
sys.path.insert(0, (REPO_ROOT / "src").as_posix())

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("funnel")


def main() -> None:
    from neorec.ranking.features import RankingFeaturizer
    from neorec.ranking.train import _build_merge_candidates, _instantiate, _load_data
    from neorec.rerank.mmr import mmr_rerank

    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="config",
            overrides=[
                "rank=din",
                "rerank=mmr",
                "data.oof_split=true",
            ],
        )
    OmegaConf.set_struct(cfg, False)

    processed, train_history_df, ranker_positives_df, test_df = _load_data(cfg)
    featurizer = RankingFeaturizer(
        processed_dir=processed,
        max_genres=int(cfg.rank.input.get("max_genres", 6)),
        max_seq_len=int(cfg.rank.input.get("max_seq_len", 50)),
    )
    featurizer.build_sequences(train_history_df)

    # ---- Stage 0: merge top-1000 -----------------------------------------
    extra_seen: dict[int, set[int]] = defaultdict(set)
    for u, i in zip(
        ranker_positives_df["user_id"].to_numpy(),
        ranker_positives_df["item_id"].to_numpy(),
    ):
        extra_seen[int(u)].add(int(i))
    candidates = _build_merge_candidates(
        cfg=cfg,
        test_user_ids=test_df["user_id"].tolist(),
        pool_size=1000,
        extra_user_seen=extra_seen,
    )

    # ---- Load DeepFM (pre-rank) and DIN (fine-rank) ----------------------
    deepfm_dir = REPO_ROOT / "artifacts" / "rank_oof" / "deepfm"
    din_dir = REPO_ROOT / "artifacts" / "rank_oof" / "din"
    log.info("Loading rankers from %s and %s", deepfm_dir, din_dir)

    deepfm_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))  # deep copy
    deepfm_cfg.rank = OmegaConf.load(REPO_ROOT / "configs" / "rank" / "deepfm.yaml")
    deepfm = _instantiate("deepfm", deepfm_cfg, featurizer)
    deepfm.load(deepfm_dir)

    din = _instantiate("din", cfg, featurizer)
    din.load(din_dir)

    # ---- Load item embeddings for MMR ------------------------------------
    iv = np.load(REPO_ROOT / "artifacts" / "recall_oof" / "two_tower" / "item_vecs.npy").astype(np.float32)
    iv = iv / np.clip(np.linalg.norm(iv, axis=1, keepdims=True), 1e-8, None)

    # ---- Walk through every test user -----------------------------------
    stage0_hits = 0  # merge top-1000
    stage1_hits = 0  # DeepFM top-100
    stage2_hits = 0  # DIN top-20
    stage3_hits = 0  # MMR top-10

    per_user = []
    user_ids = test_df["user_id"].tolist()
    truths = test_df["item_id"].tolist()

    # Score everyone in batch first to save time.
    log.info("Scoring %d users with DeepFM (1000-cand pool)…", len(user_ids))
    flat_users: list[int] = []
    flat_items: list[int] = []
    offsets = [0]
    user_seq: list[int] = []
    for u in user_ids:
        cands = candidates.get(int(u), [])
        if not cands:
            offsets.append(offsets[-1])
            user_seq.append(int(u))
            continue
        flat_users.extend([int(u)] * len(cands))
        flat_items.extend(cands)
        offsets.append(offsets[-1] + len(cands))
        user_seq.append(int(u))
    deepfm_scores = deepfm.score(
        np.asarray(flat_users, dtype=np.int64),
        np.asarray(flat_items, dtype=np.int64),
    )

    log.info("Scoring DIN top-100s…")
    # For each user, take DeepFM top-100 then re-score with DIN.
    for idx, (u, gt) in enumerate(zip(user_seq, truths)):
        cands = candidates.get(int(u), [])
        if not cands:
            per_user.append({"u": int(u), "stage0": 0, "stage1": 0, "stage2": 0, "stage3": 0})
            continue
        o1, o2 = offsets[idx], offsets[idx + 1]
        scores = deepfm_scores[o1:o2]
        # Stage 0
        s0 = int(int(gt) in cands)
        stage0_hits += s0
        # Stage 1: DeepFM top-100
        order = np.argsort(-scores)[:100]
        top100 = [cands[i] for i in order]
        s1 = int(int(gt) in top100)
        stage1_hits += s1
        # Stage 2: DIN re-score top-100 → top-20
        din_scores = din.score(
            np.array([int(u)] * len(top100), dtype=np.int64),
            np.array(top100, dtype=np.int64),
        )
        order2 = np.argsort(-din_scores)[:20]
        top20 = [top100[i] for i in order2]
        top20_scores = [float(din_scores[i]) for i in order2]
        s2 = int(int(gt) in top20)
        stage2_hits += s2
        # Stage 3: MMR top-10 (λ=0.7)
        top10 = mmr_rerank(top20, top20_scores, iv, k=10, lam=0.7)
        s3 = int(int(gt) in top10)
        stage3_hits += s3
        per_user.append({"u": int(u), "stage0": s0, "stage1": s1, "stage2": s2, "stage3": s3})

    n_eval = sum(1 for u in user_seq if candidates.get(int(u)))
    log.info(
        "Funnel: n_eval=%d, stage0=%d, stage1=%d, stage2=%d, stage3=%d",
        n_eval, stage0_hits, stage1_hits, stage2_hits, stage3_hits,
    )

    out = {
        "n_eval": int(n_eval),
        "n_test_users": int(len(user_ids)),
        "stages": [
            {"name": "merge top-1000",   "size": 1000, "positives": int(stage0_hits)},
            {"name": "DeepFM top-100",   "size": 100,  "positives": int(stage1_hits)},
            {"name": "DIN top-20",       "size": 20,   "positives": int(stage2_hits)},
            {"name": "MMR top-10",       "size": 10,   "positives": int(stage3_hits)},
        ],
        "per_user_summary_path": "experiments/ablations/funnel_per_user.parquet",
    }

    out_dir = REPO_ROOT / "experiments" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "funnel.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(per_user).to_parquet(out_dir / "funnel_per_user.parquet", index=False)
    log.info("Saved funnel summary → %s", out_dir / "funnel.json")


if __name__ == "__main__":
    main()
