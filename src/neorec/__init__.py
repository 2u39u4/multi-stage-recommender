"""NeoRec — a production-style multi-stage recommender system.

Pipeline stages:
    1. ``neorec.data``     — download, preprocess, feature engineering
    2. ``neorec.recall``   — multi-channel candidate generation
    3. ``neorec.ranking``  — DeepFM pre-rank, DIN / Transformer fine-rank
    4. ``neorec.rerank``   — MMR diversity, debias, business rules
    5. ``neorec.serving``  — FastAPI + FAISS + Redis online inference
    6. ``neorec.eval``     — offline metrics, ablations, significance tests
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
