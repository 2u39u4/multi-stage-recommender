# NeoRec Release Checklist

This checklist tracks the final W6 work for the repository code/artifact release.
Blog posts and demo videos are intentionally out of scope for this portfolio
version.

## Required Before Tagging `v1.0.0`

- [x] Rebuild / verify the active environment:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -e ".[dev]"
  ```
  Active `.venv` verified through tests and `make release-check` equivalent.

- [x] Run the fast test suite:
  ```bash
  make test-fast
  ```
  Verified with `python -m pytest -q`: 61 passed.

- [x] Run the focused W4/W5 checks:
  ```bash
  python -m pytest tests/test_api.py tests/test_serving.py tests/test_rerank.py tests/test_pipeline_e2e.py -q
  ```
  Verified: 26 passed.

- [x] Run the release import check:
  ```bash
  make release-check
  ```
  Verified: `scripts/check_release_ready.py` PASS with NumPy 1.26.4 and
  PyTorch 2.2.2, matching the Docker serving image constraints.

- [x] Generate README figures:
  ```bash
  python scripts/build_readme_figures.py
  python scripts/build_eda_notebook.py
  # then execute notebooks/01_eda.ipynb to regenerate experiments/results/eda/*.png
  ```
  Current committed README image references resolve locally:
  `experiments/results/eda/*.png` and `experiments/results/figures/*.png`.

- [x] Build the serving image:
  ```bash
  docker compose -f docker/docker-compose.yaml build
  ```
  Completed with `neorec-serve:latest`.

- [x] Start the serving stack and inspect health:
  ```bash
  docker compose -f docker/docker-compose.yaml up -d api redis dashboard
  curl http://localhost:8000/health
  curl http://localhost:8000/metrics
  ```
  Local non-Docker verification completed: FastAPI `/health`, `/metrics`,
  `/recommend/1`, and Streamlit `/_stcore/health` returned 200.
  Docker core stack verification completed for `api + redis + dashboard`.
  Optional observability services live behind the `observability` profile and
  are not required for the serving release contract.

- [x] Record a local serving latency benchmark:
  ```bash
  make serving-benchmark
  ```
  Current README benchmark: 30 requests, concurrency=4, p50=23.53 ms,
  p95=26.10 ms, p99=26.95 ms on local Uvicorn.
  Docker core stack benchmark: p50=1002.27 ms, p95=1311.45 ms,
  p99=1488.62 ms, QPS=3.85 over 30 requests at concurrency=4.

- [x] Confirm no generated secrets or local artefacts are staged before committing:
  ```bash
  git status --short
  ```
  Final git status was verified clean after commit and push.

## Explicitly Out Of Scope

- [x] Demo video: not part of this release.
- [x] Blog post: not part of this release.
- [x] Manual dashboard screenshots: replaced by reproducible static figures under `experiments/results/figures/`.
