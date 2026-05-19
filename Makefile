.PHONY: help install install-dev lint format test test-fast cov release-check \
        download preprocess \
        train-als train-two-tower train-sasrec \
        train-deepfm train-din \
        benchmark build-faiss serve dashboard mlflow-ui serving-benchmark \
        docker-build docker-up docker-down \
        clean all

# ---------- Defaults ----------
PYTHON ?= python
UV     ?= uv
CFG    ?= configs/config.yaml
PORT   ?= 8000

help:  ## Show this help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ---------- Setup ----------
install:  ## Install production deps via uv
	$(UV) pip install -e .

install-dev:  ## Install project + dev extras + pre-commit hooks
	$(UV) pip install -e ".[dev]"
	pre-commit install

# ---------- Quality ----------
lint:  ## Run ruff + mypy
	ruff check src tests
	ruff format --check src tests
	mypy src/neorec

format:  ## Auto-format with ruff
	ruff check --fix src tests
	ruff format src tests

test:  ## Run full test suite with coverage
	pytest -v

test-fast:  ## Run only fast tests (exclude slow)
	pytest -v -m "not slow"

cov:  ## Open HTML coverage report
	pytest --cov-report=html
	open htmlcov/index.html || xdg-open htmlcov/index.html

release-check:  ## Verify key imports and release-readiness environment
	$(PYTHON) scripts/check_release_ready.py

# ---------- Data pipeline ----------
download:  ## Download MovieLens dataset
	$(PYTHON) -m neorec.cli data download

preprocess:  ## Preprocess raw data into parquet
	$(PYTHON) -m neorec.cli data preprocess

# ---------- Recall ----------
train-als:          ; $(PYTHON) -m neorec.cli train recall=als
train-two-tower:    ; $(PYTHON) -m neorec.cli train recall=two_tower
train-sasrec:       ; $(PYTHON) -m neorec.cli train recall=sasrec

# ---------- Ranking ----------
train-deepfm:       ; $(PYTHON) -m neorec.cli train rank=deepfm
train-din:          ; $(PYTHON) -m neorec.cli train rank=din

# ---------- End-to-end benchmark ----------
benchmark:  ## Run the full benchmark suite (all models + ablations)
	$(PYTHON) -m neorec.cli eval pipeline=full

# ---------- Serving ----------
build-faiss:  ## Build FAISS HNSW index from Two-Tower item embeddings
	$(PYTHON) scripts/build_faiss_index.py

serve:  ## Launch FastAPI on :$(PORT)
	uvicorn neorec.serving.api:app --host 0.0.0.0 --port $(PORT) --reload

dashboard:  ## Launch Streamlit dashboard
	streamlit run src/neorec/serving/dashboard.py

serving-benchmark:  ## Benchmark local FastAPI latency
	$(PYTHON) scripts/benchmark_serving.py --url http://localhost:$(PORT)

mlflow-ui:  ## Launch MLflow UI on :5000
	mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5000

# ---------- Docker ----------
docker-build:  ## Build all docker images
	docker compose -f docker/docker-compose.yaml build

docker-up:  ## Start full stack (api, redis, mlflow, dashboard)
	docker compose -f docker/docker-compose.yaml up -d

docker-down:  ## Stop and remove all containers
	docker compose -f docker/docker-compose.yaml down

# ---------- Housekeeping ----------
clean:  ## Remove caches and build artefacts
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +

# ---------- Paper-style one-shot reproduction ----------
all: download preprocess \
     train-als train-two-tower train-sasrec \
     train-deepfm train-din \
     benchmark  ## Reproduce every number in the README
