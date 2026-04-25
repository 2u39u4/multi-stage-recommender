#!/usr/bin/env bash
# ============================================================================
# bootstrap_env.sh — one-shot local environment setup.
#
# Usage:
#   bash scripts/bootstrap_env.sh
#
# What it does:
#   1. installs uv if missing
#   2. creates .venv with Python 3.10
#   3. installs the project with dev extras
#   4. installs pre-commit hooks
#   5. prints next steps
# ============================================================================
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
fi

echo ">> creating virtualenv"
uv venv --python 3.10 .venv

# shellcheck disable=SC1091
source .venv/bin/activate

echo ">> installing project with dev extras"
uv pip install -e ".[dev]"

echo ">> installing pre-commit hooks"
pre-commit install

cat <<'EOF'

 ╭──────────────────────────────────────────────────╮
 │  ✓ Environment ready.                            │
 │                                                  │
 │  Next steps:                                     │
 │      source .venv/bin/activate                   │
 │      make help                                   │
 │      make download    # fetch MovieLens-1M       │
 │      make preprocess                             │
 │      make train-als                              │
 ╰──────────────────────────────────────────────────╯
EOF
