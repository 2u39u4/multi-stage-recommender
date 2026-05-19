"""Release-readiness checks for the W6 finish line.

This is intentionally lightweight: it verifies imports and environment shape,
then prints actionable warnings for optional showcase tasks (screenshots/video).
"""

from __future__ import annotations

import importlib
import platform
import sys


REQUIRED_IMPORTS = [
    "numpy",
    "pandas",
    "pyarrow",
    "sklearn",
    "torch",
    "fastapi",
    "uvicorn",
    "redis",
    "prometheus_client",
    "hydra",
    "omegaconf",
    "faiss",
]

OPTIONAL_SHOWCASE_IMPORTS = [
    "streamlit",
    "plotly",
    "seaborn",
    "matplotlib",
]


def _check_import(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return False, str(exc)
    version = getattr(mod, "__version__", "unknown")
    return True, str(version)


def main() -> int:
    print("NeoRec release-readiness check")
    print(f"python={sys.version.split()[0]} platform={platform.platform()}")

    failed: list[str] = []
    print("\nRequired imports:")
    for name in REQUIRED_IMPORTS:
        ok, detail = _check_import(name)
        print(f"  {'OK ' if ok else 'ERR'} {name}: {detail}")
        if not ok:
            failed.append(name)

    print("\nOptional showcase imports:")
    for name in OPTIONAL_SHOWCASE_IMPORTS:
        ok, detail = _check_import(name)
        print(f"  {'OK ' if ok else 'WARN'} {name}: {detail}")

    if failed:
        print("\nRelease readiness: FAIL")
        print("Missing/incompatible:", ", ".join(failed))
        return 1

    print("\nRelease readiness: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
