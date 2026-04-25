"""Tiny MLflow helpers — keep call sites clean and avoid hard dependency at import time."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

log = logging.getLogger(__name__)


@contextmanager
def mlflow_run(
    experiment: str,
    run_name: str | None = None,
    tracking_uri: str | None = None,
    tags: dict[str, str] | None = None,
):
    """Context manager that starts an MLflow run and degrades gracefully if the
    library is unavailable (so unit tests don't depend on MLflow being installed).
    """
    try:
        import mlflow
    except ImportError:  # pragma: no cover
        log.warning("mlflow not installed; logging will be a no-op.")

        class _Noop:
            def log_param(self, *a: Any, **kw: Any) -> None: ...
            def log_params(self, *a: Any, **kw: Any) -> None: ...
            def log_metric(self, *a: Any, **kw: Any) -> None: ...
            def log_metrics(self, *a: Any, **kw: Any) -> None: ...
            def log_artifact(self, *a: Any, **kw: Any) -> None: ...
            def set_tags(self, *a: Any, **kw: Any) -> None: ...

        yield _Noop()
        return

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        log.info("MLflow run id=%s, uri=%s", run.info.run_id, mlflow.get_tracking_uri())
        yield mlflow
