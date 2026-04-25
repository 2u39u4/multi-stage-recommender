"""Redis-backed feature cache for the online pipeline."""

from __future__ import annotations

from typing import Any


class RedisFeatureCache:
    """Thin wrapper around ``redis.Redis`` with namespaced keys and JSON values."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        namespace: str = "neorec",
        ttl_seconds: int = 300,
    ) -> None:
        self.host = host
        self.port = port
        self.db = db
        self.namespace = namespace
        self.ttl_seconds = ttl_seconds
        self._client = None

    def _connect(self) -> None:
        """Lazy-connect on first access (so import stays fast)."""
        raise NotImplementedError  # TODO(W5 Day 31)

    def get_user(self, user_id: int) -> dict[str, Any] | None:
        raise NotImplementedError  # TODO(W5)

    def set_user(self, user_id: int, features: dict[str, Any]) -> None:
        raise NotImplementedError  # TODO(W5)

    def get_item(self, item_id: int) -> dict[str, Any] | None:
        raise NotImplementedError  # TODO(W5)

    def batch_get_items(self, item_ids: list[int]) -> list[dict[str, Any] | None]:
        raise NotImplementedError  # TODO(W5)
