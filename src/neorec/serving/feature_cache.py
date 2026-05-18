"""Redis-backed feature cache for the online pipeline."""

from __future__ import annotations

import json
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
        if self._client is not None:
            return
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "redis is required for RedisFeatureCache. Install with "
                "`pip install redis` or use the Docker serving image."
            ) from exc
        self._client = redis.Redis(
            host=self.host,
            port=int(self.port),
            db=int(self.db),
            decode_responses=True,
            socket_connect_timeout=0.25,
            socket_timeout=0.25,
        )
        # Fail early if Redis is configured but unreachable.
        self._client.ping()

    def _key(self, kind: str, entity_id: int) -> str:
        return f"{self.namespace}:{kind}:{int(entity_id)}"

    def _get_json(self, key: str) -> dict[str, Any] | None:
        self._connect()
        raw = self._client.get(key)  # type: ignore[union-attr]
        if raw is None:
            return None
        return json.loads(raw)

    def _set_json(self, key: str, value: dict[str, Any]) -> None:
        self._connect()
        payload = json.dumps(value, ensure_ascii=False)
        self._client.setex(key, int(self.ttl_seconds), payload)  # type: ignore[union-attr]

    def get_user(self, user_id: int) -> dict[str, Any] | None:
        return self._get_json(self._key("user", user_id))

    def set_user(self, user_id: int, features: dict[str, Any]) -> None:
        self._set_json(self._key("user", user_id), features)

    def get_item(self, item_id: int) -> dict[str, Any] | None:
        return self._get_json(self._key("item", item_id))

    def set_item(self, item_id: int, features: dict[str, Any]) -> None:
        self._set_json(self._key("item", item_id), features)

    def batch_get_items(self, item_ids: list[int]) -> list[dict[str, Any] | None]:
        self._connect()
        keys = [self._key("item", i) for i in item_ids]
        raws = self._client.mget(keys)  # type: ignore[union-attr]
        return [json.loads(raw) if raw is not None else None for raw in raws]
