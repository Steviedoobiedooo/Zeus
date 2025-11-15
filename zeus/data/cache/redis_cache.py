# zeus/data/cache/redis_cache.py

import io
from typing import Optional

import numpy as np
import bittensor as bt

try:
    import redis  # type: ignore
except ImportError:
    redis = None


class RedisCache:
    """
    Thin wrapper around redis-py for storing numpy arrays with TTL.
    If Redis or redis-py is not available, this becomes a no-op cache.
    """

    def __init__(self, url: str | None = None) -> None:
        self.client = None

        if not url:
            bt.logging.info("[REDIS CACHE] No URL provided → Disabled.")
            return

        if redis is None:
            bt.logging.warning("[REDIS CACHE] redis-py not installed → Cache disabled.")
            return

        try:
            self.client = redis.Redis.from_url(url)

            # Test connection
            self.client.ping()
            bt.logging.info(f"[REDIS CACHE] Connected to {url}")

        except Exception as e:
            bt.logging.warning(
                f"[REDIS CACHE] Connection failed ({e}) → Cache disabled."
            )
            self.client = None

    def get(self, key: str) -> Optional[np.ndarray]:
        if self.client is None:
            return None

        try:
            data = self.client.get(key)
        except Exception as e:
            bt.logging.warning(f"[REDIS CACHE] GET failed for key={key[:12]}... ({e})")
            return None

        if data is None:
            return None

        try:
            buf = io.BytesIO(data)
            arr = np.load(buf, allow_pickle=False)
            return arr

        except Exception as e:
            bt.logging.warning(
                f"[REDIS CACHE] Failed to load key={key[:12]}... ({e})"
            )
            return None

    def set(self, key: str, value: np.ndarray, ttl_seconds: int | None = None) -> None:
        if self.client is None:
            return

        try:
            buf = io.BytesIO()
            np.save(buf, value)

            self.client.set(key, buf.getvalue(), ex=ttl_seconds)

            bt.logging.debug(
                f"[REDIS CACHE] STORED key={key[:12]}... ttl={ttl_seconds}s"
            )

        except Exception as e:
            bt.logging.warning(
                f"[REDIS CACHE] Failed to store key={key[:12]}... ({e})"
            )
