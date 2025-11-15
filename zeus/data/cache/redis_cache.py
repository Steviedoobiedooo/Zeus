# zeus/data/cache/redis_cache.py

import logging
from typing import Optional

import numpy as np

try:
    import redis  # type: ignore
except ImportError:
    redis = None

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Thin wrapper around redis-py for storing numpy arrays with TTL.
    If Redis or redis-py is not available, this becomes a no-op cache.
    """

    def __init__(self, url: str | None = None) -> None:
        self.client = None
        if not url:
            logger.info("RedisCache: no URL provided, disabled.")
            return

        if redis is None:
            logger.warning("RedisCache: redis-py not installed; cache disabled.")
            return

        try:
            self.client = redis.Redis.from_url(url)
            # Test connection
            self.client.ping()
            logger.info("RedisCache: connected to %s", url)
        except Exception as e:
            logger.warning("RedisCache: connection failed (%s); cache disabled.", e)
            self.client = None

    def get(self, key: str) -> Optional[np.ndarray]:
        if self.client is None:
            return None

        data = self.client.get(key)
        if data is None:
            return None

        import io

        try:
            buf = io.BytesIO(data)
            arr = np.load(buf, allow_pickle=False)
            return arr
        except Exception as e:
            logger.warning("RedisCache: failed to load key %s (%s)", key, e)
            return None

    def set(self, key: str, value: np.ndarray, ttl_seconds: int | None = None) -> None:
        if self.client is None:
            return

        import io

        try:
            buf = io.BytesIO()
            # Store as raw numpy .npy
            np.save(buf, value)
            self.client.set(key, buf.getvalue(), ex=ttl_seconds)
        except Exception as e:
            logger.warning("RedisCache: failed to store key %s (%s)", key, e)
