# zeus/data/cache/smart_cache.py

import hashlib
import time
from typing import Optional

import numpy as np

from .redis_cache import RedisCache
from .disk_cache import DiskCache


class SmartWeatherCache:
    """
    Hybrid cache for Zeus miner:

    - For historical requests (end_time <= now): use DiskCache (no TTL).
    - For forecast/future requests (end_time > now): use Redis with TTL.

    Keys are derived from:
        - variable name
        - start_time, end_time (float)
        - coordinate grid (lat, lon pairs)
    """

    def __init__(
        self,
        disk_dir: str,
        max_bytes: int,
        redis_url: str | None = None,
        forecast_ttl_seconds: int = 2 * 60 * 60,  # 2 hours
    ) -> None:
        self.disk = DiskCache(base_dir=disk_dir, max_bytes=max_bytes)
        self.redis = RedisCache(url=redis_url) if redis_url else None
        self.forecast_ttl_seconds = forecast_ttl_seconds

    def _make_key(
        self,
        variable: str,
        start_time: float,
        end_time: float,
        coordinates: np.ndarray,
    ) -> str:
        """
        Build a stable hash key across processes.
        We hash:
          - variable string
          - start/end as float64 bytes
          - coords as float32 bytes
        """
        coords = np.asarray(coordinates, dtype=np.float32)
        h = hashlib.sha1()
        h.update(variable.encode("utf-8"))
        h.update(np.float64(start_time).tobytes())
        h.update(np.float64(end_time).tobytes())
        h.update(coords.tobytes())
        return h.hexdigest()

    def get(
        self,
        variable: str,
        start_time: float,
        end_time: float,
        coordinates: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Look up a cached array. Chooses disk vs Redis based on whether
        end_time is in the past or future.
        """
        now_ts = time.time()
        is_historical = end_time <= now_ts

        key = self._make_key(variable, start_time, end_time, coordinates)

        if is_historical:
            return self.disk.get(key)

        # Forecast path
        if self.redis is not None:
            arr = self.redis.get(key)
            if arr is not None:
                return arr

        return None

    def set(
        self,
        variable: str,
        start_time: float,
        end_time: float,
        coordinates: np.ndarray,
        data: np.ndarray,
    ) -> None:
        """
        Store a new array in the appropriate cache tier.
        """
        now_ts = time.time()
        is_historical = end_time <= now_ts

        key = self._make_key(variable, start_time, end_time, coordinates)

        if is_historical:
            # Persistent on disk
            self.disk.set(key, data)
        else:
            # Short-term forecast in Redis
            if self.redis is not None:
                self.redis.set(key, data, ttl_seconds=self.forecast_ttl_seconds)
