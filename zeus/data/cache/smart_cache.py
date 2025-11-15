# zeus/data/cache/smart_cache.py

import hashlib
import time
from typing import Optional

import numpy as np
import bittensor as bt

from .redis_cache import RedisCache
from .disk_cache import DiskCache


class SmartWeatherCache:
    """
    Hybrid cache for Zeus miner:

    - For historical requests (end_time <= now): use DiskCache.
    - For forecast/future requests (end_time > now): use Redis with TTL.

    Cache key is SHA-1 hash of:
        - variable (string)
        - start_time (float64)
        - end_time   (float64)
        - coordinates (float32 array)
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
        """
        coords = np.asarray(coordinates, dtype=np.float32)
        h = hashlib.sha1()
        h.update(variable.encode("utf-8"))
        h.update(np.float64(start_time).tobytes())
        h.update(np.float64(end_time).tobytes())
        h.update(coords.tobytes())
        return h.hexdigest()

    def get(self, variable, start_time, end_time, coordinates):
        now_ts = time.time()
        is_historical = end_time <= now_ts

        key = self._make_key(variable, start_time, end_time, coordinates)

        bt.logging.debug(
            f"[CACHE DEBUG] GET | var={variable} | "
            f"start={start_time} | end={end_time} | "
            f"hist={is_historical} | key={key[:12]}..."
        )

        # Historical → Disk
        if is_historical:
            data = self.disk.get(key)
            if data is not None:
                bt.logging.debug(f"[CACHE DEBUG] HIT (disk) key={key[:12]}...")
            return data

        # Forecast → Redis
        if self.redis is not None:
            data = self.redis.get(key)
            if data is not None:
                bt.logging.debug(f"[CACHE DEBUG] HIT (redis) key={key[:12]}...")
            return data

        bt.logging.debug(f"[CACHE DEBUG] MISS key={key[:12]}...")
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
        Store data in the correct cache tier.
        """
        now_ts = time.time()
        is_historical = end_time <= now_ts

        key = self._make_key(variable, start_time, end_time, coordinates)

        bt.logging.debug(
            f"[CACHE DEBUG] SET | var={variable} | "
            f"start={start_time} | end={end_time} | "
            f"hist={is_historical} | key={key[:12]}..."
        )

        if is_historical:
            self.disk.set(key, data)
            bt.logging.debug(f"[CACHE DEBUG] STORED (disk) key={key[:12]}...")
        else:
            if self.redis is not None:
                self.redis.set(key, data, ttl_seconds=self.forecast_ttl_seconds)
                bt.logging.debug(f"[CACHE DEBUG] STORED (redis) key={key[:12]}...")
