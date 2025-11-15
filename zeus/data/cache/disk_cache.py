# zeus/data/cache/disk_cache.py

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DiskCache:
    """
    Very simple disk cache for numpy arrays.
    Uses one file per key under base_dir and enforces a max total size.
    """

    def __init__(self, base_dir: str, max_bytes: int) -> None:
        self.base_dir = base_dir
        self.max_bytes = max_bytes
        os.makedirs(self.base_dir, exist_ok=True)
        logger.info(
            "DiskCache: base_dir=%s, max_bytes=%.2f GB",
            self.base_dir,
            self.max_bytes / (1024**3),
        )

    def _path_for_key(self, key: str) -> str:
        # Flat layout is fine for now; key is already a short hex string.
        return os.path.join(self.base_dir, f"{key}.npy")

    def get(self, key: str) -> Optional[np.ndarray]:
        path = self._path_for_key(key)
        if not os.path.exists(path):
            return None
        try:
            return np.load(path, allow_pickle=False)
        except Exception as e:
            logger.warning("DiskCache: failed to load %s (%s), deleting file", path, e)
            try:
                os.remove(path)
            except OSError:
                pass
            return None

    def set(self, key: str, value: np.ndarray) -> None:
        if self.max_bytes <= 0:
            return
        path = self._path_for_key(key)
        try:
            np.save(path, value)
        except Exception as e:
            logger.warning("DiskCache: failed to save %s (%s)", path, e)
            return
        self._enforce_limit()

    def _enforce_limit(self) -> None:
        if self.max_bytes <= 0:
            return

        total = 0
        files = []

        try:
            for entry in os.scandir(self.base_dir):
                if not entry.is_file():
                    continue
                try:
                    st = entry.stat()
                except OSError:
                    continue
                size = st.st_size
                total += size
                files.append((entry.path, st.st_mtime, size))
        except FileNotFoundError:
            return

        if total <= self.max_bytes:
            return

        # Oldest files first
        files.sort(key=lambda x: x[1])

        for path, _mtime, size in files:
            if total <= self.max_bytes:
                break
            try:
                os.remove(path)
                total -= size
            except OSError:
                continue
