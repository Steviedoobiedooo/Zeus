# zeus/data/cache/disk_cache.py

import os
from typing import Optional

import numpy as np
import bittensor as bt


class DiskCache:
    """
    Very simple disk cache for numpy arrays.
    Uses one file per key under base_dir and enforces a max total size.
    """

    def __init__(self, base_dir: str, max_bytes: int) -> None:
        self.base_dir = base_dir
        self.max_bytes = max_bytes
        os.makedirs(self.base_dir, exist_ok=True)

        bt.logging.info(
            f"[DISK CACHE] Initialized | base_dir={self.base_dir} | "
            f"max_bytes={self.max_bytes / (1024**3):.2f} GB"
        )

    def _path_for_key(self, key: str) -> str:
        return os.path.join(self.base_dir, f"{key}.npy")

    def get(self, key: str) -> Optional[np.ndarray]:
        path = self._path_for_key(key)

        if not os.path.exists(path):
            return None

        try:
            return np.load(path, allow_pickle=False)

        except Exception as e:
            bt.logging.warning(
                f"[DISK CACHE] Failed to load {path} ({e}), deleting file."
            )
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
            bt.logging.warning(
                f"[DISK CACHE] Failed to save {path} ({e})"
            )
            return

        # Enforce storage limit after saving
        self._enforce_limit()

    def _enforce_limit(self) -> None:
        if self.max_bytes <= 0:
            return

        total_size = 0
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
                total_size += size
                files.append((entry.path, st.st_mtime, size))

        except FileNotFoundError:
            return

        if total_size <= self.max_bytes:
            return

        # Remove oldest files first
        files.sort(key=lambda x: x[1])

        for path, _mtime, size in files:
            if total_size <= self.max_bytes:
                break

            try:
                os.remove(path)
                total_size -= size
                bt.logging.debug(
                    f"[DISK CACHE] Evicted {os.path.basename(path)} "
                    f"({size / (1024**2):.2f} MB)"
                )
            except OSError:
                continue
