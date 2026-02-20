"""
GOLDWOLF — Shared utility helpers
Small, reusable functions used across the project.
"""

import logging
import time
from typing import Any


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a consistently-formatted logger.

    Parameters
    ----------
    name : str
        Logger name (use __name__ from the calling module).
    level : int
        Logging level (default INFO).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class Timer:
    """
    Simple context-manager / manual timer for logging elapsed time.

    Usage
    -----
    with Timer("loading data") as t:
        ...
    # or
    t = Timer("processing")
    t.start()
    ...
    t.stop()
    print(t.elapsed_str)
    """

    def __init__(self, label: str = "") -> None:
        self.label = label
        self._start: float = 0.0
        self._end: float = 0.0

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def stop(self) -> "Timer":
        self._end = time.perf_counter()
        return self

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        return self._end - self._start

    @property
    def elapsed_str(self) -> str:
        secs = self.elapsed
        if secs < 60:
            return f"{secs:.2f}s"
        mins, secs = divmod(secs, 60)
        return f"{int(mins)}m {secs:.2f}s"

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()
