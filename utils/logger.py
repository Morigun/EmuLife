from __future__ import annotations

import logging
from typing import Optional


_logger: Optional[logging.Logger] = None


def get_logger(name: str = "emulife") -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger
    _logger = logging.getLogger(name)
    if not _logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        _logger.addHandler(handler)
        _logger.setLevel(logging.INFO)
    return _logger
