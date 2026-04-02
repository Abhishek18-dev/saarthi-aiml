"""Structured logging configuration.

Provides a pre-configured logger so every module can simply do:
    from app.core.logger import logger
"""

from __future__ import annotations

import logging
import sys

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d — %(message)s"
)

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("skillsync")
