"""Protuna: An Optimization framework for AI-guided protein engineering."""

import importlib.metadata
import logging

try:
    __version__: str = importlib.metadata.version("protuna")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0+unknown"

logging.getLogger(__name__).addHandler(logging.NullHandler())
