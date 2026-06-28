"""uwsd: a benchmark and experimentation platform for unsupervised word sense
disambiguation (WSD) via context-aware semantic similarity.

Public API
----------
- :func:`uwsd.data.load_coarsewsd20` -- load the CoarseWSD-20 benchmark.
- :func:`uwsd.methods.get_method` / :func:`uwsd.methods.list_methods` -- access
  the pluggable method registry.
- :mod:`uwsd.metrics` -- accuracy, F1, bootstrap confidence intervals and
  paired significance testing.
- :func:`uwsd.evaluate.evaluate` -- run a method over a dataset and return a
  fully-populated, reproducible result object.

Reference
---------
Jorge Martinez-Gil. "Context-Aware Semantic Similarity Measurement for
Unsupervised Word Sense Disambiguation." arXiv:2305.03520, 2023.
"""

from __future__ import annotations

__version__ = "0.2.0"

from . import data, metrics  # noqa: F401
from .data import Dataset, Instance, WordTask, load_coarsewsd20  # noqa: F401
from .methods import get_method, list_methods, register_method  # noqa: F401
from .methods.base import Prediction, WSDMethod  # noqa: F401

__all__ = [
    "__version__",
    "data",
    "metrics",
    "Dataset",
    "Instance",
    "WordTask",
    "load_coarsewsd20",
    "get_method",
    "list_methods",
    "register_method",
    "Prediction",
    "WSDMethod",
]
