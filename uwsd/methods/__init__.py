"""Pluggable registry of WSD methods.

Add a new method in three lines::

    from uwsd.methods import register_method, WSDMethod

    @register_method("my-method", description="...")
    class MyMethod(WSDMethod):
        def predict(self, instance, task):
            ...

It is then immediately available from the CLI (``uwsd run --method my-method``)
and from :func:`get_method`.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from .base import Prediction, WSDMethod  # noqa: F401
from .baselines import MostFrequentSense, RandomBaseline
from .similarity import HashingEncoder, SubstitutionSimilarity

# name -> factory(**kwargs) -> WSDMethod instance
_REGISTRY: Dict[str, Callable[..., WSDMethod]] = {}
_DESCRIPTIONS: Dict[str, str] = {}


def register_method(name: str, factory: Callable[..., WSDMethod] = None,
                    description: str = ""):
    """Register a method factory. Usable as a decorator or a direct call."""

    def _register(fac: Callable[..., WSDMethod]):
        _REGISTRY[name] = fac
        _DESCRIPTIONS[name] = description or getattr(fac, "description", "")
        return fac

    if factory is not None:
        return _register(factory)
    return _register


def get_method(name: str, **kwargs) -> WSDMethod:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown method '{name}'. Available: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[name](**kwargs)


def list_methods() -> List[dict]:
    return [
        {"name": n, "description": _DESCRIPTIONS.get(n, "")}
        for n in sorted(_REGISTRY)
    ]


# --------------------------------------------------------------------------- #
# Built-in registrations
# --------------------------------------------------------------------------- #

register_method("mfs", MostFrequentSense,
                description="Most-Frequent-Sense supervised baseline (uses train labels).")
register_method("random", RandomBaseline,
                description="Random sense baseline (seeded).")

register_method(
    "bert",
    lambda **kw: SubstitutionSimilarity(
        backend="sentence-transformer", model=kw.pop("model", "all-MiniLM-L6-v2"), **kw
    ),
    description="UWSD via sentence-transformer (default all-MiniLM-L6-v2). [extras: bert]",
)
register_method(
    "sbert",
    lambda **kw: SubstitutionSimilarity(
        backend="sentence-transformer", model=kw.pop("model", "all-mpnet-base-v2"), **kw
    ),
    description="UWSD via Sentence-BERT (default all-mpnet-base-v2). [extras: bert]",
)
register_method(
    "similarity",
    SubstitutionSimilarity,
    description="Generic substitution-similarity method (configurable backend/model).",
)
register_method(
    "hashing",
    lambda **kw: SubstitutionSimilarity(encoder=HashingEncoder(), **kw),
    description="Offline deterministic encoder (no downloads) for testing/demos.",
)

__all__ = [
    "Prediction",
    "WSDMethod",
    "register_method",
    "get_method",
    "list_methods",
    "SubstitutionSimilarity",
    "MostFrequentSense",
    "RandomBaseline",
    "HashingEncoder",
]
