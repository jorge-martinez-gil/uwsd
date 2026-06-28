"""Abstract interface that every WSD method implements.

A method receives an :class:`~uwsd.data.Instance` together with its
:class:`~uwsd.data.WordTask` (which carries the candidate senses) and returns a
:class:`Prediction`. Predictions are *interpretable by construction*: they carry
the per-candidate scores, a confidence value, and a human-readable explanation,
so the harness can surface "why this sense was chosen" for every example.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..data import Instance, WordTask


@dataclass
class Prediction:
    label_id: int
    label: str
    scores: Dict[int, float] = field(default_factory=dict)
    confidence: float = 1.0
    explanation: str = ""
    low_confidence: bool = False
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "label_id": self.label_id,
            "label": self.label,
            "scores": {str(k): float(v) for k, v in self.scores.items()},
            "confidence": float(self.confidence),
            "low_confidence": bool(self.low_confidence),
            "explanation": self.explanation,
        }


def softmax_confidence(
    scores: Dict[int, float], temperature: float = 0.05
) -> Dict[int, float]:
    """Turn raw similarity scores into a probability-like distribution."""
    if not scores:
        return {}
    keys = list(scores)
    vals = [scores[k] / max(temperature, 1e-9) for k in keys]
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    z = sum(exps)
    return {k: e / z for k, e in zip(keys, exps)}


class WSDMethod:
    """Base class for all disambiguation methods.

    Subclasses must set :attr:`name` and implement :meth:`predict`. Methods that
    use training data (e.g. most-frequent-sense) set ``requires_train = True``
    and override :meth:`fit`.
    """

    name: str = "base"
    requires_train: bool = False
    #: short human description shown by ``uwsd list-methods``
    description: str = ""

    def __init__(self, **kwargs):
        self.config = kwargs

    # -- lifecycle ------------------------------------------------------- #
    def fit(self, task: WordTask) -> None:  # noqa: D401 - optional hook
        """Optional per-word training hook (no-op for unsupervised methods)."""

    def predict(self, instance: Instance, task: WordTask) -> Prediction:
        raise NotImplementedError

    # -- batch convenience ---------------------------------------------- #
    def predict_task(
        self, task: WordTask, limit: Optional[int] = None
    ) -> List[Prediction]:
        """Predict every test instance of ``task`` (override for batching)."""
        if self.requires_train:
            self.fit(task)
        items = task.test if limit is None else task.test[:limit]
        return [self.predict(inst, task) for inst in items]

    # -- metadata ------------------------------------------------------- #
    def metadata(self) -> dict:
        return {"name": self.name, "config": dict(self.config)}
