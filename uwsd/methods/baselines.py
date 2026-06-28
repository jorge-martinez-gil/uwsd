"""Reference baselines: Most-Frequent-Sense (MFS) and Random.

These require no model downloads and run in milliseconds, which makes them the
ideal way to (a) sanity-check the harness against the numbers published in the
paper and (b) give every neural method an honest point of comparison.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from ..data import Instance, WordTask
from .base import Prediction, WSDMethod


class MostFrequentSense(WSDMethod):
    name = "mfs"
    requires_train = True
    description = "Most-Frequent-Sense: always predict the majority sense in train."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mfs: Optional[int] = None
        self._dist: dict = {}

    def fit(self, task: WordTask) -> None:
        labels = [i.gold_label for i in task.train if i.gold_label is not None]
        if not labels:
            # Fall back to the smallest label id if no training data exists.
            self._mfs = task.label_ids[0]
            self._dist = {}
            return
        counts = Counter(labels)
        total = sum(counts.values())
        self._mfs = counts.most_common(1)[0][0]
        self._dist = {k: v / total for k, v in counts.items()}

    def predict(self, instance: Instance, task: WordTask) -> Prediction:
        if self._mfs is None:
            self.fit(task)
        lab = self._mfs
        conf = self._dist.get(lab, 1.0)
        return Prediction(
            label_id=lab,
            label=task.classes[lab],
            scores=dict(self._dist),
            confidence=conf,
            explanation=(
                f"Most frequent sense for '{task.word}' in training data "
                f"({conf:.1%} of train occurrences)."
            ),
        )


class RandomBaseline(WSDMethod):
    name = "random"
    requires_train = False
    description = "Random sense (seeded, uniform over candidate senses)."

    def __init__(self, seed: int = 42, **kwargs):
        super().__init__(seed=seed, **kwargs)
        import numpy as np

        self._rng = np.random.default_rng(seed)

    def predict(self, instance: Instance, task: WordTask) -> Prediction:
        ids = task.label_ids
        lab = int(self._rng.choice(ids))
        return Prediction(
            label_id=lab,
            label=task.classes[lab],
            scores={i: 1.0 / len(ids) for i in ids},
            confidence=1.0 / len(ids),
            explanation="Uniform random choice among candidate senses.",
            low_confidence=True,
        )
