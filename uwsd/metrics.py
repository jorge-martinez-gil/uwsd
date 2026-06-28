"""Evaluation metrics for WSD with uncertainty quantification.

All functions are pure NumPy so they have no heavy dependencies and are trivial
to unit-test. The module deliberately provides more than raw accuracy:
per-class precision/recall/F1, macro and weighted aggregates, bootstrap
confidence intervals, and paired significance tests (bootstrap + McNemar) for
comparing two systems on the same instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np


def accuracy(gold: Sequence[int], pred: Sequence[int]) -> float:
    g = np.asarray(gold)
    p = np.asarray(pred)
    if g.size == 0:
        return 0.0
    return float(np.mean(g == p))


@dataclass
class ClassScores:
    label: int
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class Report:
    """A full classification report for one set of gold/pred labels."""

    accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    per_class: List[ClassScores] = field(default_factory=list)
    n: int = 0

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "per_class": [vars(c) for c in self.per_class],
        }


def classification_report(
    gold: Sequence[int], pred: Sequence[int], labels: Sequence[int] = None
) -> Report:
    """Compute accuracy plus per-class and aggregate precision/recall/F1."""
    g = np.asarray(gold)
    p = np.asarray(pred)
    n = int(g.size)
    if labels is None:
        labels = sorted(set(g.tolist()) | set(p.tolist()))
    per_class: List[ClassScores] = []
    f1s, precs, recs, supports = [], [], [], []
    for lab in labels:
        tp = int(np.sum((p == lab) & (g == lab)))
        fp = int(np.sum((p == lab) & (g != lab)))
        fn = int(np.sum((p != lab) & (g == lab)))
        support = int(np.sum(g == lab))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class.append(ClassScores(int(lab), prec, rec, f1, support))
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)
        supports.append(support)
    supports_arr = np.asarray(supports, dtype=float)
    total_support = supports_arr.sum()
    weighted_f1 = (
        float(np.average(f1s, weights=supports_arr)) if total_support > 0 else 0.0
    )
    return Report(
        accuracy=accuracy(g, p),
        macro_f1=float(np.mean(f1s)) if f1s else 0.0,
        weighted_f1=weighted_f1,
        macro_precision=float(np.mean(precs)) if precs else 0.0,
        macro_recall=float(np.mean(recs)) if recs else 0.0,
        per_class=per_class,
        n=n,
    )


def bootstrap_accuracy_ci(
    correct: Sequence[bool],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 1234,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for accuracy.

    ``correct`` is a per-instance boolean (or 0/1) correctness vector.
    Returns ``(point_estimate, ci_low, ci_high)``.
    """
    c = np.asarray(correct, dtype=float)
    n = c.size
    if n == 0:
        return 0.0, 0.0, 0.0
    point = float(c.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = c[idx].mean(axis=1)
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    return point, low, high


def paired_bootstrap_test(
    correct_a: Sequence[bool],
    correct_b: Sequence[bool],
    n_resamples: int = 10000,
    seed: int = 1234,
) -> dict:
    """Two-sided paired bootstrap test for the accuracy difference (A - B)."""
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired test requires equal-length correctness vectors")
    diff = a - b
    n = diff.size
    observed = float(diff.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot = diff[idx].mean(axis=1)
    centered = boot - observed
    p = float(np.mean(np.abs(centered) >= np.abs(observed)))
    return {
        "delta_accuracy": observed,
        "p_value": p,
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "n_resamples": n_resamples,
    }


def mcnemar_test(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict:
    """Exact McNemar test on paired correctness vectors.

    Two-sided exact binomial p-value on discordant pairs (p=0.5), computed in
    log-space so it stays stable for thousands of discordant pairs.
    """
    from math import exp, lgamma, log

    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    b01 = int(np.sum(~a & b))
    b10 = int(np.sum(a & ~b))
    n = int(b01 + b10)
    k = int(min(b01, b10))
    if n == 0:
        p = 1.0
    else:
        log_half_n = float(n) * log(0.5)
        tail = 0.0
        for i in range(0, k + 1):
            log_coef = lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1)
            tail += exp(log_coef + log_half_n)
        p = min(1.0, 2.0 * tail)
    return {"b_a_only": b10, "b_b_only": b01, "discordant": n, "p_value": float(p)}


def aggregate_by_word(
    rows: Sequence[Tuple[str, Sequence[int], Sequence[int]]],
) -> dict:
    """Aggregate per-word (word, gold, pred) triples into a dataset summary."""
    per_word: Dict[str, dict] = {}
    all_correct: List[int] = []
    macro_f1s, weighted_f1s, supports = [], [], []
    total_hits = 0
    total_n = 0
    for word, gold, pred in rows:
        rep = classification_report(gold, pred)
        g = np.asarray(gold)
        p = np.asarray(pred)
        hits = int(np.sum(g == p))
        per_word[word] = {
            "n": rep.n,
            "hits": hits,
            "accuracy": rep.accuracy,
            "macro_f1": rep.macro_f1,
            "weighted_f1": rep.weighted_f1,
            "num_senses": len(set(g.tolist())),
        }
        all_correct.extend((g == p).astype(int).tolist())
        macro_f1s.append(rep.macro_f1)
        weighted_f1s.append(rep.weighted_f1)
        supports.append(rep.n)
        total_hits += hits
        total_n += rep.n
    supports_arr = np.asarray(supports, dtype=float)
    point, low, high = bootstrap_accuracy_ci(all_correct)
    weighted_f1_over_words = 0.0
    if supports_arr.sum() > 0:
        weighted_f1_over_words = float(np.average(weighted_f1s, weights=supports_arr))
    return {
        "micro_accuracy": total_hits / total_n if total_n else 0.0,
        "accuracy_ci95": [low, high],
        "macro_f1_over_words": float(np.mean(macro_f1s)) if macro_f1s else 0.0,
        "weighted_f1_over_words": weighted_f1_over_words,
        "hits": total_hits,
        "n": total_n,
        "per_word": per_word,
        "correct_vector": all_correct,
    }
