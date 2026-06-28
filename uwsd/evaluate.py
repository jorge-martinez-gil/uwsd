"""Run a method over a dataset and produce a reproducible result manifest.

The manifest is a plain dict (JSON-serialisable) that captures *everything*
needed to trace a number back to its origin: the method + encoder config, the
dataset, per-word and overall metrics with a bootstrap CI, environment
versions, the git commit, and (optionally) every per-instance prediction with
its explanation.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Optional

from . import __version__
from .data import Dataset
from .methods.base import WSDMethod
from .metrics import aggregate_by_word


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _env_info() -> dict:
    info = {
        "uwsd_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }
    for pkg in ("numpy", "torch", "sentence_transformers"):
        try:
            mod = __import__(pkg)
            info[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except Exception:
            pass
    return info


def evaluate(
    method: WSDMethod,
    dataset: Dataset,
    limit: Optional[int] = None,
    keep_predictions: bool = False,
    progress: bool = False,
) -> dict:
    """Evaluate ``method`` on ``dataset`` and return a result manifest dict."""
    rows = []
    predictions = {}
    for wi, task in enumerate(dataset):
        if method.requires_train:
            method.fit(task)
        preds = method.predict_task(task, limit=limit)
        items = task.test if limit is None else task.test[:limit]
        gold = [i.gold_label for i in items]
        pred = [p.label_id for p in preds]
        rows.append((task.word, gold, pred))
        if keep_predictions:
            predictions[task.word] = [
                {
                    "instance_id": inst.instance_id,
                    "sentence": inst.sentence,
                    "target": inst.target_token,
                    "gold": inst.gold_label,
                    **p.as_dict(),
                }
                for inst, p in zip(items, preds)
            ]
        if progress:
            print(
                f"  [{wi + 1}/{len(dataset)}] {task.word}: "
                f"{sum(g == pp for g, pp in zip(gold, pred))}/{len(gold)}",
                flush=True,
            )

    summary = aggregate_by_word(rows)
    correct_vector = summary.pop("correct_vector")

    manifest = {
        "method": method.metadata(),
        "dataset": {
            "name": dataset.name,
            "words": dataset.words,
            "num_test_instances": summary["n"],
            "limit_per_word": limit,
        },
        "metrics": summary,
        "environment": _env_info(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if keep_predictions:
        manifest["predictions"] = predictions
    # correctness vector kept separately for significance tests, not in manifest
    manifest["_correct_vector"] = correct_vector
    return manifest
