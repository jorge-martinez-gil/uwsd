"""Dataset loading for unsupervised WSD benchmarks.

Currently ships the CoarseWSD-20 benchmark (Loureiro et al., 2021), bundled in
the repository under ``CoarseWSD-20/``. The loader is intentionally
cross-platform (no hardcoded path separators) and returns typed, immutable-ish
data objects so that downstream code never re-parses raw files.

Directory layout expected for each ambiguous word ``<w>``::

    CoarseWSD-20/<w>/classes_map.txt   # JSON: {"0": "label_0", "1": "label_1", ...}
    CoarseWSD-20/<w>/train.data.txt    # TSV:  <token_index>\t<sentence>
    CoarseWSD-20/<w>/train.gold.txt    #       one integer label id per line
    CoarseWSD-20/<w>/test.data.txt     # TSV:  <token_index>\t<sentence>
    CoarseWSD-20/<w>/test.gold.txt     #       one integer label id per line
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

# --------------------------------------------------------------------------- #
# Locating the bundled data
# --------------------------------------------------------------------------- #

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent


def default_data_root(name: str = "CoarseWSD-20") -> Path:
    """Best-effort discovery of a bundled dataset directory.

    Search order: ``$UWSD_DATA`` env var, current working directory, the
    repository root (parent of this package). Raises ``FileNotFoundError`` if
    nothing is found so callers fail loudly rather than silently.
    """
    candidates = []
    env = os.environ.get("UWSD_DATA")
    if env:
        candidates.append(Path(env))
        candidates.append(Path(env) / name)
    candidates.append(Path.cwd() / name)
    candidates.append(_REPO_ROOT / name)
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(
        f"Could not locate dataset '{name}'. Looked in: "
        + ", ".join(str(c) for c in candidates)
        + ". Set the UWSD_DATA environment variable to the dataset directory."
    )


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class Instance:
    """A single WSD example: an ambiguous ``word`` occurring in ``sentence``."""

    word: str
    token_index: int
    sentence: str
    gold_label: Optional[int] = None
    instance_id: Optional[str] = None

    @property
    def tokens(self) -> List[str]:
        return self.sentence.split()

    @property
    def target_token(self) -> str:
        toks = self.tokens
        if 0 <= self.token_index < len(toks):
            return toks[self.token_index]
        return self.word


@dataclass
class WordTask:
    """All data for one ambiguous word: its candidate senses and examples."""

    word: str
    classes: Dict[int, str]
    train: List[Instance] = field(default_factory=list)
    test: List[Instance] = field(default_factory=list)

    @property
    def num_senses(self) -> int:
        return len(self.classes)

    @property
    def label_ids(self) -> List[int]:
        return sorted(self.classes)

    def sense_phrase(self, label_id: int, drop_target: bool = True) -> str:
        """Turn a raw class label into a natural-language substitution phrase.

        Mirrors the cleaning used in the original paper scripts, e.g.
        ``"java_java_(programming_language)"`` -> ``"programming language"``.
        Underscores become spaces, parentheses are dropped, and (optionally)
        tokens equal to the ambiguous word itself are removed.
        """
        raw = self.classes[label_id]
        s = raw.replace("_", " ").replace("(", "").replace(")", "")
        if drop_target:
            toks = [t for t in s.split() if t.lower() != self.word.lower()]
            s = " ".join(toks)
        s = " ".join(s.split()).strip()
        return s or raw.replace("_", " ").strip()


@dataclass
class Dataset:
    """A named collection of :class:`WordTask` objects."""

    name: str
    tasks: Dict[str, WordTask]
    root: Optional[Path] = None

    @property
    def words(self) -> List[str]:
        return sorted(self.tasks)

    def __iter__(self) -> Iterator[WordTask]:
        for w in self.words:
            yield self.tasks[w]

    def __len__(self) -> int:
        return len(self.tasks)

    def test_instances(self) -> Iterator[Instance]:
        for task in self:
            yield from task.test

    @property
    def num_test_instances(self) -> int:
        return sum(len(t.test) for t in self.tasks.values())

    @property
    def num_train_instances(self) -> int:
        return sum(len(t.train) for t in self.tasks.values())


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #


def _read_lines(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def _load_split(folder: Path, word: str, split: str) -> List[Instance]:
    data_path = folder / f"{split}.data.txt"
    gold_path = folder / f"{split}.gold.txt"
    if not data_path.exists():
        return []
    data_lines = _read_lines(data_path)
    gold = [int(x) for x in _read_lines(gold_path)] if gold_path.exists() else None
    if gold is not None and len(gold) != len(data_lines):
        raise ValueError(
            f"{word}/{split}: {len(data_lines)} data lines but {len(gold)} gold labels"
        )
    instances: List[Instance] = []
    for i, line in enumerate(data_lines):
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            raise ValueError(f"{word}/{split} line {i}: expected '<index>\\t<text>'")
        idx = int(parts[0])
        instances.append(
            Instance(
                word=word,
                token_index=idx,
                sentence=parts[1],
                gold_label=gold[i] if gold is not None else None,
                instance_id=f"{word}.{split}.{i}",
            )
        )
    return instances


def load_word_task(folder: os.PathLike, word: Optional[str] = None) -> WordTask:
    """Load a single ambiguous-word task from its folder."""
    folder = Path(folder)
    word = word or folder.name
    with open(folder / "classes_map.txt", "r", encoding="utf-8") as f:
        raw_classes = json.load(f)
    classes = {int(k): v for k, v in raw_classes.items()}
    return WordTask(
        word=word,
        classes=classes,
        train=_load_split(folder, word, "train"),
        test=_load_split(folder, word, "test"),
    )


def load_coarsewsd20(
    root: Optional[os.PathLike] = None,
    words: Optional[Iterable[str]] = None,
) -> Dataset:
    """Load the CoarseWSD-20 benchmark.

    Parameters
    ----------
    root:
        Path to the ``CoarseWSD-20`` directory. If ``None``, it is discovered
        automatically (see :func:`default_data_root`).
    words:
        Optional subset of ambiguous words to load (useful for quick tests).
    """
    root = Path(root) if root is not None else default_data_root("CoarseWSD-20")
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    wanted = set(words) if words is not None else None
    tasks: Dict[str, WordTask] = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if wanted is not None and entry.name not in wanted:
            continue
        if not (entry / "classes_map.txt").exists():
            continue
        tasks[entry.name] = load_word_task(entry)
    if not tasks:
        raise FileNotFoundError(f"No word folders found under {root}")
    return Dataset(name="CoarseWSD-20", tasks=tasks, root=root)
