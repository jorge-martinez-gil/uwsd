"""Context-aware semantic-similarity WSD (the method from the paper).

Idea (unsupervised): for an ambiguous word in a sentence, substitute each
candidate sense's phrase into the sentence and measure how semantically close
the rewritten sentence stays to the original. The sense whose substitution
preserves meaning best is selected. This requires *no* sense-annotated training
data -- only a sentence encoder.

Encoders are pluggable. ``SentenceTransformerEncoder`` wraps any
sentence-transformers model (BERT/MPNet/MiniLM/RoBERTa/SBERT). ``HashingEncoder``
is a dependency-free deterministic encoder used for tests and offline demos so
the full code path can run without downloading multi-gigabyte models.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol

import numpy as np

from ..data import Instance, WordTask
from .base import Prediction, WSDMethod, softmax_confidence


# --------------------------------------------------------------------------- #
# Encoder protocol + implementations
# --------------------------------------------------------------------------- #


class Encoder(Protocol):
    name: str

    def encode(self, texts: List[str]) -> np.ndarray:  # returns (n, d), L2-normed
        ...


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class SentenceTransformerEncoder:
    """Wrap a sentence-transformers model. Lazily imported and loaded."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None,
                 batch_size: int = 64):
        self.name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _ensure(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover - environment dependent
                raise ImportError(
                    "sentence-transformers is required for this encoder. "
                    "Install it with: pip install 'uwsd[bert]'"
                ) from e
            self._model = SentenceTransformer(self.name, device=self.device)

    def encode(self, texts: List[str]) -> np.ndarray:
        self._ensure()
        emb = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)


class HashingEncoder:
    """Deterministic, dependency-free encoder (hashed char n-grams).

    Not competitive with neural encoders -- it exists so the entire pipeline is
    runnable and testable offline. Use a real encoder for benchmark numbers.
    """

    def __init__(self, dim: int = 512, ngram: int = 3):
        self.name = f"hashing-{dim}d-{ngram}gram"
        self.dim = dim
        self.ngram = ngram

    def encode(self, texts: List[str]) -> np.ndarray:
        import zlib

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            t = f" {text.lower()} "
            for j in range(len(t) - self.ngram + 1):
                gram = t[j : j + self.ngram]
                # zlib.crc32 is stable across processes (unlike built-in hash()).
                h = zlib.crc32(gram.encode("utf-8")) % self.dim
                out[i, h] += 1.0
        return _l2_normalize(out)


_ENCODER_FACTORIES = {
    "sentence-transformer": SentenceTransformerEncoder,
    "hashing": HashingEncoder,
}


# --------------------------------------------------------------------------- #
# The method
# --------------------------------------------------------------------------- #


class SubstitutionSimilarity(WSDMethod):
    """Unsupervised WSD by sense-substitution + sentence similarity."""

    name = "similarity"
    requires_train = False
    description = "Context-aware semantic similarity via sense substitution."

    def __init__(
        self,
        encoder: Optional[Encoder] = None,
        model: str = "all-MiniLM-L6-v2",
        backend: str = "sentence-transformer",
        temperature: float = 0.05,
        low_conf_margin: float = 0.02,
        **kwargs,
    ):
        super().__init__(model=model, backend=backend, **kwargs)
        if encoder is None:
            factory = _ENCODER_FACTORIES[backend]
            encoder = factory(model) if backend == "sentence-transformer" else factory()
        self.encoder = encoder
        self.temperature = temperature
        self.low_conf_margin = low_conf_margin

    # -- core scoring ---------------------------------------------------- #
    def _substitute(self, sentence: str, target: str, phrase: str) -> str:
        if not phrase:
            return sentence
        return sentence.replace(target, phrase)

    def _score_from_embeddings(
        self, orig_emb: np.ndarray, sub_embs: Dict[int, np.ndarray]
    ) -> Dict[int, float]:
        return {lid: float(np.dot(orig_emb, emb)) for lid, emb in sub_embs.items()}

    def _build_prediction(
        self, scores: Dict[int, float], task: WordTask, instance: Instance,
        phrases: Dict[int, str]
    ) -> Prediction:
        best = max(scores, key=scores.get)
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        margin = (ordered[0][1] - ordered[1][1]) if len(ordered) > 1 else 1.0
        probs = softmax_confidence(scores, self.temperature)
        conf = probs.get(best, 1.0)
        evidence = "; ".join(
            f"'{phrases[lid]}'→sim={scores[lid]:.3f}" for lid, _ in ordered[:3]
        )
        expl = (
            f"Replacing '{instance.target_token}' with the sense phrase "
            f"'{phrases[best]}' best preserved sentence meaning "
            f"(cosine={scores[best]:.3f}, margin={margin:.3f}). Candidates: {evidence}."
        )
        return Prediction(
            label_id=best,
            label=task.classes[best],
            scores=scores,
            confidence=conf,
            explanation=expl,
            low_confidence=margin < self.low_conf_margin,
            extra={"margin": margin, "probs": probs},
        )

    def predict(self, instance: Instance, task: WordTask) -> Prediction:
        target = instance.target_token
        phrases = {lid: task.sense_phrase(lid) for lid in task.label_ids}
        texts = [instance.sentence]
        order = []
        for lid in task.label_ids:
            texts.append(self._substitute(instance.sentence, target, phrases[lid]))
            order.append(lid)
        embs = self.encoder.encode(texts)
        orig_emb = embs[0]
        sub_embs = {lid: embs[i + 1] for i, lid in enumerate(order)}
        scores = self._score_from_embeddings(orig_emb, sub_embs)
        return self._build_prediction(scores, task, instance, phrases)

    # -- batched over the whole word task (much faster) ----------------- #
    def predict_task(
        self, task: WordTask, limit: Optional[int] = None
    ) -> List[Prediction]:
        items = task.test if limit is None else task.test[:limit]
        phrases = {lid: task.sense_phrase(lid) for lid in task.label_ids}

        # Build one big list of unique strings to encode.
        texts: List[str] = []
        index: Dict[str, int] = {}

        def add(s: str) -> int:
            if s not in index:
                index[s] = len(texts)
                texts.append(s)
            return index[s]

        plan = []  # per-instance: (orig_idx, {lid: sub_idx})
        for inst in items:
            target = inst.target_token
            oi = add(inst.sentence)
            subs = {
                lid: add(self._substitute(inst.sentence, target, phrases[lid]))
                for lid in task.label_ids
            }
            plan.append((oi, subs))

        if not texts:
            return []
        embs = self.encoder.encode(texts)

        preds: List[Prediction] = []
        for inst, (oi, subs) in zip(items, plan):
            orig_emb = embs[oi]
            scores = {lid: float(np.dot(orig_emb, embs[si])) for lid, si in subs.items()}
            preds.append(self._build_prediction(scores, task, inst, phrases))
        return preds

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "encoder": getattr(self.encoder, "name", "unknown"),
            "config": dict(self.config),
        }
