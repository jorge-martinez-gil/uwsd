# UWSD: Unsupervised Word Sense Disambiguation Benchmark

**A reproducible benchmark and experimentation platform for unsupervised word sense disambiguation (WSD) via context-aware semantic similarity.**

[![arXiv preprint](https://img.shields.io/badge/arXiv-2305.03520-brightgreen.svg)](https://arxiv.org/abs/2305.03520) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Citations](https://img.shields.io/badge/citations-3-blue)](https://scholar.google.com/citations?view_op=view_citation&hl=en&citation_for_view=X1pRUYcAAAAJ:7XUxBq3GufIC)

This repository accompanies Jorge Martinez-Gil's paper *Context-Aware Semantic Similarity Measurement for Unsupervised Word Sense Disambiguation* ([arXiv:2305.03520](https://arxiv.org/abs/2305.03520); [Medium summary](https://medium.com/@jorgemarcc/applications-of-context-aware-semantic-similarity-9c62492be392)). It has grown from the paper's original scripts into an **installable Python package with a one-command benchmark CLI**, so that anyone proposing a new unsupervised WSD method can evaluate it on a common footing and compare against published baselines.

![Summary](uwsd.png)

---

## Table of contents

- [What problem does this solve?](#what-problem-does-this-solve)
- [Why unsupervised WSD?](#why-unsupervised-wsd)
- [How this differs from supervised WSD](#how-this-differs-from-supervised-wsd)
- [Install](#install)
- [Quickstart (CLI)](#quickstart-cli)
- [Reproducing the published experiments](#reproducing-the-published-experiments)
- [Evaluate your own WSD algorithm](#evaluate-your-own-wsd-algorithm)
- [Add a new embedding model / similarity measure](#add-a-new-embedding-model--similarity-measure)
- [Python API](#python-api)
- [Interpretability](#interpretability)
- [Results](#results)
- [Dataset](#dataset)
- [Citation](#citation)
- [Research that has cited this work](#research-that-has-cited-this-work)
- [Contributing](#contributing)
- [License](#license)

---

## What problem does this solve?

Many words have more than one meaning. *Java* can be a programming language or an island; *bank* can be a financial institution or the side of a river. **Word sense disambiguation (WSD)** is the task of selecting the intended sense of an ambiguous word given its context. It underpins machine translation, information retrieval, question answering, and knowledge-graph construction.

This project provides:

1. A faithful, well-tested implementation of the **context-aware semantic-similarity** approach to *unsupervised* WSD.
2. A **standard benchmark harness** so different methods, encoders, and similarity measures are evaluated identically.
3. **Reproducible, traceable results**: every run emits a JSON manifest containing the metrics, a 95% bootstrap confidence interval, the environment, and the git commit.

## Why unsupervised WSD?

Supervised WSD needs large amounts of sense-annotated text, which is expensive, language-specific, and quickly goes stale as senses drift. **Unsupervised** WSD needs no sense-labelled training data — only a sense inventory (candidate meanings) and a sentence encoder. That makes it attractive for low-resource languages, technical/biomedical/software terminology, and rapidly evolving domains.

The core idea here: to disambiguate a word, **substitute each candidate sense's phrase into the sentence and measure how well meaning is preserved** (via embedding cosine similarity). The sense whose substitution keeps the sentence closest to the original wins. No training labels are used.

## How this differs from supervised WSD

| | Supervised WSD | This (unsupervised) |
| --- | --- | --- |
| Needs sense-annotated training data | Yes | **No** |
| Adapts to new domains/languages | Retrain | Swap the encoder / sense inventory |
| What it learns from | Labelled examples | Pretrained sentence embeddings |
| Baselines here | MFS (uses train labels) | substitution-similarity, random |

The Most-Frequent-Sense (MFS) baseline *does* use training labels and is included as a strong reference point that unsupervised methods aim to beat.

---

## Install

```bash
git clone https://github.com/jorge-martinez-gil/uwsd
cd uwsd
pip install -e .            # core (numpy only)
pip install -e ".[bert]"    # + sentence-transformers for BERT/SBERT methods
pip install -e ".[all]"     # + gensim/WMD and tensorflow-hub/USE
```

Python ≥ 3.9. The CoarseWSD-20 dataset is bundled in this repo, so no download is required to get started.

## Quickstart (CLI)

```bash
# List the available methods
uwsd list-methods

# Run the Most-Frequent-Sense baseline on the full benchmark
uwsd run --method mfs --output results/mfs.json

# Run the context-aware similarity method with a sentence-transformer
uwsd run --method bert --model all-MiniLM-L6-v2 --output results/bert.json

# Build a publication-ready comparison table (Markdown or LaTeX)
uwsd report results/*.json --format markdown
uwsd report results/*.json --format latex --caption "Unsupervised WSD on CoarseWSD-20."

# Statistical significance between two systems (paired bootstrap + McNemar)
uwsd run --method mfs    --output results/mfs.json    --keep-correct
uwsd run --method bert   --output results/bert.json   --keep-correct
uwsd compare results/bert.json results/mfs.json

# Disambiguate a single sentence interactively, with an explanation
uwsd predict --method bert \
  --sentence "i wrote the backend in java ." --target java \
  --senses "isl=javanese island" "prog=programming language"
```

Every `run` writes a **reproducibility manifest** (`--output`) recording the method/encoder config, per-word and overall metrics, a 95% bootstrap CI, the environment versions, the git commit, and a timestamp. Add `--keep-predictions` to store every per-instance prediction with its explanation.

## Reproducing the published experiments

The harness reproduces the paper's baselines from the bundled data with a single command. As an integrity check, the **Most-Frequent-Sense baseline reproduces the paper's number exactly**:

```
$ uwsd run --method mfs
=== mfs on CoarseWSD-20 ===
  instances : 10,196
  hits      : 7,487
  accuracy  : 73.43%  (95% CI [72.57, 74.25])
```

This matches the **7,487 hits / 73.43%** reported in the paper. The neural methods (`bert`, `sbert`) require `pip install -e ".[bert]"` and download their model weights on first use; run them the same way (`uwsd run --method bert --model all-MiniLM-L6-v2`).

## Evaluate your own WSD algorithm

1. Implement a method (see below) or point an existing one at your data.
2. Run it: `uwsd run --method <name> --output results/<name>.json --keep-correct`.
3. Compare against baselines with significance testing: `uwsd compare results/<name>.json results/mfs.json`.
4. Drop the manifests into a table: `uwsd report results/*.json --format latex`.

Because every method is evaluated through the same loader, metrics, and confidence intervals, comparisons are apples-to-apples.

## Add a new embedding model / similarity measure

Adding a method is a few lines — it then works from the CLI automatically:

```python
from uwsd.methods import register_method, WSDMethod, Prediction

@register_method("my-method", description="My great WSD idea.")
class MyMethod(WSDMethod):
    def predict(self, instance, task):
        scores = {lid: my_score(instance, task, lid) for lid in task.label_ids}
        best = max(scores, key=scores.get)
        return Prediction(label_id=best, label=task.classes[best], scores=scores,
                          confidence=scores[best], explanation="...")
```

To benchmark a **new embedding model**, just pass it: `uwsd run --method bert --model <hf-model-name>`. To benchmark a **new similarity measure**, subclass `uwsd.methods.similarity.SubstitutionSimilarity` and override the scoring, or plug in a custom `Encoder` (any object with `encode(list[str]) -> np.ndarray`).

## Python API

```python
from uwsd import load_coarsewsd20, get_method
from uwsd.evaluate import evaluate

ds = load_coarsewsd20()                       # bundled benchmark
method = get_method("bert", model="all-MiniLM-L6-v2")
manifest = evaluate(method, ds, keep_predictions=True)
print(manifest["metrics"]["micro_accuracy"], manifest["metrics"]["accuracy_ci95"])
```

## Interpretability

Every prediction is interpretable by construction. It carries the candidate senses, their similarity scores, a confidence value, a human-readable explanation of *why* the sense was chosen, and a low-confidence flag when the top candidates are close:

```
Predicted: programming language  (confidence 71.0%)
Why: Replacing 'java' with the sense phrase 'programming language' best preserved
     sentence meaning (cosine=0.95, margin=0.06). Candidates:
     'programming language'->sim=0.95; 'javanese island'->sim=0.89.
```

## Results

**Reported in the paper** (CoarseWSD-20, accuracy):

| Strategy | Hits | Accuracy |
| --- | --- | --- |
| UWSD+BERT | 7,927 | 77.74% |
| MFS-Baseline | 7,487 | 73.43% |
| UWSD+USE | 7,335 | 71.94% |
| UWSD+ELMo | 7,010 | 68.75% |
| UWSD+WMD | 5,868 | 57.55% |
| RO-Baseline | 4,459 | 43.73% |

**Reproduced by this harness** (with 95% bootstrap CIs; `git` commit and full manifest under `results/`):

| Method | Hits | Accuracy | 95% CI |
| --- | --- | --- | --- |
| MFS (uses train labels) | 7,487/10,196 | 73.43% | [72.57, 74.25] |
| Random (seeded) | 4,506/10,196 | 44.19% | [43.26, 45.18] |

The MFS reproduction matches the paper exactly. Neural baselines (`bert`, `sbert`, `use`, `wmd`) reproduce the same way once their optional dependencies and model weights are available; run `uwsd run --method bert` to regenerate them and append to the table with `uwsd report`.

## Dataset

[CoarseWSD-20](https://github.com/danlou/bert-disambiguation) (Loureiro et al., 2021) is a coarse-grained WSD benchmark over 20 ambiguous words (e.g. *bank*, *java*, *crane*), with 10,196 test instances. It is bundled under `CoarseWSD-20/`. Set the `UWSD_DATA` environment variable to point the loader at a different copy or another dataset in the same layout.

## Citation

```bibtex
@inproceedings{martinez2023b,
  author     = {Jorge Martinez-Gil},
  title      = {Context-Aware Semantic Similarity Measurement for Unsupervised Word Sense Disambiguation},
  journal    = {CoRR},
  volume     = {abs/2305.03520},
  year       = {2023},
  url        = {https://arxiv.org/abs/2305.03520},
  doi        = {https://doi.org/10.48550/arXiv.2305.03520},
  eprinttype = {arXiv},
  eprint     = {2305.03520}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is also provided.

## Research that has cited this work

1. **[Pantip Multi-turn Datasets Generating from Thai Large Social Platform Forum Using Sentence Similarity Techniques](https://ieeexplore.ieee.org/iel8/10799229/10799211/10799403.pdf)** — A. Sae-Oueng, K. Kerdthaisong, et al. *Joint Symposium*, 2024 (IEEE).
2. **[Assessing GPT's Potential for Word Sense Disambiguation: A Quantitative Evaluation on Prompt Engineering Techniques](https://doi.org/10.1109/icsgrc62081.2024.10691283)** — D. Sumanathilaka, N. Micallef, J. Hough. *IEEE ICSGRC*, 2024.
3. **[GlossGPT: GPT for Word Sense Disambiguation using Few-shot Chain-of-Thought Prompting](https://www.sciencedirect.com/science/article/pii/S1877050925008385)** — D. Sumanathilaka, N. Micallef, J. Hough. *Procedia Computer Science*, 2025 (Elsevier).

## Contributing

Contributions of new methods, encoders, datasets, and metrics are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Run the test suite with `pytest`.

## License

Released under the MIT License. [View License](LICENSE).
