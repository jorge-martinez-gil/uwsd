# Contributing to UWSD

Thanks for helping make this the standard open benchmark for unsupervised word
sense disambiguation. Contributions of new methods, encoders, datasets, metrics,
tutorials, and bug fixes are all welcome.

## Development setup

```bash
git clone https://github.com/jorge-martinez-gil/uwsd
cd uwsd
pip install -e ".[dev]"     # add [bert] / [all] for neural methods
pytest                      # run the test suite
```

## Adding a new method

Register a class in `uwsd/methods/` and it becomes available from the CLI:

```python
from uwsd.methods import register_method, WSDMethod, Prediction

@register_method("my-method", description="One-line summary.")
class MyMethod(WSDMethod):
    def predict(self, instance, task):
        scores = {lid: my_score(instance, task, lid) for lid in task.label_ids}
        best = max(scores, key=scores.get)
        return Prediction(
            label_id=best, label=task.classes[best], scores=scores,
            confidence=scores[best], explanation="why this sense",
        )
```

Guidelines:

- **Interpretability is required.** Populate `scores`, `confidence`, and a short
  `explanation` so users can see *why* a sense was chosen.
- **No fabricated results.** Numbers in the README/tables must come from an
  actual `uwsd run` manifest. Never hand-edit metrics.
- **Add a test.** At minimum a smoke test that runs your method on one word.
- Keep heavy dependencies optional (add them to an extra in `pyproject.toml`).

## Adding a dataset

Implement a loader returning a `uwsd.data.Dataset` of `WordTask` objects (see
`uwsd/data.py`). Datasets in the CoarseWSD-20 directory layout work out of the
box via `UWSD_DATA`.

## Reporting results

Run with `--keep-correct` so others can re-check significance:

```bash
uwsd run --method my-method --output results/my-method.json --keep-correct
uwsd compare results/my-method.json results/mfs.json
```

## Pull requests

- Run `pytest` and make sure it passes.
- Describe the scientific motivation and include a manifest for any new numbers.
- One focused change per PR where possible.
