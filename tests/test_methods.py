"""Tests for the method registry, baselines, and the similarity pipeline."""

from uwsd.data import Instance, WordTask, load_coarsewsd20
from uwsd.evaluate import evaluate
from uwsd.methods import get_method, list_methods
from uwsd.methods.baselines import MostFrequentSense
from uwsd.methods.similarity import HashingEncoder, SubstitutionSimilarity


def _toy_task():
    return WordTask(
        word="bank",
        classes={0: "bank_(financial_institution)", 1: "bank_(geography)"},
        train=[
            Instance("bank", 0, "a b", gold_label=0),
            Instance("bank", 0, "a b", gold_label=0),
            Instance("bank", 0, "a b", gold_label=1),
        ],
        test=[Instance("bank", 0, "the river bank flooded .", gold_label=1)],
    )


def test_registry_lists_core_methods():
    names = {m["name"] for m in list_methods()}
    assert {"mfs", "random", "bert", "sbert", "similarity", "hashing"} <= names


def test_mfs_predicts_majority():
    task = _toy_task()
    mfs = MostFrequentSense()
    mfs.fit(task)
    pred = mfs.predict(task.test[0], task)
    assert pred.label_id == 0  # majority sense in train
    assert 0.0 <= pred.confidence <= 1.0


def test_similarity_pipeline_runs_offline():
    method = SubstitutionSimilarity(encoder=HashingEncoder())
    task = _toy_task()
    preds = method.predict_task(task)
    assert len(preds) == 1
    p = preds[0]
    assert p.label_id in task.classes
    assert set(p.scores) == set(task.label_ids)
    assert p.explanation  # interpretable by construction


def test_predict_single_instance_has_explanation():
    method = get_method("hashing")
    task = WordTask(word="java", classes={0: "javanese island",
                                          1: "programming language"})
    inst = Instance("java", 4, "i wrote it in java .")
    pred = method.predict(inst, task)
    assert pred.label in task.classes.values()
    assert "java" in pred.explanation.lower() or "sense" in pred.explanation.lower()


def test_evaluate_end_to_end_smoke():
    ds = load_coarsewsd20(words=["java"])
    method = get_method("hashing")
    manifest = evaluate(method, ds, limit=10, keep_predictions=True)
    assert manifest["metrics"]["n"] == 10
    assert 0.0 <= manifest["metrics"]["micro_accuracy"] <= 1.0
    assert "java" in manifest["predictions"]
    assert manifest["environment"]["uwsd_version"]
