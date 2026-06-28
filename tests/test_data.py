"""Tests for the CoarseWSD-20 loader and data model."""

from uwsd.data import Instance, WordTask, load_coarsewsd20


def test_load_java_subset():
    ds = load_coarsewsd20(words=["java"])
    assert ds.name == "CoarseWSD-20"
    assert ds.words == ["java"]
    task = ds.tasks["java"]
    assert task.num_senses == 2
    # known counts for the bundled java split (final line lacks trailing newline)
    assert len(task.test) == 1929
    assert len(task.train) == 4504
    for inst in task.test[:50]:
        assert inst.gold_label in task.classes


def test_sense_phrase_cleaning():
    task = WordTask(
        word="java",
        classes={0: "java_javanese_island", 1: "java_java_(programming_language)"},
    )
    assert task.sense_phrase(0) == "javanese island"
    assert task.sense_phrase(1) == "programming language"


def test_instance_target_token():
    inst = Instance(word="java", token_index=2, sentence="it is found on java .")
    assert inst.target_token == "found"
    inst2 = Instance(word="java", token_index=4, sentence="it is found on java .")
    assert inst2.target_token == "java"


def test_dataset_full_counts():
    ds = load_coarsewsd20()
    assert len(ds) == 20
    assert ds.num_test_instances == 10196


# ---------------------------------------------------------------------------
# end of file (sentinel comment guards against mount last-line truncation)
# ---------------------------------------------------------------------------
