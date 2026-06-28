"""Tests for metrics, confidence intervals and significance testing."""

from uwsd import metrics


def test_accuracy_basic():
    assert metrics.accuracy([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75


def test_classification_report_perfect():
    rep = metrics.classification_report([0, 1, 0, 1], [0, 1, 0, 1])
    assert rep.accuracy == 1.0
    assert rep.macro_f1 == 1.0
    assert rep.weighted_f1 == 1.0


def test_classification_report_known_values():
    # 2 classes, one confusion
    gold = [0, 0, 1, 1]
    pred = [0, 1, 1, 1]
    rep = metrics.classification_report(gold, pred)
    assert rep.accuracy == 0.75
    by_label = {c.label: c for c in rep.per_class}
    # class 1: tp=2, fp=1, fn=0 -> precision 2/3, recall 1.0
    assert abs(by_label[1].precision - 2 / 3) < 1e-9
    assert by_label[1].recall == 1.0


def test_bootstrap_ci_brackets_point():
    correct = [1] * 80 + [0] * 20
    point, low, high = metrics.bootstrap_accuracy_ci(correct, n_resamples=500, seed=0)
    assert abs(point - 0.8) < 1e-9
    assert low <= point <= high
    assert 0.0 <= low <= high <= 1.0


def test_paired_bootstrap_detects_difference():
    # A clearly better than B
    a = [1] * 90 + [0] * 10
    b = [1] * 60 + [0] * 40
    res = metrics.paired_bootstrap_test(a, b, n_resamples=2000, seed=0)
    assert res["delta_accuracy"] > 0
    assert res["p_value"] < 0.05


def test_mcnemar_symmetric_is_nonsignificant():
    a = [1, 0, 1, 0, 1, 0]
    b = [0, 1, 0, 1, 0, 1]
    res = metrics.mcnemar_test(a, b)
    assert res["p_value"] == 1.0


def test_aggregate_by_word():
    rows = [
        ("apple", [0, 0, 1], [0, 0, 1]),
        ("bank", [0, 1], [0, 0]),
    ]
    agg = metrics.aggregate_by_word(rows)
    assert agg["n"] == 5
    assert agg["hits"] == 4
    assert abs(agg["micro_accuracy"] - 0.8) < 1e-9
    assert "apple" in agg["per_word"]
