import math

from moose.prob import choose_argmax, normalize_scores


def test_normalize_scores_sum_to_one():
    scores = {"A": 1.0, "B": 3.0}
    normalized = normalize_scores(scores)
    total = sum(normalized.values())
    assert math.isclose(total, 1.0, rel_tol=1e-9)


def test_normalize_scores_zeros():
    scores = {"A": 0.0, "B": 0.0}
    normalized = normalize_scores(scores)
    assert normalized == {"A": 0.5, "B": 0.5}


def test_choose_argmax():
    scores = {"A": 1.0, "B": 2.0}
    type_id, confidence, normalized = choose_argmax(scores)
    assert type_id == "B"
    assert math.isclose(confidence, normalized["B"], rel_tol=1e-9)
