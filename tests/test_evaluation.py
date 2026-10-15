import numpy as np
import pytest

from spotbot.evaluation import evaluate_predictions


def test_shared_metrics_and_domain_breakdown():
    scores = evaluate_predictions([0, 0, 1, 1], [0, 1, 1, 1], ["a", "a", "b", "b"])
    assert scores["accuracy"] == 0.75
    assert scores["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    np.testing.assert_array_equal(scores["confusion_matrix"], [[1, 1], [0, 2]])
    assert scores["per_class"].loc["human", "precision"] == 1
    assert scores["per_class"].loc["human", "recall"] == 0.5
    assert scores["per_class"].loc["ai", "precision"] == pytest.approx(2 / 3)
    assert scores["per_class"].loc["ai", "recall"] == 1
    assert scores["per_domain"]["a"]["accuracy"] == 0.5
    assert scores["per_domain"]["b"]["accuracy"] == 1
    assert scores["per_domain"]["b"]["macro_f1"] == 0.5


@pytest.mark.parametrize("truth,pred,sources", [
    ([], [], []), ([0], [0, 1], ["a"]), ([2], [0], ["a"]),
    ([0], [0.7], ["a"]), ([0], [0], [None]), ([[0]], [0], ["a"]),
])
def test_invalid_metric_inputs_are_rejected(truth, pred, sources):
    with pytest.raises(ValueError):
        evaluate_predictions(truth, pred, sources)
