"""Hand-computed cases: synthetic fixtures validate code, not project results."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from fraud.metrics import calibration_table, expected_calibration_error, score_metrics
from fraud.policy import (
    evaluate_policies, fit_cost_threshold, fit_threshold, fixed_threshold, scenario_costs,
    select_top_rate,
)


def test_metric_denominators_and_amount_capture():
    result = score_metrics(
        [1, 0, 1, 0], [0.9, 0.8, 0.2, 0.1], [100, 20, 300, 50], [1, 1, 0, 0]
    )
    assert result["tp"] == result["fp"] == result["fn"] == result["tn"] == 1
    assert result["precision"] == result["recall"] == result["normal_flag_rate"] == 0.5
    assert result["captured_fraud_amount_share"] == 0.25
    assert result["fraud_amount_total"] == 400
    assert result["captured_fraud_amount"] == 100
    assert result["average_precision"] == pytest.approx(
        average_precision_score([1, 0, 1, 0], [0.9, 0.8, 0.2, 0.1])
    )
    assert result["brier"] == pytest.approx((0.01 + 0.64 + 0.64 + 0.01) / 4)


def test_undefined_denominators_and_empty_groups():
    result = score_metrics([0, 0], [0.1, 0.2], [0, 0], [0, 0])
    assert result["precision"] is result["recall"] is result["roc_auc"] is None
    assert result["average_precision"] is result["captured_fraud_amount_share"] is None
    assert result["normal_flag_rate"] == 0
    assert score_metrics([], [], [], [])["fraud_rate"] is None
    assert score_metrics([1], [10], [20])["brier"] is None
    assert expected_calibration_error([], []) is None


def test_invalid_labels_amounts_and_selection_rejected():
    with pytest.raises(ValueError):
        score_metrics([1, np.nan], [0.1, 0.2], [1, 2])
    with pytest.raises(ValueError):
        score_metrics([1], [0.1], [-1])
    with pytest.raises(ValueError):
        score_metrics([1], [np.inf], [1])
    with pytest.raises(ValueError):
        score_metrics([1], [0.1], [1], [2])


def test_calibration_includes_probability_one_and_empty_bins():
    table = calibration_table([0, 1, 1], [0.0, 0.15, 1.0], n_bins=10)
    assert table["count"].sum() == 3
    assert table.loc[9, "count"] == 1
    assert table.loc[9, "observed_fraud_rate"] == 1
    assert np.isnan(table.loc[2, "mean_predicted"])
    assert expected_calibration_error([0, 1, 1], [0, 0.15, 1]) == pytest.approx(0.85 / 3)
    with pytest.raises(ValueError):
        calibration_table([1], [1.01])


def test_per_batch_capacity_ties_and_order_independence():
    frame = pd.DataFrame({
        "TransactionID": [4, 2, 3, 1, 8, 5, 6, 7, 9],
        "TransactionDT": [1, 2, 3, 4, 86401, 86402, 86403, 86404, 172800],
        "score": [0.8] * 9,
    }, index=np.arange(10, 19))
    selected = select_top_rate(frame, frame["score"], 0.5)
    assert set(frame.loc[selected, "TransactionID"]) == {1, 2, 5, 6}
    shuffled = frame.sample(frac=1, random_state=42)
    after_shuffle = select_top_rate(shuffled, shuffled["score"], 0.5)
    assert set(shuffled.loc[after_shuffle, "TransactionID"]) == {1, 2, 5, 6}
    assert not select_top_rate(frame, frame["score"], 0).any()
    assert select_top_rate(frame, frame["score"], 1).all()


def test_fixed_threshold_fit_respects_development_capacity_with_ties():
    scores = [0.9, 0.8, 0.8, 0.1]
    threshold = fit_threshold([1, 0, 1, 0], scores, [1, 2, 3, 4], capacity_rate=0.5)
    selected = fixed_threshold(scores, threshold)
    assert selected.tolist() == [True, False, False, False]
    no_reviews = fit_threshold([1, 0, 1, 0], scores, [1, 2, 3, 4], capacity_rate=0)
    assert not fixed_threshold(scores, no_reviews).any()


def test_scenario_cost_has_no_double_counting():
    scenario = {"name": "example", "loss_rate": 0.5, "review_effectiveness": 0.8,
                "review_cost": 2, "false_positive_cost": 3}
    result = scenario_costs([1, 0, 1], [100, 50, 200], [1, 1, 0], scenario)
    assert result["missed_potential_loss"] == 100
    assert result["reviewed_residual_potential_loss"] == pytest.approx(10)
    assert result["total_review_cost"] == 4
    assert result["total_false_positive_cost"] == 3
    assert result["hypothetical_loss_reduction"] == 40
    assert result["scenario_total_cost"] == pytest.approx(117)
    assert result["no_review_scenario_cost"] == 150
    assert result["scenario_net_benefit"] == pytest.approx(33)
    assert result["scenario_total_cost"] == pytest.approx(
        result["no_review_scenario_cost"] - result["hypothetical_loss_reduction"]
        + result["total_review_cost"] + result["total_false_positive_cost"]
    )


def test_cost_threshold_and_policy_evaluation_do_not_fit_test_thresholds():
    frame = pd.DataFrame({"TransactionID": [1, 2, 3, 4], "TransactionDT": [1, 2, 3, 4],
                          "isFraud": [1, 0, 1, 0], "TransactionAmt": [100, 10, 200, 10]})
    scores = np.array([0.9, 0.7, 0.6, 0.2])
    scenario = {"name": "example", "loss_rate": 1, "review_effectiveness": 1,
                "review_cost": 1, "false_positive_cost": 1}
    threshold = fit_cost_threshold(frame["isFraud"], scores, frame["TransactionAmt"], scenario)
    assert fixed_threshold(scores, threshold).tolist() == [True, True, True, False]
    metrics, costs = evaluate_policies(frame, scores, threshold, [0.5], [scenario])
    assert metrics["policy"].tolist() == ["fixed_threshold", "top_rate_0.5"]
    assert metrics["selected_n"].tolist() == [3, 2]
    assert costs["scenario"].tolist() == ["example", "example"]
    assert costs.loc[0, "scenario_total_cost"] == 4


@pytest.mark.parametrize("field,value", [("loss_rate", 1.1), ("review_cost", -1),
                                          ("review_effectiveness", -0.1)])
def test_invalid_scenarios_rejected(field, value):
    scenario = {"loss_rate": 0.5, "review_effectiveness": 0.8,
                "review_cost": 2, "false_positive_cost": 3}
    scenario[field] = value
    with pytest.raises(ValueError):
        scenario_costs([1], [100], [1], scenario)
