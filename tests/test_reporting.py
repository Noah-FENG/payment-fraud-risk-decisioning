"""Report integration tests on explicitly synthetic fixtures, never portfolio results."""

import json

import numpy as np
import pandas as pd
import pytest

from fraud.io import digest, dump_json
from fraud.metrics import calibration_table, score_metrics
from fraud.policy import evaluate_policies, select_top_rate
from fraud.relations import audit_relations
from fraud.reporting import generate_report


@pytest.fixture
def report_run(tmp_path):
    run = tmp_path / "synthetic_fixture_run"
    processed = tmp_path / "synthetic_fixture_data"
    run.mkdir()
    processed.mkdir()
    frame = pd.DataFrame({
        "TransactionID": np.arange(12) + 100,
        "TransactionDT": [0, 10, 86400, 86410, 172800, 172810, 259200, 259210,
                          864000, 864001, 950400, 950401],
        "TransactionAmt": [10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120.],
        "isFraud": [0, 1] * 4 + [0, 1, 1, 0],
        "ProductCD": ["A", "B"] * 6,
        "DeviceType": ["mobile"] * 12,
        "DeviceInfo": ["dev-phone"] * 8 + ["test-only-phone"] * 4,
        "identity_missing": [0, 1] * 6,
        "card1": [1] * 11 + [2], "card2": [2] * 12,
        "card3": [3] * 12, "card5": [5] * 12, "addr1": [10] * 12,
    })
    split = ["train"] * 2 + ["tune"] * 2 + ["calibration"] * 2 + ["policy"] * 2 + ["test"] * 4
    frame.to_parquet(processed / "transactions.parquet", index=False)
    frame[["TransactionID", "TransactionDT"]].assign(split=split).to_parquet(run / "split_index.parquet", index=False)
    test = frame.iloc[-4:].copy()
    scores = [0.9, 0.2, 0.8, 0.3]
    test["score__xgboost"] = scores
    test["score__amount_rule"] = test.TransactionAmt
    test.to_parquet(run / "test_predictions.parquet", index=False)
    dump_json(run / "evaluation.json", {"status": "evaluated", "chosen_model": "xgboost", "exposure": {"status": "synthetic_test_fixture_only"}})
    dump_json(run / "development.json", {"chosen_model": "xgboost", "data_sha256": digest(processed / "transactions.parquet"), "models": {
        "xgboost": {"training_seconds": 1.0, "feature_count": 10}, "amount_rule": {"training_seconds": 0.0},
    }})
    dump_json(run / "relation_audit.json", audit_relations(frame.iloc[:8]))
    selected = select_top_rate(test, scores, 0.5)
    metrics = [{"model": "xgboost", "period": "test", "calibration": "sigmoid",
                **score_metrics(test.isFraud, scores, test.TransactionAmt, selected)}]
    pd.DataFrame(metrics).to_csv(run / "test_metrics.csv", index=False)
    pd.DataFrame(metrics).assign(period="policy").to_csv(run / "development_metrics.csv", index=False)
    scenarios = [{"name": "synthetic_fixture", "loss_rate": 0.5, "review_effectiveness": 0.5,
                  "review_cost": 1.0, "false_positive_cost": 2.0}]
    policies, costs = evaluate_policies(test, scores, 0.5, [0.25, 0.5], scenarios)
    policies.assign(model="xgboost", period="test").to_csv(run / "policy_metrics.csv", index=False)
    costs.assign(model="xgboost", period="test").to_csv(run / "cost_sensitivity.csv", index=False)
    calibration_table(test.isFraud, scores).assign(model="xgboost", calibration="sigmoid", period="test").to_csv(run / "calibration.csv", index=False)
    config = {"run_dir": str(run), "processed_dir": str(processed),
              "primary_review_rate": 0.5, "batch_seconds": 86400}
    return config, frame


def test_report_requires_completed_evaluation(tmp_path):
    with pytest.raises(FileNotFoundError, match="final evaluation"):
        generate_report({"run_dir": str(tmp_path)})


def test_reporting_reconciles_grain_shares_and_preserves_global_review_decisions(report_run):
    config, frame = report_run
    result = generate_report(config)
    run = pd.io.common.stringify_path(config["run_dir"])
    risk = pd.read_csv(run + "/tableau/risk_segments.csv")
    for (period, _), group in risk.groupby(["period", "dimension"]):
        expected = 4 if period == "test" else 2
        assert group.n.sum() == expected
        assert group.fraud_count_share.sum() == pytest.approx(1)
        assert group.fraud_amount_share.sum() == pytest.approx(1)
    stability = pd.read_csv(run + "/tableau/stability.csv")
    product = stability[(stability.model == "xgboost") & (stability.dimension == "ProductCD")]
    # Whole-day ranking flags A twice. Re-ranking within each product would change this.
    assert product.selected_n.sum() == 2
    assert product.loc[product.segment == "A", "selected_n"].iloc[0] == 2
    assert product.loc[product.segment == "B", "selected_n"].iloc[0] == 0
    assert product.tp.sum() == 1
    assert product.fp.sum() == 1
    assert product.fn.sum() == 1
    devices = risk[(risk.period == "test") & (risk.dimension == "DeviceInfo_group")]
    assert devices.segment.tolist() == ["Other / unseen in development"]
    errors = pd.read_csv(run + "/tableau/error_segments.csv")
    assert (errors.model == "xgboost").all()
    assert result["test_rows"] == 4
    assert len(frame) == 12


def test_report_outputs_traceable_sources_and_local_cases(report_run):
    config, _ = report_run
    result = generate_report(config)
    from pathlib import Path
    run = Path(config["run_dir"])
    schema = json.loads((run / "tableau/schema.json").read_text())
    assert result["csv_outputs"] == len(schema["files"])
    assert schema["review_definition"].endswith("no within-group reranking")
    assert all(len(info["sha256"]) == 64 for info in schema["files"].values())
    cases = pd.read_csv(run / "error_cases_local_only.csv")
    assert set(cases.error_type) == {"FP", "FN"}
    assert len(cases) == 2
    assert "synthetic_test_fixture_only" in (run / "report.md").read_text(encoding="utf-8")
    assert "No GraphSAGE experiment was run" in (run / "gnn_decision.md").read_text(encoding="utf-8")
    assert len(list((run / "figures").glob("*.png"))) == 4
    assert (run / "resume_bullets.md").exists()
    assert not any(name.startswith("TransactionID") for info in schema["files"].values() for name in [c["name"] for c in info["columns"]])


def test_report_rejects_changed_prepared_data(report_run):
    config, frame = report_run
    from pathlib import Path
    frame.loc[0, "TransactionAmt"] = 999999
    frame.to_parquet(Path(config["processed_dir"]) / "transactions.parquet", index=False)
    with pytest.raises(ValueError, match="Prepared data changed"):
        generate_report(config)
