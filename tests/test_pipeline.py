"""End-to-end software verification on clearly synthetic, temporary fixtures.

No fixture metric is published as IEEE-CIS evidence. The source fingerprint is
fixed per fixture because the repository may be edited concurrently during tests.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from fraud.data import ingest
from fraud.io import read_json
from fraud import pipeline


@pytest.fixture
def software_run(tmp_path, monkeypatch):
    root = tmp_path / "synthetic_software_test"
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    n = 500
    tx = pd.DataFrame({
        "TransactionID": np.arange(10000, 10000 + n),
        "TransactionDT": np.arange(n) * 1000,
        "TransactionAmt": 5 + (np.arange(n) * 17) % 250,
        "isFraud": (np.arange(n) % 7 == 0).astype(int),
        "ProductCD": np.where(np.arange(n) % 3 == 0, "W", "C"),
        "card1": np.arange(n) % 20,
        "card2": np.where(np.arange(n) % 13 == 0, np.nan, np.arange(n) % 5),
        "card3": np.full(n, 150),
        "card5": np.arange(n) % 4,
        "addr1": np.arange(n) % 10,
        "C1": np.arange(n) % 11,
        "V1": np.arange(n) % 6,
    })
    tx.to_csv(raw / "train_transaction.csv", index=False)
    identity = pd.DataFrame({
        "TransactionID": tx.loc[tx.index % 2 == 0, "TransactionID"],
        "DeviceType": "desktop",
        "DeviceInfo": np.where(np.arange(n // 2) % 2 == 0, "test_device_a", "test_device_b"),
        "id_01": np.arange(n // 2) % 4,
    })
    identity.to_csv(raw / "train_identity.csv", index=False)
    default = Path(__file__).resolve().parents[1] / "config" / "default.json"
    config = read_json(default)
    config.update({
        "raw_dir": str(raw),
        "processed_dir": str(root / "data" / "processed"),
        "run_dir": str(root / "artifacts" / "first"),
        "project_root": str(root),
        "config_path": str(root / "config" / "test.json"),
        "threads": 1,
        "logistic_max_iter": 2000,
        "primary_review_rate": 0.1,
        "review_rates": [0.1, 0.2],
        "xgboost_candidates": [{
            "n_estimators": 2, "max_depth": 2, "learning_rate": 0.1,
            "min_child_weight": 1,
        }],
        "shap_sample": 5,
    })
    monkeypatch.setattr(pipeline, "code_fingerprint", lambda: {"software_fixture": "stable"})
    audit = ingest(Path(config["raw_dir"]), Path(config["processed_dir"]))
    manifest = pipeline.develop(config)
    return {"root": root, "config": config, "audit": audit, "manifest": manifest}


def test_prepare_develop_freeze_evaluate_and_exposure_guards(software_run):
    config = software_run["config"]
    run = Path(config["run_dir"])
    assert software_run["audit"]["joined_rows"] == 500
    manifest = software_run["manifest"]
    assert manifest["status"] == "developed"
    assert "test_metrics" not in manifest
    assert {row["split"] for row in manifest["split_summary"]} == {
        "train", "tune", "calibration", "policy", "test",
    }
    metrics = pd.read_csv(run / "development_metrics.csv")
    assert metrics["period"].eq("policy").all()
    assert not (run / "test_metrics.csv").exists()
    index = pd.read_parquet(run / "split_index.parquet")
    policy = pd.read_parquet(run / "policy_predictions.parquet")
    assert set(policy.TransactionID) == set(index.loc[index["split"] == "policy", "TransactionID"])
    plan = pipeline.freeze(config)
    assert plan["chosen_model"] == manifest["chosen_model"]
    evaluated = pipeline.evaluate(config)
    assert evaluated["status"] == "evaluated"
    assert evaluated["exposure"]["status"] == "first_recorded_local_evaluation"
    assert evaluated["final_test_selection_permitted"] is False
    scored = pd.read_parquet(run / "test_predictions.parquet")
    assert set(scored.TransactionID) == set(index.loc[index["split"] == "test", "TransactionID"])
    for name in manifest["models"]:
        assert f"score__{name}" in scored
    with pytest.raises(FileExistsError):
        pipeline.evaluate(config)
    with pytest.raises(FileExistsError):
        pipeline.develop(config)

    # A separate run on the same prepared data is explicitly labeled re-exposure.
    later = copy.deepcopy(config)
    later["run_dir"] = str(software_run["root"] / "artifacts" / "later")
    pipeline.develop(later)
    pipeline.freeze(later)
    second = pipeline.evaluate(later)
    assert second["exposure"]["status"] == "previously_exposed"
    ledger = Path(config["processed_dir"]) / "test_exposure_ledger.jsonl"
    entries = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(entries) == 2
    assert entries[0]["data_sha256"] == entries[1]["data_sha256"]


def test_frozen_config_model_source_and_data_mutations_are_rejected(software_run, monkeypatch):
    config = software_run["config"]
    run = Path(config["run_dir"])
    pipeline.freeze(config)
    altered_config = {**config, "primary_review_rate": 0.2}
    with pytest.raises(ValueError, match="config/source"):
        pipeline.verify_frozen(altered_config)
    monkeypatch.setattr(pipeline, "code_fingerprint", lambda: {"software_fixture": "changed"})
    with pytest.raises(ValueError, match="config/source"):
        pipeline.verify_frozen(config)
    monkeypatch.setattr(pipeline, "code_fingerprint", lambda: {"software_fixture": "stable"})

    model_path = next((run / "models").glob("*.joblib"))
    original_bytes = model_path.read_bytes()
    model_path.write_bytes(original_bytes + b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        pipeline.verify_frozen(config)
    model_path.write_bytes(original_bytes)
    prepared = Path(config["processed_dir"]) / "transactions.parquet"
    original_data = prepared.read_bytes()
    prepared.write_bytes(original_data + b"changed")
    with pytest.raises(ValueError, match="data mismatch"):
        pipeline.verify_frozen(config)
    prepared.write_bytes(original_data)
    pipeline.verify_frozen(config)


def test_development_excludes_final_labels_and_calibration_uses_only_its_block(
    software_run, monkeypatch,
):
    config = software_run["config"]
    run = Path(config["run_dir"])
    index = pipeline.split_index(config)
    test_min = index.loc[index["split"] == "test", "TransactionDT"].min()
    original_read = pd.read_parquet
    reads = []

    def logged_read(*args, **kwargs):
        reads.append(kwargs.copy())
        return original_read(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", logged_read)
    development = pipeline._load_development(config, index)
    assert development.TransactionDT.max() < test_min
    assert set(development["split"]) == {"train", "tune", "calibration", "policy"}
    assert reads[0]["filters"] == [("TransactionDT", "<", test_min)]
    monkeypatch.setattr(pd, "read_parquet", original_read)

    # Change every final label and retrain in a fresh temporary run. Every learned
    # parameter and development probability must be unchanged; timing/hash metadata
    # is expected to differ and is intentionally excluded from this comparison.
    prepared = Path(config["processed_dir"]) / "transactions.parquet"
    frame = original_read(prepared)
    final_ids = set(index.loc[index["split"] == "test", "TransactionID"])
    final_mask = frame.TransactionID.isin(final_ids)
    frame.loc[final_mask, "isFraud"] = 1 - frame.loc[final_mask, "isFraud"]
    frame.to_parquet(prepared, index=False)
    later = copy.deepcopy(config)
    later["run_dir"] = str(software_run["root"] / "artifacts" / "changed_final_labels")
    calibration_labels_seen = []
    original_fit = pipeline.SigmoidCalibrator.fit

    def logged_calibration_fit(self, probabilities, labels):
        calibration_labels_seen.append(np.asarray(labels).copy())
        return original_fit(self, probabilities, labels)

    monkeypatch.setattr(pipeline.SigmoidCalibrator, "fit", logged_calibration_fit)
    later_manifest = pipeline.develop(later)
    assert later_manifest["chosen_model"] == software_run["manifest"]["chosen_model"]
    later_run = Path(later["run_dir"])
    pd.testing.assert_frame_equal(
        pd.read_parquet(run / "policy_predictions.parquet"),
        pd.read_parquet(later_run / "policy_predictions.parquet"),
    )
    expected_labels = development.loc[development["split"] == "calibration", "isFraud"].to_numpy()
    assert len(calibration_labels_seen) == 4
    for labels in calibration_labels_seen:
        np.testing.assert_array_equal(labels, expected_labels)
    for name, specification in software_run["manifest"]["models"].items():
        if specification["kind"] == "rule":
            continue
        before = joblib.load(run / "models" / f"{name}.joblib")
        after = joblib.load(later_run / "models" / f"{name}.joblib")
        np.testing.assert_array_equal(before.calibrator.model.coef_, after.calibrator.model.coef_)


def test_prepare_rejects_modified_model_before_freeze(software_run):
    """Development metrics must correspond to the model that is later frozen."""
    config = software_run["config"]
    run = Path(config["run_dir"])
    model_path = next((run / "models").glob("*.joblib"))
    model_path.write_bytes(model_path.read_bytes() + b"changed_before_freeze")
    with pytest.raises(ValueError):
        pipeline.freeze(config)


def test_report_generation_smoke_and_aggregate_reconciliation(software_run):
    from fraud.reporting import generate_report

    config = software_run["config"]
    pipeline.freeze(config)
    pipeline.evaluate(config)
    result = generate_report(config)
    run = Path(config["run_dir"])
    assert result["status"] == "generated_from_saved_evaluation"
    assert (run / "report.md").is_file()
    assert (run / "gnn_decision.md").is_file()
    assert (run / "resume_bullets.md").is_file()
    schema = read_json(run / "tableau" / "schema.json")
    assert schema["publication_status"].startswith("local_only")
    risk = pd.read_csv(run / "tableau" / "risk_segments.csv")
    assert risk.groupby("dimension")["n"].sum().eq(500).all()
    assert set(risk.period) == {"train", "tune", "calibration", "policy", "test"}
    predictions = pd.read_parquet(run / "test_predictions.parquet")
    stability = pd.read_csv(run / "tableau" / "stability.csv")
    n_by_dimension_model = stability.groupby(["model", "dimension"])["n"].sum()
    assert n_by_dimension_model.eq(len(predictions)).all()
    assert (run / "figures" / "calibration.png").is_file()
    with pytest.raises(ValueError, match="config|source|Frozen"):
        generate_report({**config, "primary_review_rate": 0.2})
