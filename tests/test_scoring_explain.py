"""Temporary toy-bundle checks for CLI scoring and raw-margin explanations.

The hand-built bundle below is a software fixture, not a model experiment.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import joblib

from fraud.explain import explain
from fraud.features import build_features
from fraud.io import code_fingerprint, digest, dump_json, read_json
from fraud.models import ModelBundle, canonical_frame, make_pipeline
from fraud.scoring import score_file


@pytest.fixture
def tiny_frozen_bundle(tmp_path):
    root = tmp_path / "toy_bundle_software_test"
    run = root / "artifacts" / "toy"
    processed = root / "data" / "processed"
    (run / "models").mkdir(parents=True)
    processed.mkdir(parents=True)
    n = 80
    frame = pd.DataFrame({
        "TransactionID": np.arange(n) + 500,
        "TransactionDT": np.arange(n) * 10,
        "TransactionAmt": 5 + (np.arange(n) * 13) % 150,
        "isFraud": ((np.arange(n) * 13) % 150 > 100).astype(int),
        "card1": np.arange(n) % 5,
        "card2": np.ones(n),
        "card3": np.full(n, 150),
        "card5": np.full(n, 100),
        "addr1": np.arange(n) % 3,
        "ProductCD": np.where(np.arange(n) % 2 == 0, "W", "C"),
        "DeviceInfo": "toy_device",
        "identity_missing": np.zeros(n, dtype=int),
    })
    features = canonical_frame(build_features(frame, include_relations=True))
    name = "xgboost_relations"
    model = make_pipeline(features, "xgboost", {
        "seed": 42, "threads": 1, "logistic_max_iter": 100,
    }, {"n_estimators": 3, "max_depth": 2, "min_child_weight": 1})
    model.fit(features.iloc[:60], frame.isFraud.iloc[:60])
    bundle = ModelBundle(name, model, list(features), True, True)
    model_path = run / "models" / f"{name}.joblib"
    joblib.dump(bundle, model_path)
    frame.to_parquet(processed / "transactions.parquet", index=False)
    index = frame[["TransactionID", "TransactionDT"]].copy()
    index["split"] = np.where(np.arange(n) >= 70, "test", "train")
    index.to_parquet(run / "split_index.parquet", index=False)
    config = {
        "seed": 42, "threads": 1, "run_dir": str(run), "processed_dir": str(processed),
        "project_root": str(root), "shap_sample": 5,
    }
    fingerprint = code_fingerprint()
    manifest = {"chosen_model": name, "config": config, "code": fingerprint,
                "data_sha256": digest(processed / "transactions.parquet")}
    dump_json(run / "development.json", manifest)
    tracked = [model_path, run / "development.json", run / "split_index.parquet"]
    dump_json(run / "frozen_plan.json", {
        "chosen_model": name, "config": config, "code": fingerprint,
        "data_sha256": manifest["data_sha256"],
        "files": {str(path.relative_to(run)): digest(path) for path in tracked},
    })
    dump_json(run / "evaluation.json", {
        "status": "evaluated", "chosen_model": name, "source_outputs": {},
    })
    return config, frame, features, bundle


def test_scoring_matches_bundle_and_excludes_future_history(tiny_frozen_bundle):
    config, frame, _, bundle = tiny_frozen_bundle
    root = Path(config["project_root"])
    history = frame.iloc[:70].drop(columns="isFraud")
    rows = frame.iloc[70:].copy()
    # Scoring input labels and extraneous previous score columns must not predict.
    rows["isFraud"] = 1 - rows.isFraud
    rows["score__old_model"] = 999
    history_path, input_path = root / "history.parquet", root / "input.parquet"
    history.to_parquet(history_path, index=False)
    rows.to_parquet(input_path, index=False)
    out = root / "scored.parquet"
    result = score_file(config, input_path, out, history_path)
    assert result["rows"] == 10
    assert result["history_rows"] == 70
    saved = pd.read_parquet(out)
    expected_features = build_features(frame, include_relations=True)
    expected = bundle.predict(expected_features.iloc[70:])
    np.testing.assert_allclose(saved.risk_probability, expected, rtol=0, atol=0)
    assert saved.TransactionID.tolist() == rows.TransactionID.tolist()
    with pytest.raises(FileExistsError):
        score_file(config, input_path, out, history_path)

    # The first batch score is unchanged if later scoring rows are absent.
    first = root / "first.parquet"
    rows.iloc[:1].to_parquet(first, index=False)
    score_file(config, first, root / "first_scored.parquet", history_path)
    assert pd.read_parquet(root / "first_scored.parquet").risk_probability.iloc[0] == expected[0]

    invalid_history = pd.concat([history, rows.iloc[:1]], ignore_index=True)
    invalid_history.to_parquet(root / "not_prior.parquet", index=False)
    with pytest.raises(ValueError, match="strictly before"):
        score_file(config, input_path, root / "bad.parquet", root / "not_prior.parquet")
    with pytest.raises(ValueError, match="history"):
        score_file(config, input_path, root / "missing_history.parquet")


def test_shap_local_contributions_reconstruct_raw_xgboost_margin(tiny_frozen_bundle):
    config, frame, features, bundle = tiny_frozen_bundle
    run = Path(config["run_dir"])
    status = explain(config)
    assert status["sample_rows"] == 5
    assert "before sigmoid" in status["interpretation"]
    local = pd.read_parquet(run / "explanations" / "local_shap.parquet")
    positions = frame.index[frame.TransactionID.isin(local.TransactionID)]
    x = bundle.pipeline.named_steps["transform"].transform(features.loc[positions])
    raw_margin = bundle.pipeline.named_steps["model"].predict(x, output_margin=True)
    reconstructed = local.drop(columns=["TransactionID", "base_log_odds"]).sum(axis=1) + local.base_log_odds
    np.testing.assert_allclose(reconstructed, raw_margin, rtol=1e-5, atol=1e-6)
    assert (run / "explanations" / "shap_importance.csv").is_file()
    assert (run / "explanations" / "shap_importance.png").is_file()


@pytest.mark.parametrize("field,value", [
    ("TransactionDT", np.nan), ("TransactionDT", -1), ("TransactionID", np.nan),
])
def test_invalid_history_is_rejected(tiny_frozen_bundle, field, value):
    config, frame, _, _ = tiny_frozen_bundle
    root = Path(config["project_root"])
    history = frame.iloc[:70].copy()
    history[field] = history[field].astype(float)
    history.loc[0, field] = value
    history.to_parquet(root / "invalid_history.parquet", index=False)
    frame.iloc[70:].to_parquet(root / "score_input.parquet", index=False)
    with pytest.raises(ValueError):
        score_file(config, root / "score_input.parquet", root / "bad_score.parquet",
                   root / "invalid_history.parquet")


def test_scoring_rejects_incompatible_feature_source_and_explanation_rejects_model_change(
    tiny_frozen_bundle,
):
    config, frame, _, _ = tiny_frozen_bundle
    root, run = Path(config["project_root"]), Path(config["run_dir"])
    frame.iloc[:70].to_parquet(root / "history.parquet", index=False)
    frame.iloc[70:].to_parquet(root / "score_input.parquet", index=False)
    plan_path = run / "frozen_plan.json"
    plan = read_json(plan_path)
    original_code = plan["code"]
    plan["code"] = {"deliberately_incompatible_feature_source": "changed"}
    dump_json(plan_path, plan)
    with pytest.raises(ValueError, match="source|code|fingerprint"):
        score_file(config, root / "score_input.parquet", root / "bad_score.parquet",
                   root / "history.parquet")
    plan["code"] = original_code
    dump_json(plan_path, plan)
    model_path = run / "models" / "xgboost_relations.joblib"
    model_path.write_bytes(model_path.read_bytes() + b"modified_after_freeze")
    with pytest.raises(ValueError, match="artifact|model|Model"):
        explain(config)
