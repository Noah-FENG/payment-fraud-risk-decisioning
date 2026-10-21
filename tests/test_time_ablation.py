"""The time-ablation variant must not contain the time column and must never see the test block."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud.io import read_json

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "time_ablation.py"
spec = importlib.util.spec_from_file_location("time_ablation", SCRIPT)
time_ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(time_ablation)


def synthetic_frame(splits=("train", "tune", "calibration", "policy")):
    n = 800
    frame = pd.DataFrame({
        "TransactionID": np.arange(n),
        "TransactionDT": np.arange(n) * 3600,
        "TransactionAmt": 5 + (np.arange(n) * 17) % 250,
        "isFraud": (np.arange(n) % 6 == 0).astype(int),
        "ProductCD": np.where(np.arange(n) % 3 == 0, "W", "C"),
        "card1": np.arange(n) % 20,
        "C1": np.arange(n) % 11,
    })
    frame["split"] = np.repeat(np.asarray(splits), -(-n // len(splits)))[:n]
    return frame


def config():
    cfg = read_json(Path(__file__).resolve().parents[1] / "config" / "default.json")
    cfg.update({"threads": 1, "primary_review_rate": 0.2})
    return cfg


def test_strip_time_removes_time_feature_and_flag():
    from fraud.pipeline import _features

    features = _features(synthetic_frame(), True, False)
    assert "TransactionDT" in features.columns
    stripped = time_ablation.strip_time(features)
    assert not set(time_ablation.TIME_COLUMNS) & set(stripped.columns)
    assert len(stripped.columns) < len(features.columns)


def test_variant_trains_without_time_and_reports_policy_metrics():
    params = {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1}
    result = time_ablation.evaluate_variant(synthetic_frame(), params, config())
    assert result["time_columns_present"] == []
    assert 0 <= result["tune_ap"] <= 1
    assert result["policy"]["selected_n"] > 0


def test_variant_refuses_the_test_block():
    with pytest.raises(ValueError, match="development blocks"):
        time_ablation.evaluate_variant(synthetic_frame(("train", "tune", "policy", "test")), {}, config())
