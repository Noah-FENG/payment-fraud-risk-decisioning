"""SQL files drive the analysis tables, and the day-block bootstrap is reproducible and consistent."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud import reporting
from fraud.io import read_sql

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("test_bootstrap_script", ROOT / "scripts" / "test_bootstrap.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def analysis_frame(n=600):
    i = np.arange(n)
    frame = pd.DataFrame({
        "TransactionDT": i * 600,
        "TransactionAmt": 5.0 + (i * 13) % 300,
        "isFraud": (i % 9 == 0).astype(int),
        "ProductCD": np.where(i % 3 == 0, "W", "C"),
        "DeviceType": np.where(i % 4 == 0, None, "desktop"),
        "identity_missing": (i % 2).astype("uint8"),
        "DeviceInfo_group": np.where(i % 5 == 0, "Missing", "Windows"),
        "period": np.where(i < 400, "train", "test"),
    })
    frame["relative_week"] = np.floor(frame.TransactionDT / 604800).astype("int64")
    frame["amount_bucket"] = pd.cut(frame.TransactionAmt, reporting.AMOUNT_EDGES, right=False,
                                    labels=reporting.AMOUNT_LABELS).astype("string")
    return frame


def test_read_sql_fills_placeholders_and_strips_terminator():
    text = read_sql("audit_segments.sql", column="ProductCD", limit_clause="LIMIT 3")
    assert "ProductCD" in text and text.rstrip().endswith("LIMIT 3") and not text.rstrip().endswith(";")


def test_sql_risk_tables_match_pandas_and_shares_sum_to_one():
    frame = analysis_frame()
    risk, time_profile, amount = reporting._risk_tables(frame)
    product = risk[(risk.dimension == "ProductCD") & (risk.period == "train")].set_index("segment")
    expected = frame[frame.period == "train"].groupby("ProductCD").isFraud.agg(["size", "sum"])
    assert (product.n == expected["size"]).all() and (product.n_fraud == expected["sum"]).all()
    shares = risk.groupby(["period", "dimension"])[["fraud_count_share", "fraud_amount_share"]].sum()
    assert np.allclose(shares.to_numpy(), 1.0)
    assert (risk[risk.dimension == "DeviceType"].segment == "Missing").any()
    assert time_profile.n.sum() == len(frame) and amount.n.sum() == len(frame)


def bootstrap_frame(n_days=12, per_day=120):
    rng = np.random.default_rng(0)
    n = n_days * per_day
    y = (rng.random(n) < 0.08).astype(int)
    frame = pd.DataFrame({"TransactionID": np.arange(n), "TransactionDT": np.repeat(np.arange(n_days), per_day) * 86400 + np.tile(np.arange(per_day), n_days),
                          "isFraud": y, "TransactionAmt": 10.0 + rng.random(n) * 100})
    noise = rng.normal(size=(3, n))
    frame["score__xgboost"] = 0.6 * y + 0.3 * noise[0]
    frame["score__logistic"] = 0.3 * y + 0.3 * noise[1]
    frame["score__amount_rule"] = frame["TransactionAmt"]
    return frame


def test_bootstrap_intervals_contain_point_and_are_reproducible():
    frame = bootstrap_frame()
    first = bootstrap.bootstrap_intervals(frame, n_boot=200, seed=42, rate=0.05)
    second = bootstrap.bootstrap_intervals(frame, n_boot=200, seed=42, rate=0.05)
    assert first == second
    assert first["n_days"] == 12
    for metrics in first["models"].values():
        for item in metrics.values():
            low, high = item["interval"]
            assert low <= item["point"] <= high
    diff = first["ap_difference_xgboost_minus_logistic"]
    assert diff["interval"][0] <= diff["point"] <= diff["interval"][1]


def test_bootstrap_other_seed_changes_draws():
    frame = bootstrap_frame()
    a = bootstrap.bootstrap_intervals(frame, n_boot=100, seed=1, rate=0.05)
    b = bootstrap.bootstrap_intervals(frame, n_boot=100, seed=2, rate=0.05)
    assert a["models"]["xgboost"]["ap"]["interval"] != b["models"]["xgboost"]["ap"]["interval"]


@pytest.mark.parametrize("name", ["risk_segments.sql", "time_profile.sql", "amount_distribution.sql"])
def test_analysis_sql_files_document_their_grain(name):
    assert "Grain:" in read_sql(name).splitlines()[0]
