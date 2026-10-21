"""Tiny synthetic fixtures test invariants; these are NOT empirical project results."""

import json

import numpy as np
import pandas as pd
import pytest

from fraud.data import ingest, load_dataset
from fraud.features import assign_splits, build_features, historical_features, proxy_codes
from fraud.relations import audit_relations


@pytest.fixture
def transactions():
    return pd.DataFrame({
        "TransactionID": range(100, 110), "TransactionDT": np.arange(10) * 100,
        "TransactionAmt": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "isFraud": [0, 1] * 5, "ProductCD": ["W"] * 10,
        "card1": [1] * 10, "card2": [2] * 10, "card3": [3] * 10,
        "card5": [5] * 10, "addr1": [10] * 10, "V1": np.arange(10.0),
    })


@pytest.fixture
def source_files(tmp_path, transactions):
    raw = tmp_path / "raw"
    raw.mkdir()
    transactions.to_csv(raw / "train_transaction.csv", index=False)
    pd.DataFrame({"TransactionID": [100, 102], "DeviceInfo": ["phone-a", "desktop-b"]}).to_csv(
        raw / "train_identity.csv", index=False
    )
    return raw, tmp_path / "processed"


def test_left_join_preserves_grain_and_missing_identity(source_files):
    raw, processed = source_files
    audit = ingest(raw, processed)
    joined = load_dataset(processed / "transactions.parquet")
    assert len(joined) == audit["joined_rows"] == 10
    assert joined.TransactionID.is_unique
    assert joined.identity_missing.sum() == 8
    assert joined.loc[joined.TransactionID == 101, "DeviceInfo"].isna().all()
    assert audit["tables"]["transaction"]["rows"] == 10
    assert len(audit["source_files"]["transaction"]["sha256"]) == 64
    assert json.loads((processed / "data_audit.json").read_text())["joined_rows"] == 10


@pytest.mark.parametrize("source", ["transaction", "identity"])
def test_duplicate_primary_keys_rejected(source_files, source):
    raw, processed = source_files
    path = raw / f"train_{source}.csv"
    table = pd.read_csv(path)
    pd.concat([table, table.iloc[[0]]], ignore_index=True).to_csv(path, index=False)
    with pytest.raises(ValueError, match="TransactionID"):
        ingest(raw, processed)
    assert not (processed / "transactions.parquet").exists()


@pytest.mark.parametrize("column,value", [
    ("isFraud", np.nan), ("isFraud", 0.2), ("isFraud", 2),
    ("TransactionDT", -1), ("TransactionDT", np.inf),
    ("TransactionAmt", -10), ("TransactionAmt", np.inf),
])
def test_invalid_targets_time_or_amount_rejected(source_files, column, value):
    raw, processed = source_files
    path = raw / "train_transaction.csv"
    table = pd.read_csv(path)
    table[column] = table[column].astype(float)
    table.loc[0, column] = value
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Invalid"):
        ingest(raw, processed)


def test_orphan_identity_rejected(source_files):
    raw, processed = source_files
    pd.DataFrame({"TransactionID": [999], "DeviceInfo": ["unknown"]}).to_csv(
        raw / "train_identity.csv", index=False
    )
    with pytest.raises(ValueError, match="absent"):
        ingest(raw, processed)


def test_unlabelled_test_data_does_not_satisfy_input(tmp_path):
    (tmp_path / "test_transaction.csv").write_text("TransactionID\n1\n")
    with pytest.raises(FileNotFoundError, match="Official labelled"):
        ingest(tmp_path, tmp_path / "processed")


def test_time_splits_are_strict_and_do_not_split_ties(transactions):
    frame = pd.concat([transactions, transactions.iloc[[2]].assign(TransactionID=999)])
    frame = frame.sample(frac=1, random_state=3).reset_index(drop=True)
    fractions = {"train": 0.4, "tune": 0.15, "calibration": 0.15, "policy": 0.15, "test": 0.15}
    splits = assign_splits(frame, fractions)
    assert splits.index.equals(frame.index)
    for _, group in frame.assign(split=splits).groupby("TransactionDT"):
        assert group.split.nunique() == 1
    nonempty = [frame.loc[splits == name, "TransactionDT"] for name in fractions]
    for earlier, later in zip(nonempty, nonempty[1:]):
        assert earlier.max() < later.min()
    shuffled_labels = frame.copy()
    shuffled_labels.isFraud = 1 - shuffled_labels.isFraud
    pd.testing.assert_series_equal(splits, assign_splits(shuffled_labels, fractions))


def test_history_excludes_current_ties_future_and_labels(transactions):
    frame = transactions.iloc[:4].copy()
    frame["TransactionDT"] = [0, 10, 10, 20]
    frame["TransactionAmt"] = [10, 20, 30, 40]
    hist = historical_features(frame)
    assert hist.hist_card_address_count.tolist() == [0, 1, 1, 3]
    assert hist.hist_card_address_mean_amount.iloc[1] == 10
    assert hist.hist_card_address_mean_amount.iloc[2] == 10
    assert hist.hist_card_address_mean_amount.iloc[3] == 20
    assert hist.hist_card_address_seconds_since_previous.iloc[3] == 10
    changed = pd.concat([frame, transactions.iloc[[4]].assign(TransactionDT=1000)])
    changed["isFraud"] = 1 - changed["isFraud"]
    pd.testing.assert_frame_equal(hist, historical_features(changed).iloc[:4])


def test_missing_proxy_components_do_not_form_shared_entity(transactions):
    frame = transactions.iloc[:3].copy()
    frame["card2"] = [np.nan, np.nan, 2]
    assert proxy_codes(frame, ("card1", "card2")).tolist() == [-1, -1, 0]
    hist = historical_features(frame)
    assert hist.hist_card_address_count.tolist() == [0, 0, 0]
    assert hist.hist_card_address_missing.tolist() == [1, 1, 0]


def test_feature_columns_are_label_free_and_group_ablatable(transactions):
    frame = transactions.assign(id_01=1.0, id_12="Found", M1="T", DeviceInfo=pd.NA)
    features = build_features(frame, include_anonymous=False, include_relations=True)
    assert not {"TransactionID", "isFraud", "V1", "id_01", "id_12", "M1"}.intersection(features)
    assert features.ProductCD.dtype.name == "string"
    assert "id_12" in build_features(frame)
    assert "TransactionAmt__missing" in features
    assert "hist_card_address_count" in features
    changed = frame.copy()
    changed.isFraud = 1 - changed.isFraud
    pd.testing.assert_frame_equal(features, build_features(changed, False, True))


def test_relation_audit_has_no_label_dependence_or_semantic_auto_approval(transactions):
    frame = transactions.assign(DeviceInfo="generic-phone", P_emaildomain="common.example")
    report = audit_relations(frame)
    assert not report["gnn_gate"]["approved"]
    assert not report["uses_labels"]
    assert report["combined_candidate_transaction_components"]["largest_component_fraction"] == 1
    assert report["candidates"]["broad_email_domain_control"]["role"].startswith("broadness")
    json.dumps(report)
    changed = frame.copy()
    changed.isFraud = 1 - changed.isFraud
    assert report == audit_relations(changed)


@pytest.mark.parametrize("column", ["card1", "card2", "card3", "card5", "addr1", "addr2", "id_13"])
def test_numeric_categories_are_consistent_across_train_score_csv_dtypes(transactions, column):
    train = transactions.iloc[:2].copy()
    train[column] = [500.0, np.nan]  # CSV inference when the training column contains missing.
    train_features = build_features(train)
    score = transactions.iloc[:1].copy()
    score[column] = 500  # The same CSV column can infer integer when a score batch is complete.
    assert train_features[column].iloc[0] == build_features(score)[column].iloc[0] == "500"
    assert pd.isna(train_features[column].iloc[1])
    score[column] = "500.000"
    assert build_features(score)[column].iloc[0] == "500"
    score[column] = "500.1250"
    assert build_features(score)[column].iloc[0] == "500.125"


def test_text_categories_are_not_numerically_reinterpreted(transactions):
    frame = transactions.iloc[:2].assign(DeviceInfo="500.0", ProductCD="500.0", M1="500.0",
                                         card4="500.0", card6="500.0", id_31=["browser-1", "500.0"])
    features = build_features(frame)
    for column in ("DeviceInfo", "ProductCD", "M1", "card4", "card6"):
        assert features[column].tolist() == ["500.0", "500.0"]
    assert features.id_31.tolist() == ["browser-1", "500"]


def test_proxy_history_uses_same_numeric_code_normalisation(transactions):
    frame = transactions.iloc[:3].copy()
    frame["card2"] = pd.Series([500.0, "500", "500.000"], dtype=object)
    assert historical_features(frame).hist_card_address_count.tolist() == [0, 1, 2]


def test_numeric_id_category_normalisation_does_not_depend_on_future_values(transactions):
    prefix = transactions.iloc[:2].assign(id_13=[500.0, 501.0])
    future = transactions.iloc[[2]].assign(id_13="unseen-text-code")
    before = build_features(prefix)
    after = build_features(pd.concat([prefix, future]))
    pd.testing.assert_frame_equal(before, after.iloc[:2])
