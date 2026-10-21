"""Predictor construction with strict pre-transaction history and no labels.

Build history on an ordered prefix containing the scoring rows and their context.
Rows at identical TransactionDT are simultaneous: none can observe its peers.
No fitted transformation is performed here; model encoders fit on training only.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd


RELATION_CANDIDATES = {
    "card_address": ("card1", "card2", "card3", "card5", "addr1"),
    "device_address": ("DeviceInfo", "addr1"),
}

CATEGORICAL_COLUMNS = {
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo",
    *(f"M{i}" for i in range(1, 10)),
    *(f"id_{i}" for i in range(12, 39)),
}

# These fields contain numeric attribute codes, although they are categorical.
# card4/card6 and DeviceInfo are text and must not be numerically normalised.
NUMERIC_CODE_CATEGORICAL = {"card1", "card2", "card3", "card5", "addr1", "addr2"}


def _category_strings(column: pd.Series, name: str) -> pd.Series:
    """Use stable numeric-code strings across CSV int/float inference differences.

    For example, integer 500, float 500.0 and numeric string '500.000' represent
    the same numeric attribute code. Decimal formatting avoids changing fractional
    values through an intermediate binary-float conversion. Missing stays pd.NA.
    Text attributes keep their exact string values. In anonymous id categories,
    numeric-looking values are normalised individually; text values stay intact.
    The result of one row cannot depend on values observed in a future batch.
    """
    strings = column.astype("string")
    anonymous_id = bool(re.fullmatch(r"id_\d+", name))
    if name not in NUMERIC_CODE_CATEGORICAL and not anonymous_id:
        return strings
    mapping = {}
    for value in strings.dropna().unique():
        try:
            number = Decimal(value)
        except InvalidOperation:
            mapping[value] = value
            continue
        if not number.is_finite():
            mapping[value] = pd.NA
        elif number == 0:
            mapping[value] = "0"
        else:
            text = format(number, "f")
            mapping[value] = text.rstrip("0").rstrip(".") if "." in text else text
    return strings.map(mapping).astype("string")


def is_anonymous(name: str) -> bool:
    """Groups with insufficient public provenance for point-in-time verification."""
    return bool(re.fullmatch(r"[CDVM]\d+", name) or re.fullmatch(r"id_\d+", name))


def assign_splits(frame: pd.DataFrame, fractions: dict[str, float]) -> pd.Series:
    """Split by cumulative row fraction at complete time groups, never by labels.

    For each requested cumulative fraction take the time at ceil(fraction*n)-1;
    all rows equal to that time belong to the earlier split. Large time groups can
    change realised fractions or make a split empty; the caller must reject empties.
    """
    expected = ["train", "tune", "calibration", "policy", "test"]
    if list(fractions) != expected or any(value <= 0 for value in fractions.values()):
        raise ValueError(f"Positive chronological split fractions required in order {expected}")
    if not np.isclose(sum(fractions.values()), 1):
        raise ValueError("Split fractions must sum to one")
    times = pd.to_numeric(frame["TransactionDT"], errors="raise").to_numpy(dtype=float)
    if len(times) == 0 or not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("Nonempty finite nonnegative relative times required")
    ordered = np.sort(times, kind="stable")
    positions = np.ceil(np.cumsum(list(fractions.values()))[:-1] * len(times)).astype(int) - 1
    boundaries = ordered[np.clip(positions, 0, len(times) - 1)]
    assignment = np.searchsorted(boundaries, times, side="left")
    return pd.Series(np.asarray(expected)[assignment], index=frame.index, name="split")


def proxy_codes(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    """Exact attribute-tuple codes, with -1 for ANY missing component.

    Codes are only temporary grouping keys. They are not customer/card/device IDs
    and never enter the model. Blank strings also mean missing. No hash collision
    approximation or common missing-value entity is used.
    """
    result = pd.Series(-1, index=frame.index, dtype="int64")
    if not set(columns).issubset(frame.columns):
        return result
    values = frame.loc[:, list(columns)].copy()
    for name in columns:
        if name in NUMERIC_CODE_CATEGORICAL:
            values[name] = _category_strings(values[name], name)
    complete = values.notna().all(axis=1)
    for name in columns:
        if pd.api.types.is_object_dtype(values[name]) or isinstance(values[name].dtype, pd.StringDtype):
            complete &= values[name].astype("string").str.strip().ne("").fillna(False)
    if complete.any():
        tuples = pd.MultiIndex.from_frame(values.loc[complete])
        codes, _ = pd.factorize(tuples, sort=False)
        result.loc[complete] = codes
    return result


def historical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Label-free cumulative counts, mean amounts and recency at DT strictly < t."""
    out = pd.DataFrame(index=frame.index)
    for proxy_name, columns in RELATION_CANDIDATES.items():
        keys = proxy_codes(frame, columns).to_numpy()
        present = keys >= 0
        base = f"hist_{proxy_name}_"
        out[base + "missing"] = (~present).astype("uint8")
        out[base + "count"] = np.zeros(len(frame), dtype="float32")
        out[base + "mean_amount"] = np.full(len(frame), np.nan, dtype="float32")
        out[base + "seconds_since_previous"] = np.full(len(frame), np.nan, dtype="float32")
        if not present.any():
            continue
        rows = pd.DataFrame({
            "key": keys[present],
            "time": frame["TransactionDT"].to_numpy()[present],
            "amount": frame["TransactionAmt"].to_numpy(dtype=float)[present],
            "position": np.flatnonzero(present),
        })
        batches = rows.groupby(["key", "time"], sort=True, observed=True).agg(
            batch_count=("amount", "size"), batch_amount=("amount", "sum")
        ).reset_index()
        grouped = batches.groupby("key", sort=False, observed=True)
        batches["count"] = grouped["batch_count"].cumsum().groupby(batches["key"]).shift(1).fillna(0)
        batches["amount_before"] = grouped["batch_amount"].cumsum().groupby(batches["key"]).shift(1).fillna(0)
        batches["mean_amount"] = batches["amount_before"].div(batches["count"].replace(0, np.nan))
        batches["seconds_since_previous"] = batches["time"] - grouped["time"].shift(1)
        mapped = rows.merge(
            batches[["key", "time", "count", "mean_amount", "seconds_since_previous"]],
            on=["key", "time"], how="left", validate="many_to_one", sort=False,
        )
        for suffix in ("count", "mean_amount", "seconds_since_previous"):
            out.iloc[mapped["position"].to_numpy(), out.columns.get_loc(base + suffix)] = (
                mapped[suffix].to_numpy(dtype="float32")
            )
    return out


def build_features(
    frame: pd.DataFrame, include_anonymous: bool = True, include_relations: bool = False
) -> pd.DataFrame:
    """Make features without fitting any information from future or labels.

    Missing numeric values remain NaN and get explicit flags. Category strings have
    no imputation token yet; model preprocessing owns missing/unknown handling.
    Raw amount is retained with a log1p companion. Missing identity is preserved.
    TransactionDT remains an offset; no real date, holiday, timezone or night label.
    """
    if not frame.index.is_unique:
        raise ValueError("Unique DataFrame index required for feature alignment")
    forbidden = {"TransactionID", "isFraud", "split", "identity_missing"}
    names = [name for name in frame.columns if name not in forbidden]
    if not include_anonymous:
        names = [name for name in names if not is_anonymous(name)]
    values: dict[str, pd.Series | np.ndarray | int] = {}
    for name in names:
        column = frame[name]
        if name in CATEGORICAL_COLUMNS or not pd.api.types.is_numeric_dtype(column):
            values[name] = _category_strings(column, name)
        else:
            numeric = pd.to_numeric(column, errors="coerce").replace([np.inf, -np.inf], np.nan)
            values[name] = numeric.astype("float32")
            # Always create the flag: a column complete in train can be missing later.
            values[name + "__missing"] = numeric.isna().astype("uint8")
    amounts = pd.to_numeric(frame["TransactionAmt"], errors="raise")
    if amounts.isna().any() or not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("Amounts must be finite and nonnegative before feature construction")
    values["log_amount"] = np.log1p(amounts).astype("float32")
    if "identity_missing" in frame:
        values["identity_missing"] = frame["identity_missing"].astype("uint8")
    else:
        identity_columns = [c for c in frame if c.startswith("id_") or c in {"DeviceType", "DeviceInfo"}]
        values["identity_missing"] = (
            frame[identity_columns].isna().all(axis=1).astype("uint8") if identity_columns else 1
        )
    out = pd.DataFrame(values, index=frame.index)
    if include_relations:
        out = pd.concat([out, historical_features(frame)], axis=1)
    return out.copy()  # consolidate blocks for downstream sklearn conversion
