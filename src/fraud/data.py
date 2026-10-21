"""Validate and join the two official labelled training tables using DuckDB.

The file names are a user provenance assertion, not cryptographic proof of origin.
No official unlabelled competition test file is read by this module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd

from fraud.io import read_sql


def _literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_table(connection: duckdb.DuckDBPyConnection, name: str) -> dict:
    count, keys, missing_keys = connection.execute(
        f"SELECT count(*), count(DISTINCT TransactionID), "
        f"count(*) FILTER (WHERE TransactionID IS NULL "
        f"OR trim(CAST(TransactionID AS VARCHAR))='') FROM {name}"
    ).fetchone()
    if count == 0 or keys != count or missing_keys:
        raise ValueError(f"{name}: TransactionID must be nonempty, unique and non-null")
    columns = connection.execute(f"DESCRIBE {name}").fetchall()
    expressions = [f"count(*) FILTER (WHERE {_identifier(c[0])} IS NULL)" for c in columns]
    missing = connection.execute(f"SELECT {', '.join(expressions)} FROM {name}").fetchone()
    return {
        "rows": count,
        "unique_transaction_ids": keys,
        "columns": [
            {"name": c[0], "type": c[1], "missing_count": n, "missing_fraction": n / count}
            for c, n in zip(columns, missing, strict=True)
        ],
    }


def ingest(raw_dir: Path, processed_dir: Path) -> dict:
    """Validate labelled CSVs, left join identity, and persist an auditable Parquet table.

    Amounts remain in original unspecified units. Negative/nonfinite amounts are rejected,
    rather than silently clipped. Extreme positive values remain and are audited.
    """
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    paths = {name: raw_dir / f"train_{name}.csv" for name in ("transaction", "identity")}
    absent = [str(path) for path in paths.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError(
            "Official labelled training files are missing: " + ", ".join(absent)
            + ". Sign in to Kaggle, review/accept the competition rules, then download "
            "the two training CSVs. Do not provide credentials in this repository."
        )
    processed_dir.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as connection:
        connection.execute("SET memory_limit='2GB'")
        connection.execute("SET threads=4")
        for name, path in paths.items():
            connection.execute(
                f"CREATE VIEW {name} AS SELECT * FROM read_csv_auto("
                f"{_literal(path.resolve())}, header=true, sample_size=200000, "
                "nullstr=['', 'nan', 'NaN'], ignore_errors=false)"
            )
        transaction_names = [c[0] for c in connection.execute("DESCRIBE transaction").fetchall()]
        identity_names = [c[0] for c in connection.execute("DESCRIBE identity").fetchall()]
        required = {"TransactionID", "TransactionDT", "TransactionAmt", "isFraud"}
        if not required.issubset(transaction_names) or "TransactionID" not in identity_names:
            raise ValueError("Required labelled transaction/identity columns are missing")
        overlap = set(transaction_names).intersection(identity_names) - {"TransactionID"}
        if overlap:
            raise ValueError(f"Ambiguous overlapping source columns: {sorted(overlap)}")
        tables = {name: _validate_table(connection, name) for name in paths}
        bad_labels, bad_time, bad_amount = connection.execute(read_sql("invalid_values.sql")).fetchone()
        if bad_labels or bad_time or bad_amount:
            raise ValueError(
                f"Invalid labels={bad_labels}, relative times={bad_time}, amounts={bad_amount}"
            )
        orphan_identity = connection.execute(
            "SELECT count(*) FROM identity i ANTI JOIN transaction t USING (TransactionID)"
        ).fetchone()[0]
        if orphan_identity:
            raise ValueError(f"Identity contains {orphan_identity} IDs absent from transactions")
        connection.execute("CREATE VIEW joined AS " + read_sql("join_training.sql"))
        joined_count, joined_keys, identity_missing = connection.execute(
            "SELECT count(*), count(DISTINCT TransactionID), sum(identity_missing) FROM joined"
        ).fetchone()
        if joined_count != tables["transaction"]["rows"] or joined_keys != joined_count:
            raise AssertionError("Left join changed the transaction grain")
        tmin, tmax, amin, amedian, ap99, amax, zero_amounts = connection.execute(
            read_sql("audit_summary.sql")
        ).fetchone()
        target = processed_dir / "transactions.parquet"
        temporary = processed_dir / "transactions.partial.parquet"
        connection.execute(
            f"COPY (SELECT * FROM joined ORDER BY TransactionDT, TransactionID) "
            f"TO {_literal(temporary.resolve())} (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        temporary.replace(target)
    audit = {
        "status": "validated_user_supplied_official_named_files",
        "source": "https://www.kaggle.com/competitions/ieee-fraud-detection/data",
        "source_provenance_note": "User must obtain files under applicable Kaggle rules; names do not prove origin.",
        "tables": tables,
        "joined_rows": joined_count,
        "identity_missing_count": identity_missing,
        "identity_missing_fraction": identity_missing / joined_count,
        "orphan_identity_rows": orphan_identity,
        "time": {"minimum_offset_seconds": tmin, "maximum_offset_seconds": tmax},
        "amount": {
            "unit": "original unspecified amount unit", "minimum": amin, "median": amedian,
            "p99": ap99, "maximum": amax, "zero_count": zero_amounts,
            "treatment": "Retain nonnegative finite amounts; log1p for modelling; no loss interpretation.",
        },
        "source_files": {
            name: {"name": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
    }
    (processed_dir / "data_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the validated transaction-grain Parquet, retaining stable column order."""
    frame = pd.read_parquet(path)
    for name in frame.select_dtypes(include=["float64"]).columns:
        if name not in {"TransactionAmt", "TransactionDT"}:
            frame[name] = pd.to_numeric(frame[name], downcast="float")
    return frame
