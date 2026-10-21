"""Small provenance helpers. Paths in config resolve from the project, not the shell."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path


SQL_DIR = Path(__file__).resolve().parents[2] / "sql"


def read_sql(name: str, **parameters: str) -> str:
    """Return the text of ``sql/<name>`` with ``{placeholder}`` parameters filled in.

    Only trusted, code-supplied identifiers are substituted (column names, LIMIT clauses);
    data values never go through this function.
    """
    text = (SQL_DIR / name).read_text(encoding="utf-8").strip().rstrip(";")
    return text.format(**parameters) if parameters else text


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def dump_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> dict:
    path = path.resolve()
    config = read_json(path)
    root = path.parent.parent
    for key in ("raw_dir", "processed_dir", "run_dir"):
        value = Path(config[key])
        config[key] = str(value if value.is_absolute() else root / value)
    config["project_root"] = str(root)
    config["config_path"] = str(path)
    if abs(sum(config["split_fractions"].values()) - 1) > 1e-9:
        raise ValueError("split_fractions must sum to one")
    if list(config["split_fractions"]) != ["train", "tune", "calibration", "policy", "test"]:
        raise ValueError("Use the five ordered temporal blocks")
    if not 0 < config["primary_review_rate"] <= 1:
        raise ValueError("primary_review_rate must be in (0, 1]")
    return config


def code_fingerprint() -> dict:
    package = Path(__file__).parent
    root = package.parent.parent
    files = [*package.glob("*.py"), *root.glob("sql/*.sql"), root / "uv.lock", root / "pyproject.toml"]
    return {str(p.relative_to(root)): digest(p) for p in sorted(files) if p.exists()}


def environment() -> dict:
    packages = ["duckdb", "pandas", "numpy", "pyarrow", "scikit-learn", "xgboost", "shap", "joblib", "psutil"]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in packages}}
