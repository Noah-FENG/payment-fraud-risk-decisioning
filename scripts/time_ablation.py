"""Time-column ablation: the frozen XGBoost recipe without TransactionDT.

The frozen main model uses the relative time offset `TransactionDT` as a numeric feature. Test-block
values lie outside the training range, so the model might partly learn "which period had more fraud".
This script refits the same XGBoost configuration (hyperparameters read from the frozen development
record, same chronological splits) with `TransactionDT` and its missing-value flag removed, and
evaluates it on the tune and policy blocks only.

Safety rules:
- Only development rows are loaded (`fraud.pipeline._load_development` filters out the final test
  block), and the code refuses to run if a "test" split label appears.
- Nothing under `artifacts/main/` is written; results go to `artifacts/time_ablation/` and
  `reports/generated/time_ablation.md`. The frozen model choice is not changed.

Usage: uv run python scripts/time_ablation.py
The launcher retries after the native crash seen on the author's Windows machine and runs with
single-threaded BLAS.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "time_ablation"
REPORT = ROOT / "reports" / "generated" / "time_ablation.md"
TIME_COLUMNS = ("TransactionDT", "TransactionDT__missing")
MAX_ATTEMPTS = 6


def strip_time(features: pd.DataFrame) -> pd.DataFrame:
    """Drop the relative-time feature and its missing-value flag."""
    return features.drop(columns=[c for c in TIME_COLUMNS if c in features.columns])


def evaluate_variant(frame: pd.DataFrame, params: dict, config: dict) -> dict:
    """Fit on train (tune AP), refit on train+tune, calibrate, score policy. Test rows are refused."""
    from fraud.models import ModelBundle, SigmoidCalibrator, make_pipeline
    from fraud.metrics import score_metrics
    from fraud.pipeline import _features
    from fraud.policy import select_top_rate

    splits = set(frame["split"].unique())
    if "test" in splits or not splits <= {"train", "tune", "calibration", "policy"}:
        raise ValueError(f"Only development blocks are allowed, got {sorted(splits)}")
    x = strip_time(_features(frame, True, False))
    assert not set(TIME_COLUMNS) & set(x.columns)
    y = frame["isFraud"]
    masks = {name: (frame["split"] == name).to_numpy() for name in ("train", "tune", "calibration", "policy")}
    estimator = make_pipeline(x, "xgboost", config, params)
    estimator.fit(x.loc[masks["train"]], y[masks["train"]])
    tune_ap = float(average_precision_score(y[masks["tune"]], estimator.predict_proba(x.loc[masks["tune"]])[:, 1]))
    refit = masks["train"] | masks["tune"]
    estimator = make_pipeline(x, "xgboost", config, params)
    estimator.fit(x.loc[refit], y[refit])
    bundle = ModelBundle("xgboost_no_time", estimator, list(x), True, False)
    raw = bundle.predict(x.loc[masks["calibration"]], calibrated=False)
    bundle.calibrator = SigmoidCalibrator().fit(raw, y[masks["calibration"]])
    policy = frame.loc[masks["policy"]]
    scores = bundle.predict(x.loc[masks["policy"]])
    selected = select_top_rate(policy, scores, config["primary_review_rate"], config["batch_seconds"])
    metrics = score_metrics(policy.isFraud, scores, policy.TransactionAmt, selected)
    return {"tune_ap": tune_ap, "policy": metrics, "feature_count": int(x.shape[1]),
            "time_columns_present": sorted(set(TIME_COLUMNS) & set(x.columns))}


def run() -> None:
    from fraud.io import read_json
    from fraud.pipeline import _load_development, split_index

    config = read_json(ROOT / "config" / "default.json")
    os.chdir(ROOT)
    run_dir = ROOT / config["run_dir"]
    development = read_json(run_dir / "development.json")
    params = development["models"]["xgboost"]["params"]
    frame = _load_development(config, split_index(config))
    start = time.perf_counter()
    result = evaluate_variant(frame, params, config)
    result["seconds"] = time.perf_counter() - start
    main_row = pd.read_csv(run_dir / "development_metrics.csv").query(
        "model == 'xgboost' and calibration == 'sigmoid'").iloc[0]
    comparison = {
        "xgboost_main": {"tune_ap": development["models"]["xgboost"]["tune_ap"],
                         **{k: float(main_row[k]) for k in ("average_precision", "roc_auc", "precision",
                                                              "recall", "normal_flag_rate")}},
        "xgboost_no_time": {"tune_ap": result["tune_ap"],
                            **{k: result["policy"][k] for k in ("average_precision", "roc_auc", "precision",
                                                                  "recall", "normal_flag_rate")}},
        "params": params, "feature_count_no_time": result["feature_count"],
        "scope": "tune and policy blocks only; final test block not read",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    a, b = comparison["xgboost_main"], comparison["xgboost_no_time"]
    rows = [("Tune AP", "tune_ap", ".4f"), ("Policy AP", "average_precision", ".4f"),
            ("Policy ROC-AUC", "roc_auc", ".4f"), ("Policy precision at top 1%/day", "precision", ".3f"),
            ("Policy recall at top 1%/day", "recall", ".3f"),
            ("Policy normal-transaction flag rate", "normal_flag_rate", ".4f")]
    lines = ["# Time-column ablation (tune and policy blocks only)", "",
             "Same XGBoost configuration and chronological splits as the frozen main model; the only change is "
             "that `TransactionDT` and its missing-value flag are removed. The final test block is not read.", "",
             "| Metric | Main model (with TransactionDT) | Without TransactionDT |", "| --- | ---: | ---: |"]
    lines += [f"| {label} | {a[key]:{fmt}} | {b[key]:{fmt}} |" for label, key, fmt in rows]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def supervise() -> None:
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "NUMEXPR_NUM_THREADS": "1", "FRAUD_TIME_ABLATION_CHILD": "1"}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"[time_ablation attempt {attempt}]", flush=True)
        code = subprocess.call([sys.executable, str(Path(__file__).resolve())], cwd=ROOT, env=env)
        if code == 0 and (OUT_DIR / "results.json").exists():
            return
        time.sleep(10)
    sys.exit("time_ablation: all attempts failed")


if __name__ == "__main__":
    if os.environ.get("FRAUD_TIME_ABLATION_CHILD"):
        run()
    else:
        supervise()
