"""Day-block bootstrap intervals for the registered test-block metrics (descriptive uncertainty only).

Reads the existing `artifacts/main/test_predictions.parquet` and resamples whole relative days
(day = floor(TransactionDT / 86400)) with replacement. The daily top-1% review selection is made
inside each day by `fraud.policy.select_top_rate` (floor capacity, same rule as the evaluation), so a
resampled set of days fixes the selection set. Percentile 95% intervals are reported for AP, ROC-AUC,
and top-1%-per-day precision and recall for xgboost, logistic and the amount rule, plus the xgboost
minus logistic AP difference.

This is not a new selection step. The test block was used once; nothing here changes the frozen
model, `evaluation.json`, the exposure ledger or any test output. Results are written to
`artifacts/main/test_bootstrap.json` (local).

Usage: uv run python scripts/test_bootstrap.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud.io import read_json
from fraud.policy import select_top_rate

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("xgboost", "logistic", "amount_rule")
N_BOOT = 1000
SEED = 42


def bootstrap_intervals(frame: pd.DataFrame, models=MODELS, n_boot: int = N_BOOT, seed: int = SEED,
                        rate: float = 0.01, batch_seconds: int = 86400, level: float = 0.95) -> dict:
    """Day-block bootstrap of AP, ROC-AUC and top-`rate` precision/recall for each model.

    `frame` needs TransactionDT, isFraud, TransactionAmt and one `score__<model>` column per model.
    """
    day = np.floor(frame["TransactionDT"].to_numpy() / batch_seconds).astype(np.int64)
    days = np.unique(day)
    rows_of_day = {d: np.flatnonzero(day == d) for d in days}
    y = frame["isFraud"].to_numpy().astype(int)
    scores = {m: frame[f"score__{m}"].to_numpy(dtype=float) for m in models}
    selected = {m: np.asarray(select_top_rate(frame, scores[m], rate, batch_seconds), dtype=bool) for m in models}
    day_stats = {m: {d: (int((selected[m][r] & (y[r] == 1)).sum()), int(selected[m][r].sum()), int(y[r].sum()))
                     for d, r in rows_of_day.items()} for m in models}

    def point(m):
        tp = sum(v[0] for v in day_stats[m].values())
        sel = sum(v[1] for v in day_stats[m].values())
        pos = sum(v[2] for v in day_stats[m].values())
        return {"ap": float(average_precision_score(y, scores[m])), "roc_auc": float(roc_auc_score(y, scores[m])),
                "precision": tp / sel if sel else float("nan"), "recall": tp / pos if pos else float("nan")}

    rng = np.random.default_rng(seed)
    draws = {m: {k: [] for k in ("ap", "roc_auc", "precision", "recall")} for m in models}
    ap_difference = []
    for _ in range(n_boot):
        picked = rng.choice(days, size=len(days), replace=True)
        rows = np.concatenate([rows_of_day[d] for d in picked])
        yb = y[rows]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        for m in models:
            sb = scores[m][rows]
            draws[m]["ap"].append(average_precision_score(yb, sb))
            draws[m]["roc_auc"].append(roc_auc_score(yb, sb))
            tp = sum(day_stats[m][d][0] for d in picked)
            sel = sum(day_stats[m][d][1] for d in picked)
            pos = sum(day_stats[m][d][2] for d in picked)
            draws[m]["precision"].append(tp / sel if sel else np.nan)
            draws[m]["recall"].append(tp / pos if pos else np.nan)
        if "xgboost" in models and "logistic" in models:
            ap_difference.append(draws["xgboost"]["ap"][-1] - draws["logistic"]["ap"][-1])
    low, high = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    interval = lambda values: [float(np.nanpercentile(values, low)), float(np.nanpercentile(values, high))]  # noqa: E731
    result = {"method": "day-block bootstrap, resampling whole relative days with replacement",
              "n_days": int(len(days)), "n_rows": int(len(frame)), "n_boot_requested": n_boot,
              "n_boot_used": len(draws[models[0]]["ap"]), "seed": seed, "level": level,
              "review_rate_per_day": rate, "models": {}}
    for m in models:
        p = point(m)
        result["models"][m] = {k: {"point": p[k], "interval": interval(draws[m][k])} for k in p}
    if ap_difference:
        diff = float(average_precision_score(y, scores["xgboost"]) - average_precision_score(y, scores["logistic"]))
        result["ap_difference_xgboost_minus_logistic"] = {"point": diff, "interval": interval(ap_difference)}
    return result


def main() -> None:
    config = read_json(ROOT / "config" / "default.json")
    run_dir = ROOT / config["run_dir"]
    frame = pd.read_parquet(run_dir / "test_predictions.parquet")
    result = bootstrap_intervals(frame, rate=config["primary_review_rate"], batch_seconds=config["batch_seconds"])
    (run_dir / "test_bootstrap.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"days={result['n_days']} rows={result['n_rows']} boot={result['n_boot_used']}")
    for m, metrics in result["models"].items():
        for k, v in metrics.items():
            print(f"{m:12s} {k:10s} {v['point']:.4f}  [{v['interval'][0]:.4f}, {v['interval'][1]:.4f}]")
    d = result["ap_difference_xgboost_minus_logistic"]
    print(f"AP diff xgboost-logistic {d['point']:.4f}  [{d['interval'][0]:.4f}, {d['interval'][1]:.4f}]")


if __name__ == "__main__":
    main()
