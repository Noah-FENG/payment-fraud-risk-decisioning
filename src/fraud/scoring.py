"""Batch scoring from a frozen bundle, with explicit history context."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraud.features import build_features
from fraud.io import code_fingerprint, digest, dump_json, read_json
from fraud.metrics import score_metrics
from fraud.policy import scenario_costs, select_top_rate


def read_frame(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def score_file(config, input_path, output_path, history_path=None):
    run = Path(config["run_dir"])
    plan = read_json(run / "frozen_plan.json")
    if plan["code"] != code_fingerprint():
        raise ValueError("Scoring source/dependency lock differs from the frozen plan")
    name = plan["chosen_model"]
    model_path = run / "models" / f"{name}.joblib"
    if digest(model_path) != plan["files"][str(model_path.relative_to(run))]:
        raise ValueError("Scoring model does not match the frozen plan")
    bundle = joblib.load(model_path)
    rows = read_frame(input_path).reset_index(drop=True)
    for key in ("TransactionID", "TransactionDT", "TransactionAmt"):
        if key not in rows or rows[key].isna().any():
            raise ValueError(f"Scoring input requires complete {key}")
    if rows.TransactionID.duplicated().any() or len(rows) == 0:
        raise ValueError("Scoring rows must have unique TransactionID and be nonempty")
    if not np.isfinite(rows.TransactionDT).all() or (rows.TransactionDT < 0).any():
        raise ValueError("Scoring time must be a finite nonnegative relative offset")
    history_n = 0
    if bundle.include_relations:
        if history_path is None:
            raise ValueError("Relation model needs --history containing only earlier joined rows")
        history = read_frame(history_path).reset_index(drop=True)
        for key in ("TransactionID", "TransactionDT", "TransactionAmt"):
            if key not in history or history[key].isna().any():
                raise ValueError(f"History requires complete {key}")
        for key in ("TransactionDT", "TransactionAmt"):
            if not np.isfinite(history[key]).all() or (history[key] < 0).any():
                raise ValueError(f"History requires finite nonnegative {key}")
        if len(history) and history.TransactionDT.max() >= rows.TransactionDT.min():
            raise ValueError("History must end strictly before the first input transaction")
        if history.TransactionID.duplicated().any() or history.TransactionID.isin(rows.TransactionID).any():
            raise ValueError("History contains duplicate or input TransactionID")
        history_n = len(history)
        context = pd.concat([history, rows], ignore_index=True)
    else:
        context = rows
    # Labels and generated score fields cannot enter features.
    context = context.drop(columns=[c for c in context if c == "isFraud" or c.startswith("score__")], errors="ignore")
    features = build_features(context, bundle.include_anonymous, bundle.include_relations)
    predictions = bundle.predict(features.iloc[history_n:])
    result = rows[["TransactionID", "TransactionDT"]].copy()
    result["risk_probability"] = predictions
    result["model"] = name
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a scoring file")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    return {"rows": len(result), "model": name, "history_rows": history_n, "output": str(output)}


def sensitivity_scenarios(base):
    result = []
    values = {"loss_rate": [0.25, 0.5, 1.0], "review_effectiveness": [0.25, 0.5, 0.9],
              "review_cost": [0.5, 2.0, 5.0], "false_positive_cost": [0.0, 5.0, 20.0]}
    for variable, alternatives in values.items():
        for value in alternatives:
            result.append({**base, variable: value, "name": f"{variable}={value:g}",
                           "varied_parameter": variable, "varied_value": value})
    return result


def simulate(config, review_rate, loss_rate, effectiveness, review_cost, false_positive_cost, output):
    run = Path(config["run_dir"])
    manifest = read_json(run / "evaluation.json")
    frame = pd.read_parquet(run / "test_predictions.parquet")
    name = manifest["chosen_model"]
    scores = frame[f"score__{name}"].to_numpy()
    selected = select_top_rate(frame, scores, review_rate, config["batch_seconds"])
    scenario = {"name": "user_scenario", "loss_rate": loss_rate, "review_effectiveness": effectiveness,
                "review_cost": review_cost, "false_positive_cost": false_positive_cost}
    result = {"kind": "posthoc_scenario_not_model_selection", "model": name,
              "review_rate_target": review_rate, "exposure": manifest["exposure"],
              "metrics": score_metrics(frame.isFraud, scores, frame.TransactionAmt, selected),
              "costs": scenario_costs(frame.isFraud, frame.TransactionAmt, selected, scenario)}
    if Path(output).exists():
        raise FileExistsError("Choose a new output path for the scenario")
    dump_json(Path(output), result)
    return result
