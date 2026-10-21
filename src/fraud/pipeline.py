"""Temporal development, explicit freezing, and recorded final evaluation."""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import average_precision_score

from fraud.features import assign_splits, build_features
from fraud.io import code_fingerprint, digest, dump_json, environment, read_json
from fraud.metrics import calibration_table, score_metrics
from fraud.models import ModelBundle, SigmoidCalibrator, canonical_frame, make_pipeline, timed_fit
from fraud.policy import (evaluate_policies, fit_cost_threshold, fit_threshold,
                          fixed_threshold, scenario_costs, select_top_rate)
from fraud.relations import audit_relations
from fraud.scoring import sensitivity_scenarios

LOG = logging.getLogger(__name__)


class ResourceMonitor:
    """Sample whole-process RSS, not an incremental or GPU memory estimate.

    Set FRAUD_MONITOR=0 to disable the sampler thread: on this Windows host the
    background psutil sampling correlates with sporadic native GIL fatals inside
    sklearn/numpy/scipy fits. Peak RSS then degrades to a two-point estimate.
    """
    def __enter__(self):
        self.peak = psutil.Process().memory_info().rss
        if os.environ.get("FRAUD_MONITOR", "1") == "0":
            self.thread = None
            return self
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(0.05):
                self.peak = max(self.peak, psutil.Process().memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, psutil.Process().memory_info().rss)
        if self.thread is not None:
            self.stop.set()
            self.thread.join()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def prepared(config):
    path = Path(config["processed_dir"]) / "transactions.parquet"
    if not path.exists():
        raise FileNotFoundError("Run fraud prepare using the official labeled training files first")
    return path


def split_index(config):
    frame = pd.read_parquet(prepared(config), columns=["TransactionID", "TransactionDT"])
    frame["split"] = assign_splits(frame, config["split_fractions"])
    frame = frame.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    return frame


def _load_development(config, index):
    cutoff = float(index.loc[index["split"] == "test", "TransactionDT"].min())
    frame = pd.read_parquet(prepared(config), filters=[("TransactionDT", "<", cutoff)])
    frame = frame.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    frame["split"] = frame["TransactionID"].map(index.set_index("TransactionID")["split"])
    for name in ("train", "tune", "calibration", "policy"):
        if frame.loc[frame["split"] == name, "isFraud"].nunique() != 2:
            raise ValueError(f"{name} needs both classes; do not silently change the final test")
    return frame


def _features(frame, anonymous, relations):
    return canonical_frame(build_features(frame.drop(columns=["split"], errors="ignore"),
                                          include_anonymous=anonymous,
                                          include_relations=relations))


def _metrics_rows(frame, score, model, threshold, config):
    selected = select_top_rate(frame, score, config["primary_review_rate"], config["batch_seconds"])
    return {"model": model, "score_kind": "probability", "period": "policy",
            "policy": "top_rate_per_relative_day", "review_rate_target": config["primary_review_rate"],
            **score_metrics(frame.isFraud, score, frame.TransactionAmt, selected)}


def develop(config):
    run = Path(config["run_dir"])
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("Run directory is not empty. Preserve it and choose a new run_dir")
    index = split_index(config)
    frame = _load_development(config, index)
    run.mkdir(parents=True, exist_ok=True)
    (run / "models").mkdir()
    dump_json(run / "run_manifest.json", {"status": "developing", "started_utc": utc_now()})
    index.to_parquet(run / "split_index.parquet", index=False)
    audit = audit_relations(frame.drop(columns=["isFraud", "split"]))
    dump_json(run / "relation_audit.json", audit)
    masks = {name: (frame["split"] == name).to_numpy() for name in ("train", "tune", "calibration", "policy")}
    refit = masks["train"] | masks["tune"]
    policy_frame = frame.loc[masks["policy"]].copy()
    policy_predictions = policy_frame[["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]].copy()
    rows, tuning, calibrated_rows, candidates = [], [], [], {}
    amount = policy_frame.TransactionAmt.to_numpy()
    rule_threshold = fit_threshold(policy_frame.isFraud, amount, amount, config["primary_review_rate"])
    rule_row = _metrics_rows(policy_frame, amount, "amount_rule", rule_threshold, config)
    rule_row.update(score_kind="heuristic_amount", brier=None)
    rows.append(rule_row)
    policy_predictions["amount_rule"] = amount
    candidates["amount_rule"] = {"threshold": rule_threshold, "kind": "rule", "training_seconds": 0.0}
    variants = [("logistic", "logistic", True, False),
                ("xgboost", "xgboost", True, False),
                ("xgboost_no_anonymous", "xgboost", False, False),
                ("xgboost_relations", "xgboost", True, True)]
    for name, kind, anonymous, relations in variants:
        LOG.info("Developing %s", name)
        with ResourceMonitor() as resources:
            x = _features(frame, anonymous, relations)
            if relations and not any(c.startswith("hist_") for c in x):
                LOG.warning("No valid proxy-history columns: skipping relation challenger")
                continue
            params_grid = config["xgboost_candidates"] if kind == "xgboost" else [{}]
            best_ap, best_params, tuning_seconds = -1.0, None, 0.0
            for params in params_grid:
                estimator = make_pipeline(x, kind, config, params)
                elapsed = timed_fit(estimator, x.loc[masks["train"]], frame.loc[masks["train"], "isFraud"])
                p_tune = estimator.predict_proba(x.loc[masks["tune"]])[:, 1]
                ap = float(average_precision_score(frame.loc[masks["tune"], "isFraud"], p_tune))
                tuning.append({"model": name, "params": params, "tune_ap": ap, "fit_seconds": elapsed})
                tuning_seconds += elapsed
                if ap > best_ap:
                    best_ap, best_params = ap, params
            estimator = make_pipeline(x, kind, config, best_params)
            fit_seconds = timed_fit(estimator, x.loc[refit], frame.loc[refit, "isFraud"])
            bundle = ModelBundle(name, estimator, list(x), anonymous, relations)
            cal_raw = bundle.predict(x.loc[masks["calibration"]], calibrated=False)
            bundle.calibrator = SigmoidCalibrator().fit(cal_raw, frame.loc[masks["calibration"], "isFraud"])
            start = time.perf_counter()
            p_raw = bundle.predict(x.loc[masks["policy"]], calibrated=False)
            p = bundle.calibrator.predict(p_raw)
            scoring_seconds = time.perf_counter() - start
            threshold = fit_threshold(policy_frame.isFraud, p, amount, config["primary_review_rate"])
            cost_thresholds = {s["name"]: fit_cost_threshold(policy_frame.isFraud, p, amount, s)
                               for s in config["cost_scenarios"]}
            for calibration, scores in (("raw", p_raw), ("sigmoid", p)):
                row = _metrics_rows(policy_frame, scores, name, threshold, config)
                row["calibration"] = calibration
                rows.append(row)
                table = calibration_table(policy_frame.isFraud, scores)
                table = table.assign(model=name, period="policy", calibration=calibration)
                calibrated_rows.append(table)
            policy_predictions[name] = p
            joblib.dump(bundle, run / "models" / f"{name}.joblib")
            candidates[name] = {"kind": kind, "params": best_params, "tune_ap": best_ap,
                                "include_anonymous": anonymous, "include_relations": relations,
                                "threshold": threshold, "cost_thresholds": cost_thresholds,
                                "training_seconds": fit_seconds, "tuning_seconds": tuning_seconds,
                                "scoring_seconds": scoring_seconds, "scoring_rows": len(p),
                                "feature_count": len(x.columns),
                                "model_bytes": (run / "models" / f"{name}.joblib").stat().st_size,
                                "model_sha256": digest(run / "models" / f"{name}.joblib"),
                                "n_iter": int(np.max(getattr(estimator.named_steps["model"], "n_iter_", [0]))),
                                "converged": bool(kind != "logistic" or np.max(estimator.named_steps["model"].n_iter_) < config["logistic_max_iter"]),
                                "calibration_slope": float(bundle.calibrator.model.coef_[0, 0])}
            del x, estimator, bundle
        candidates[name]["process_peak_rss_mb"] = resources.peak / (1024 ** 2)
    eligible = [r for r in rows if r.get("calibration") == "sigmoid"]
    # All selection is confined to policy. No final-test result participates here.
    chosen = max(eligible, key=lambda r: (r["recall"] or 0, r["average_precision"] or 0))["model"]
    pd.DataFrame(rows).to_csv(run / "development_metrics.csv", index=False)
    pd.concat(calibrated_rows).to_csv(run / "development_calibration.csv", index=False)
    policy_predictions.to_parquet(run / "policy_predictions.parquet", index=False)
    manifest = {"status": "developed", "completed_utc": utc_now(), "config": config,
                "data_sha256": digest(prepared(config)), "code": code_fingerprint(),
                "environment": environment(), "models": candidates, "chosen_model": chosen,
                "tuning": tuning,
                "split_summary": index.groupby("split", sort=False).agg(
                    n=("TransactionID", "size"), dt_min=("TransactionDT", "min"),
                    dt_max=("TransactionDT", "max")).reset_index().to_dict("records"),
                "selection_rule": "Supervised model selected by policy recall at primary daily review rate, tie-break AP; amount rule is a separate benchmark",
                "training_label_availability_assumption": "Historical batch study assumes labels of each fit block are known before next block; actual confirmation delay is unavailable. No label-history features."}
    dump_json(run / "development.json", manifest)
    dump_json(run / "run_manifest.json", {"status": "developed", "chosen_model": chosen})
    return manifest


def freeze(config):
    run = Path(config["run_dir"])
    manifest = read_json(run / "development.json")
    if manifest["config"] != config or manifest["code"] != code_fingerprint():
        raise ValueError("Config or source changed since development; use a new run")
    if manifest["data_sha256"] != digest(prepared(config)):
        raise ValueError("Prepared data changed since development")
    for name, model in manifest["models"].items():
        if model["kind"] != "rule" and digest(run / "models" / f"{name}.joblib") != model["model_sha256"]:
            raise ValueError(f"Model changed since development: {name}")
    if (run / "frozen_plan.json").exists():
        raise FileExistsError("Plan already frozen")
    tracked = [run / "development.json", run / "split_index.parquet", *sorted((run / "models").glob("*.joblib"))]
    plan = {"frozen_utc": utc_now(), "chosen_model": manifest["chosen_model"],
            "files": {str(p.relative_to(run)): digest(p) for p in tracked},
            "data_sha256": manifest["data_sha256"], "config": config, "code": code_fingerprint()}
    dump_json(run / "frozen_plan.json", plan)
    return plan


def verify_frozen(config):
    run = Path(config["run_dir"])
    plan = read_json(run / "frozen_plan.json")
    if plan["config"] != config or plan["code"] != code_fingerprint():
        raise ValueError("Frozen config/source mismatch")
    if plan["data_sha256"] != digest(prepared(config)):
        raise ValueError("Frozen data mismatch")
    for name, expected in plan["files"].items():
        if digest(run / name) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    return plan, read_json(run / "development.json")


def evaluate(config):
    run = Path(config["run_dir"])
    plan, manifest = verify_frozen(config)
    marker = run / "evaluation_started.json"
    # Exclusive creation also prevents a second concurrent evaluation of this run.
    with marker.open("x", encoding="utf-8") as handle:
        handle.write('{"status":"started"}\n')
    ledger_path = Path(config["processed_dir"]) / "test_exposure_ledger.jsonl"
    previous = [__import__("json").loads(s) for s in ledger_path.read_text().splitlines()] if ledger_path.exists() else []
    seen = any(r["data_sha256"] == plan["data_sha256"] for r in previous)
    exposure = {"utc": utc_now(), "data_sha256": plan["data_sha256"], "run_dir": str(run),
                "status": "previously_exposed" if seen else "first_recorded_local_evaluation"}
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(__import__("json").dumps(exposure) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    frame = pd.read_parquet(prepared(config)).sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    index = pd.read_parquet(run / "split_index.parquet")
    frame["split"] = frame.TransactionID.map(index.set_index("TransactionID")["split"])
    mask = (frame["split"] == "test").to_numpy()
    test = frame.loc[mask].copy()
    predictions = test.copy()
    metric_rows, policy_rows, cost_rows, calibration_rows, sensitivity_rows = [], [], [], [], []
    for name, specification in manifest["models"].items():
        start = time.perf_counter()
        with ResourceMonitor() as resources:
            if specification["kind"] == "rule":
                scores = test.TransactionAmt.to_numpy()
            else:
                bundle = joblib.load(run / "models" / f"{name}.joblib")
                x = _features(frame.drop(columns=["isFraud"]), bundle.include_anonymous, bundle.include_relations)
                raw = bundle.predict(x.loc[mask], calibrated=False)
                scores = bundle.calibrator.predict(raw)
                for calibration, values in (("raw", raw), ("sigmoid", scores)):
                    selected = select_top_rate(test, values, config["primary_review_rate"], config["batch_seconds"])
                    metric_rows.append({"model": name, "period": "test", "calibration": calibration,
                                        **score_metrics(test.isFraud, values, test.TransactionAmt, selected)})
                    calibration_rows.append(calibration_table(test.isFraud, values).assign(
                        model=name, period="test", calibration=calibration))
                del x, bundle
        elapsed = time.perf_counter() - start
        predictions[f"score__{name}"] = scores
        policies, costs = evaluate_policies(test, scores, specification["threshold"],
                                            config["review_rates"], config["cost_scenarios"], config["batch_seconds"])
        policies = policies.assign(model=name, period="test", scoring_including_features_seconds=elapsed,
                                   scoring_rows=len(test), process_peak_rss_mb=resources.peak / 1024 ** 2)
        if specification["kind"] == "rule":
            policies["brier"] = np.nan
            metric_rows.append({"model": name, "period": "test", "calibration": "heuristic_amount",
                                **score_metrics(test.isFraud, scores, test.TransactionAmt), "brier": None})
        policy_rows.append(policies)
        cost_rows.append(costs.assign(model=name, period="test"))
        if name == plan["chosen_model"]:
            base = next((s for s in config["cost_scenarios"] if s["name"] == "base"), config["cost_scenarios"][0])
            for rate in config["review_rates"]:
                selected = select_top_rate(test, scores, rate, config["batch_seconds"])
                for scenario in sensitivity_scenarios(base):
                    sensitivity_rows.append({"model": name, "period": "test", "target_review_rate": rate,
                                             "varied_parameter": scenario["varied_parameter"], "varied_value": scenario["varied_value"],
                                             **scenario_costs(test.isFraud, test.TransactionAmt, selected, scenario)})
        for scenario in config["cost_scenarios"]:
            if "cost_thresholds" in specification:
                threshold = specification["cost_thresholds"][scenario["name"]]
                selected = fixed_threshold(scores, threshold)
                policy_rows.append(pd.DataFrame([{"model": name, "period": "test", "policy": f"cost_threshold_{scenario['name']}",
                                                   "threshold": threshold, **score_metrics(test.isFraud, scores, test.TransactionAmt, selected)}]))
                # Evaluate each frozen scenario-optimized threshold under every scenario.
                for evaluation_scenario in config["cost_scenarios"]:
                    cost_rows.append(pd.DataFrame([{"model": name, "period": "test", "policy": f"cost_threshold_{scenario['name']}",
                                                     "threshold": threshold, **scenario_costs(test.isFraud, test.TransactionAmt, selected, evaluation_scenario)}]))
    predictions.to_parquet(run / "test_predictions.parquet", index=False)
    pd.DataFrame(metric_rows).to_csv(run / "test_metrics.csv", index=False)
    pd.DataFrame([row for part in policy_rows for row in part.to_dict("records")]).to_csv(run / "policy_metrics.csv", index=False)
    pd.concat(cost_rows).to_csv(run / "cost_sensitivity.csv", index=False)
    pd.concat(calibration_rows).to_csv(run / "calibration.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(run / "sensitivity_ofat.csv", index=False)
    dump_json(run / "evaluation.json", {"status": "evaluated", "completed_utc": utc_now(),
                                        "exposure": exposure, "chosen_model": plan["chosen_model"],
                                        "source_outputs": {p.name: digest(p) for p in [run / name for name in ("test_predictions.parquet", "test_metrics.csv", "policy_metrics.csv", "cost_sensitivity.csv", "calibration.csv", "sensitivity_ofat.csv")]},
                                        "final_test_selection_permitted": False})
    return read_json(run / "evaluation.json")
