"""Generate descriptive and validation artefacts from a completed evaluation.

This module never fits a model, calibrator, threshold, or final model selection.
All transaction-level outputs stay inside the ignored local run directory.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fraud.features import historical_features
from fraud.io import digest, dump_json, read_json, read_sql
from fraud.metrics import score_metrics
from fraud.policy import select_top_rate


AMOUNT_EDGES = [0, 25, 50, 100, 250, 500, 1000, np.inf]
AMOUNT_LABELS = ["[0,25)", "[25,50)", "[50,100)", "[100,250)",
                 "[250,500)", "[500,1000)", "[1000,infinity)"]
DIMENSIONS = ["ProductCD", "amount_bucket", "DeviceType", "identity_missing", "DeviceInfo_group"]


def _sql_literal(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _read_analysis_frame(config, index: pd.DataFrame) -> pd.DataFrame:
    path = Path(config["processed_dir"]) / "transactions.parquet"
    with duckdb.connect() as connection:
        connection.execute(f"CREATE VIEW source AS SELECT * FROM read_parquet({_sql_literal(path)})")
        available = {row[0] for row in connection.execute("DESCRIBE source").fetchall()}
        requested = ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud", "ProductCD",
                     "DeviceType", "DeviceInfo", "identity_missing", "card1", "card2", "card3",
                     "card5", "addr1"]
        columns = [name for name in requested if name in available]
        frame = connection.execute(f"SELECT {', '.join(_quote(c) for c in columns)} FROM source").df()
    frame = frame.merge(index[["TransactionID", "split"]], on="TransactionID", how="left", validate="one_to_one")
    if frame["split"].isna().any() or len(frame) != len(index):
        raise ValueError("Prepared data and saved split index do not have identical transaction IDs")
    frame = frame.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    for name in ("ProductCD", "DeviceType", "DeviceInfo"):
        if name not in frame:
            frame[name] = pd.Series(pd.NA, index=frame.index, dtype="string")
    if "identity_missing" not in frame:
        raise ValueError("Prepared data needs the exact identity_missing join flag")
    frame["period"] = frame["split"]
    frame["relative_week"] = np.floor(frame.TransactionDT / (86400 * 7)).astype("int64")
    frame["amount_bucket"] = pd.cut(frame.TransactionAmt, AMOUNT_EDGES, right=False, labels=AMOUNT_LABELS).astype("string")
    dev_device = frame.loc[frame["split"] != "test", "DeviceInfo"].dropna().astype(str)
    # This vocabulary is determined from development transactions only, never the test mix.
    top_device = dev_device.value_counts().head(20).index
    frame["DeviceInfo_group"] = frame.DeviceInfo.astype("string").where(
        frame.DeviceInfo.astype("string").isin(top_device), "Other / unseen in development"
    ).where(frame.DeviceInfo.notna(), "Missing")
    return frame


def _risk_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """DuckDB summaries from sql/risk_segments.sql, time_profile.sql and amount_distribution.sql."""
    with duckdb.connect() as connection:
        connection.register("analysis", frame)
        risk = connection.execute(read_sql("risk_segments.sql")).df()
        time_profile = connection.execute(read_sql("time_profile.sql")).df()
        amount_distribution = connection.execute(read_sql("amount_distribution.sql")).df()
    return risk, time_profile, amount_distribution


def _group_rows(frame: pd.DataFrame, scores, decisions, model, dimensions) -> list[dict]:
    rows = []
    values = np.asarray(scores)
    selected = np.asarray(decisions, dtype=bool)
    for dimension in dimensions:
        groups = frame[dimension].astype("string").fillna("Missing")
        for segment in sorted(groups.unique()):
            mask = (groups == segment).to_numpy(dtype=bool)
            subset = frame.loc[mask]
            row = {
                "period": "test", "model": model, "dimension": dimension, "segment": str(segment),
                **score_metrics(subset.isFraud, values[mask], subset.TransactionAmt, selected[mask]),
            }
            if model == "amount_rule":
                row["brier"] = None
            rows.append(row)
    return rows


def _markdown_table(frame: pd.DataFrame, columns: list[str], limit=12) -> str:
    columns = [name for name in columns if name in frame.columns]
    if not len(frame) or not columns:
        return "No eligible rows."
    def cell(value):
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{value:,.5g}"
        return str(value).replace("|", "/").replace("\n", " ")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame[columns].head(limit).itertuples(index=False, name=None)]
    return "\n".join(lines)


def _plots(output: Path, risk, time_profile, calibration, policies, chosen):
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    product = risk[(risk.period == "test") & (risk.dimension == "ProductCD")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(product.segment, product.fraud_rate)
    axes[0].set(ylabel="Fraud count / transactions", title="Test fraud rate by product")
    axes[1].bar(product.segment, product.n_fraud)
    axes[1].set(ylabel="Labelled fraud transactions", title="Test fraud count contribution")
    fig.tight_layout()
    fig.savefig(figures / "risk_segments.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for period, group in time_profile.groupby("period", sort=False):
        axes[0].plot(group.relative_week, group.fraud_rate, marker=".", label=period)
        axes[1].plot(group.relative_week, group.n, marker=".", label=period)
    axes[0].set(ylabel="Fraud count / transactions", title="Relative-time risk and transaction structure")
    axes[0].legend(ncol=5)
    axes[1].set(xlabel="floor(TransactionDT / 604800), not calendar weeks", ylabel="Transactions")
    fig.tight_layout()
    fig.savefig(figures / "time_profile.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    selected = calibration[(calibration.model == chosen) & (calibration["count"] > 0)]
    for method, group in selected.groupby("calibration", sort=False):
        ax.plot(group.mean_predicted, group.observed_fraud_rate, marker="o", label=method)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraud fraction", title=f"Test calibration: {chosen}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "calibration.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    capacity = policies[policies.policy.astype(str).str.startswith("top_rate_")]
    for model, group in capacity.groupby("model", sort=False):
        group = group.sort_values("target_review_rate")
        ax.plot(group.target_review_rate, group.recall, marker="o", label=model)
    ax.set(xlabel="Target fraction per completed relative day", ylabel="Fraud count recall", title="Frozen test review policies")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "review_capacity.png", dpi=160)
    plt.close(fig)


def _write_relation_report(run, manifest, comparison, stability):
    relation = read_json(run / "relation_audit.json")
    gate = relation["gnn_gate"]
    rows = []
    for name, candidate in relation["candidates"].items():
        rows.append({"proxy": name, "coverage": candidate["coverage_fraction"],
                     "repeated_share_of_covered": candidate["repeated_fraction_of_covered_transactions"],
                     "largest_proxy_share": candidate["largest_proxy_fraction_of_all_transactions"],
                     "passes_structure": candidate["passes_structural_screen"]})
    weekly = stability[(stability.dimension == "relative_week") & stability.model.isin(["xgboost", "xgboost_relations"])]
    weekly_delta = None
    if weekly.model.nunique() == 2:
        pivot = weekly.pivot(index="segment", columns="model", values="recall").dropna()
        if len(pivot):
            weekly_delta = (pivot.xgboost_relations - pivot.xgboost)
    delta_text = "Weekly relation-model recall differences cannot be calculated from both comparators."
    if weekly_delta is not None:
        delta_text = (f"On {len(weekly_delta)} test relative weeks with defined recall, the historical-statistic "
                      f"challenger has higher recall in {int((weekly_delta > 0).sum())}, equal recall in "
                      f"{int((weekly_delta == 0).sum())}, and lower recall in {int((weekly_delta < 0).sum())}. "
                      "This is descriptive evidence; it does not establish a stable causal benefit.")
    content = (
        "# Relation audit and GNN decision\n\n"
        f"Development-only audit: {relation['rows']:,} transactions. Gate decision: **{gate['decision']}**. "
        f"{gate['reason']}\n\n"
        + _markdown_table(pd.DataFrame(rows), ["proxy", "coverage", "repeated_share_of_covered", "largest_proxy_share", "passes_structure"])
        + "\n\nEqual anonymous tuples are proxy attribute signatures, not verified customers, cards, devices, or fraud rings. "
        "Any missing tuple component is excluded. Broad email/device controls are not graph relations. "
        "Components use development history for structural diagnosis, not future-inclusive model messages.\n\n"
        "## Tabular relation comparator\n\n"
        + _markdown_table(comparison, ["period", "model", "average_precision", "recall", "precision", "normal_flag_rate", "captured_fraud_amount_share"])
        + "\n\n" + delta_text + "\n\n"
        "The relation challenger uses strict prior counts, mean amounts, and recency with the same time blocks, "
        "daily capacity and bounded XGBoost search. Feature preparation adds computation. "
        f"The final main model remains the policy-selected **{manifest['chosen_model']}**. "
        "No GraphSAGE experiment was run because the semantic/time-availability gate is not approved. "
        "A graph benefit, graph comparison, or GNN result is therefore not claimed.\n"
    )
    (run / "gnn_decision.md").write_text(content, encoding="utf-8")


def generate_report(config) -> dict:
    """Export local aggregates and a report from already frozen/evaluated outputs."""
    run = Path(config["run_dir"])
    evaluation_path = run / "evaluation.json"
    if not evaluation_path.exists():
        raise FileNotFoundError("Complete frozen final evaluation before generating the report")
    evaluation = read_json(evaluation_path)
    if evaluation.get("status") != "evaluated":
        raise ValueError("Only a completed evaluation can generate a report")
    manifest = read_json(run / "development.json")
    if manifest.get("config") is not None and manifest["config"] != config:
        raise ValueError("Reporting config differs from the frozen development config")
    if digest(Path(config["processed_dir"]) / "transactions.parquet") != manifest["data_sha256"]:
        raise ValueError("Prepared data changed after development/evaluation")
    for name, expected_hash in evaluation.get("source_outputs", {}).items():
        if digest(run / name) != expected_hash:
            raise ValueError(f"Saved evaluation output changed: {name}")
    chosen = evaluation["chosen_model"]
    if chosen != manifest["chosen_model"]:
        raise ValueError("Evaluation and development disagree about the frozen main model")
    index = pd.read_parquet(run / "split_index.parquet")
    frame = _read_analysis_frame(config, index)
    predictions = pd.read_parquet(run / "test_predictions.parquet")
    test = frame.loc[frame["split"] == "test"].copy()
    score_columns = [name for name in predictions if name.startswith("score__")]
    test = test.merge(predictions[["TransactionID", *score_columns]], on="TransactionID", how="left", validate="one_to_one")
    if len(test) != len(predictions) or not score_columns or test[score_columns].isna().any().any():
        raise ValueError("Test predictions do not cover the exact saved final-test transactions")
    history = historical_features(frame)
    states = np.where(history.hist_card_address_missing.eq(1), "missing_proxy",
                      np.where(history.hist_card_address_count.eq(0), "new_proxy", "seen_proxy"))
    state_by_id = pd.Series(states, index=frame.TransactionID)
    test["history_new_proxy"] = test.TransactionID.map(state_by_id)
    valid_proxy = history.hist_card_address_missing.eq(0) | history.hist_device_address_missing.eq(0)
    prior_proxy = history.hist_card_address_count.gt(0) | history.hist_device_address_count.gt(0)
    context = np.where(~valid_proxy, "no_valid_proxy", np.where(prior_proxy, "prior_proxy_match", "no_prior_proxy_match"))
    test["relation_context"] = test.TransactionID.map(pd.Series(context, index=frame.TransactionID))
    risk, time_profile, amount_distribution = _risk_tables(frame)
    tableau = run / "tableau"
    tableau.mkdir(exist_ok=True)
    tables = {
        "risk_segments.csv": (risk, "one period x independently analysed dimension x segment"),
        "time_profile.csv": (time_profile, "one temporal split x relative week"),
        "amount_distribution.csv": (amount_distribution, "one temporal split x observed fraud label"),
    }
    stability_rows = []
    selected_by_model = {}
    rate, batch = config["primary_review_rate"], config["batch_seconds"]
    for score_column in score_columns:
        model = score_column.removeprefix("score__")
        decisions = select_top_rate(test, test[score_column], rate, batch)
        selected_by_model[model] = decisions
        stability_rows += _group_rows(
            test, test[score_column], decisions, model,
            ["relative_week", *DIMENSIONS, "history_new_proxy", "relation_context"],
        )
    stability = pd.DataFrame(stability_rows)
    stability["target_review_rate"] = rate
    stability["selection_scope"] = "whole_test_relative_day_before_grouping"
    tables["stability.csv"] = (stability, "one model x final-test dimension x segment, whole-day review decisions")
    errors = stability[(stability.model == chosen) & ~stability.dimension.eq("relative_week")].copy()
    errors["false_negative_rate"] = errors.fn.div(errors.n_fraud.replace(0, np.nan))
    errors["share_of_all_false_positives"] = errors.fp.div(errors.groupby("dimension").fp.transform("sum").replace(0, np.nan))
    errors["share_of_all_false_negatives"] = errors.fn.div(errors.groupby("dimension").fn.transform("sum").replace(0, np.nan))
    tables["error_segments.csv"] = (errors, "one frozen-main-model final-test dimension x segment")
    test_metrics = pd.read_csv(run / "test_metrics.csv")
    development_metrics = pd.read_csv(run / "development_metrics.csv")
    relation_comparison = pd.concat([development_metrics, test_metrics], ignore_index=True)
    relation_comparison = relation_comparison[
        relation_comparison.model.isin(["xgboost", "xgboost_relations"])
        & relation_comparison.calibration.eq("sigmoid")
    ].copy()
    tables["relation_comparison.csv"] = (relation_comparison, "one XGBoost relation comparator x policy/final-test period")
    model_resources = pd.DataFrame([
        {"model": model, **{field: specification.get(field) for field in (
            "training_seconds", "tuning_seconds", "scoring_seconds", "scoring_rows",
            "feature_count", "model_bytes", "process_peak_rss_mb", "n_iter", "converged", "calibration_slope",
        )}} for model, specification in manifest["models"].items()
    ])
    tables["model_resources.csv"] = (model_resources, "one model, development-stage measured resources")
    tables["model_metrics.csv"] = (test_metrics, "one model x probability calibration method x final-test period")
    for source in ("policy_metrics.csv", "cost_sensitivity.csv", "calibration.csv"):
        table = pd.read_csv(run / source)
        tables[source] = (table, {
            "policy_metrics.csv": "one model x frozen review policy x period",
            "cost_sensitivity.csv": "one model x frozen review policy x hypothetical cost scenario x period",
            "calibration.csv": "one model x calibration method x equal-width probability bin x period",
        }[source])
    if (run / "sensitivity_ofat.csv").exists():
        tables["sensitivity_ofat.csv"] = (pd.read_csv(run / "sensitivity_ofat.csv"), "one main model x daily review rate x one varied cost assumption; other assumptions fixed to base")
    policies = tables["policy_metrics.csv"][0]
    primary_comparison = policies[
        policies.policy.astype(str).str.startswith("top_rate_")
        & np.isclose(policies.target_review_rate.fillna(-1), rate)
    ].copy()
    tables["primary_capacity_comparison.csv"] = (primary_comparison, "one model at the frozen primary relative-day review capacity on final test")
    schema = {
        "source": "IEEE-CIS official labelled training data, internally time-split",
        "source_url": "https://www.kaggle.com/competitions/ieee-fraud-detection/data",
        "publication_status": "local_only_review_data_rules_before_publishing_aggregates",
        "amount_unit": "original unspecified units, not asserted to be AED or realised loss",
        "time_definition": "TransactionDT relative offset seconds; relative_week=floor(TransactionDT/604800)",
        "review_definition": "floor(n_in_completed_relative_day * target_rate); no within-group reranking",
        "metric_definitions": {
            "fraud_rate": "n_fraud / n, at the output row's period/dimension/segment",
            "fraud_count_share": "segment fraud count / period fraud count for that dimension",
            "fraud_amount_share": "segment fraud-labelled amount / period fraud-labelled amount for that dimension",
            "average_precision": "sklearn average_precision_score: sum of recall increments times precision; no-fraud groups undefined",
            "precision": "tp / selected_n", "recall": "tp / n_fraud",
            "normal_flag_rate": "fp / n_normal", "captured_fraud_amount_share": "selected fraud-labelled amount / all fraud-labelled amount",
            "brier": "mean squared probability error; undefined for heuristic amount scores",
            "undefined_rates": "Blank/NaN when denominator is zero; never replace with zero",
            "history_new_proxy": "card_address tuple: missing_proxy if incomplete, new_proxy if no strictly prior match, otherwise seen_proxy",
            "relation_context": "both candidate proxies: no_valid_proxy when neither complete, no_prior_proxy_match when all complete proxies have no strictly earlier match, otherwise prior_proxy_match; not real graph entity labels",
            "DeviceInfo_group": "20 most frequent DeviceInfo values in development only; others grouped; nulls Missing",
        },
        "files": {},
    }
    for name, (table, grain) in tables.items():
        table.to_csv(tableau / name, index=False)
        schema["files"][name] = {"grain": grain, "rows": len(table), "columns": [
            {"name": name, "dtype": str(dtype)} for name, dtype in table.dtypes.items()
        ], "sha256": digest(tableau / name)}
    dump_json(tableau / "schema.json", schema)
    decisions = selected_by_model[chosen]
    labels = test.isFraud.to_numpy()
    error_rows = test[["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]].copy()
    error_rows["score"] = test[f"score__{chosen}"].to_numpy()
    error_rows["error_type"] = np.where((labels == 0) & decisions, "FP", np.where((labels == 1) & ~decisions, "FN", "correct"))
    # High-score false positives and high-amount misses make deterministic review examples.
    cases = pd.concat([
        error_rows[error_rows.error_type == "FP"].sort_values(["score", "TransactionID"], ascending=[False, True]).head(10),
        error_rows[error_rows.error_type == "FN"].sort_values(["TransactionAmt", "TransactionID"], ascending=[False, True]).head(10),
    ])
    cases.to_csv(run / "error_cases_local_only.csv", index=False)
    calibration = tables["calibration.csv"][0]
    _plots(run, risk, time_profile, calibration, policies, chosen)
    _write_relation_report(run, manifest, relation_comparison, stability)
    main = test_metrics[(test_metrics.model == chosen) & test_metrics.calibration.eq("sigmoid")]
    chosen_errors = errors[errors.dimension == "ProductCD"].sort_values("fn", ascending=False)
    high_rate = risk[(risk.period == "test") & (risk.dimension == "ProductCD")].sort_values("fraud_rate", ascending=False)
    high_count = risk[(risk.period == "test") & (risk.dimension == "ProductCD")].sort_values("n_fraud", ascending=False)
    costs = tables["cost_sensitivity.csv"][0]
    exposure = evaluation.get("exposure", {}).get("status", "not recorded")
    report = (
        "# Payment Fraud Analytics and Risk Decisioning\n\n"
        f"The frozen main model is **{chosen}**, selected exclusively on the policy development block. "
        f"The final test contains {len(test):,} transactions. Local test exposure status: **{exposure}**. "
        "The tables below describe offline identification and review workload; they do not establish actual avoided losses.\n\n"
        "## Risk distribution\n\n"
        "Fraud rates describe concentration; fraud counts describe workload contribution. "
        "Small high-rate segments should be assessed together with their sample counts.\n\n"
        + _markdown_table(high_rate, ["segment", "n", "n_fraud", "fraud_rate", "fraud_count_share", "fraud_amount_share"])
        + "\n\nProducts ordered by fraud count:\n\n"
        + _markdown_table(high_count, ["segment", "n_fraud", "fraud_count_share", "fraud_amount_total"])
        + "\n\nSee `tableau/amount_distribution.csv` for label-specific amount quantiles and `tableau/time_profile.csv` "
        "for relative-week volume, fraud rate and identity coverage. `figures/risk_segments.png` and `figures/time_profile.png` visualise these exact exports.\n\n"
        "## Model validation\n\n"
        + _markdown_table(primary_comparison, ["model", "average_precision", "roc_auc", "brier", "precision", "recall", "normal_flag_rate", "captured_fraud_amount_share"])
        + "\n\nAverage Precision uses scikit-learn's recall-increment weighted precision definition, not trapezoidal PR area. "
        "ROC-AUC is supplementary. Brier compares probability forecasts; the amount heuristic has no probability interpretation. "
        "Calibration was fitted on calibration data only. Raw and sigmoid results are retained in `tableau/model_metrics.csv` "
        "and `tableau/calibration.csv`; a calibration benefit is not assumed.\n\n"
        f"The main-model review metrics use a target rate of {rate:.3%} per completed relative day, "
        "with floor rounding. Each group's decisions are inherited from whole-day ranking.\n\n"
        + _markdown_table(main, ["n", "n_fraud", "selected_n", "tp", "fp", "fn", "precision", "recall", "normal_flag_rate"])
        + "\n\n## Review strategy and hypothetical costs\n\n"
        + _markdown_table(policies[policies.model == chosen], ["policy", "selected_n", "selected_rate", "precision", "recall", "normal_flag_rate", "captured_fraud_amount_share"])
        + "\n\nCompare capacity and fixed-threshold policies against the amount rule under the same time splits. "
        "Fixed thresholds can exceed or underfill the future workload budget. Daily top-rate review waits until the batch is complete; "
        "it is a historical offline replay, not a production real-time system. "
        "Cost outputs separate missed potential loss, residual loss on reviewed fraud, review costs and false-positive costs. "
        "Loss rates, review effectiveness and unit costs are scenario assumptions. "
        "Historical detected fraud amounts are not realised prevented losses. Review is not rejection.\n\n"
        + _markdown_table(costs[costs.model == chosen], ["policy", "scenario", "missed_potential_loss", "total_review_cost", "total_false_positive_cost", "scenario_total_cost", "scenario_net_benefit"], limit=15)
        + "\n\nTraining, scoring time, model file size and sampled whole-process peak RSS are exported in `tableau/model_resources.csv`. "
        "Peak RSS is a process measurement, not incremental model memory; feature construction is included only where explicitly named in the policy export.\n\n"
        "## Errors, stability and explanations\n\n"
        + _markdown_table(chosen_errors, ["segment", "n", "n_fraud", "fp", "fn", "normal_flag_rate", "false_negative_rate", "share_of_all_false_negatives"])
        + "\n\nUse `tableau/stability.csv` to compare weeks, product, amount, device, identity coverage and new proxy signatures. "
        "Undefined group metrics remain blank when the denominator is zero. `error_cases_local_only.csv` contains up to ten high-score "
        "false positives and ten high-amount false negatives for private case reading; examples are selected, not representative. "
        "See `explanations/` if `fraud explain` has been run; SHAP describes model associations, not causes.\n\n"
        "## Relation study and limitations\n\n"
        "See `gnn_decision.md` and `relation_audit.json` for the development-only relation audit and the historical-statistic XGBoost comparator. "
        "Anonymous equal attributes do not establish a common real customer, device or fraud group. "
        "GNN training has not been authorised by the audit gate.\n\n"
        "Actual label-confirmation delay, upstream construction times for anonymised engineered fields, realised losses, "
        "review outcomes and intervention effects are unavailable. Batch training assumes the previous fit block's labels are available; "
        "anonymous-field ablation is a sensitivity check, not proof that all remaining data are deployable. "
        "This dataset does not establish UAE population relevance, protected-group fairness, calibrated future losses or deployment readiness. "
        "Final-test inspection supports reporting only; new choices require a fresh holdout or explicit repeated-exposure labelling.\n\n"
        "## How to read the tables\n\n"
        "Read each summary table's grain and denominator first, then compare fraud rate with total fraud count. `fp/n_normal` answers how many legitimate transactions are disturbed; "
        "`tp/n_fraud` answers how much historical fraud is caught; amount capture answers how much fraud-labelled transaction amount is identified. These are three different business questions. "
        "Review slots are chosen over the whole day first and then inspected by group; no group gets its own fresh review capacity.\n\n"
        "All report figures and numbers trace to the saved run and `tableau/schema.json`; no Tableau workbook is claimed.\n"
    )
    (run / "report.md").write_text(report, encoding="utf-8")
    bullet = (
        "# Resume evidence from this run\n\n"
        "Draft; verify ownership and explain every component before using.\n\n"
        f"- Built a reproducible DuckDB/Python payment-fraud analytics workflow on {len(frame):,} labelled IEEE-CIS transactions, "
        "preserving missing identity records and validating strict chronological feature histories.\n"
    )
    if len(main):
        row = main.iloc[0]
        bullet += (f"- Compared rule, logistic and XGBoost risk scores with development-only calibration; the frozen {chosen} "
                   f"achieved AP {row.average_precision:.4f} and fraud recall {row.recall:.2%} on the recorded temporal test "
                   f"at a {rate:.2%} target daily review capacity, with normal flag rate {row.normal_flag_rate:.2%}.\n")
    bullet += "- Evaluated review-capacity and hypothetical-cost trade-offs, temporal/group errors and proxy-relation feasibility; no production deployment or realised savings claimed.\n"
    (run / "resume_bullets.md").write_text(bullet, encoding="utf-8")
    result = {"status": "generated_from_saved_evaluation", "main_model": chosen,
              "report": str(run / "report.md"), "tableau_dir": str(tableau),
              "test_rows": len(test), "csv_outputs": len(tables),
              "source_evaluation_sha256": digest(evaluation_path)}
    dump_json(run / "report_manifest.json", result)
    return result
