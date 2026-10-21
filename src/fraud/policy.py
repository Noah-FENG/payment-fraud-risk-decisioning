"""Offline review policies and explicitly hypothetical cost scenarios.

Top-rate review ranks a completed relative-time batch. It is not a claim of
online scoring, transaction rejection, or actual prevented losses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud.metrics import _selection, _validated_arrays, score_metrics


def fixed_threshold(scores, threshold: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError("Scores and threshold must be finite; scores must be 1D.")
    return scores >= threshold


def select_top_rate(frame, scores, rate: float, batch_seconds: int = 86400) -> np.ndarray:
    """Select floor(batch size * rate) per relative day; ties use ID ascending.

    TransactionID is used only as a deterministic tie-break, never a predictor.
    Input row order and pandas index do not affect results.
    """
    if not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("Review rate must be between zero and one.")
    if not np.isfinite(batch_seconds) or batch_seconds <= 0:
        raise ValueError("batch_seconds must be positive.")
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) != len(frame) or not np.isfinite(scores).all():
        raise ValueError("Provide one finite score per transaction.")
    required = {"TransactionID", "TransactionDT"}
    if not required.issubset(frame.columns):
        raise ValueError("Batch policies require TransactionID and TransactionDT.")
    if frame["TransactionID"].isna().any() or frame["TransactionID"].duplicated().any():
        raise ValueError("TransactionID must be unique and complete.")
    times = frame["TransactionDT"].to_numpy(dtype=float)
    if not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("TransactionDT must be finite and nonnegative.")
    table = pd.DataFrame({
        "position": np.arange(len(frame)),
        "batch": np.floor(times / batch_seconds).astype(np.int64),
        "score": scores,
        "id": frame["TransactionID"].to_numpy(),
    })
    ranked = table.sort_values(["batch", "score", "id"], ascending=[True, False, True])
    ranked["rank"] = ranked.groupby("batch", sort=False).cumcount()
    ranked["capacity"] = np.floor(
        ranked.groupby("batch", sort=False)["position"].transform("size") * rate
    ).astype(int)
    selected = np.zeros(len(frame), dtype=bool)
    selected[ranked.loc[ranked["rank"] < ranked["capacity"], "position"].to_numpy()] = True
    return selected


def fit_threshold(y, score, amount, capacity_rate: float = 0.01, objective="recall") -> float:
    """Choose a fixed global threshold on the policy development period only.

    At most floor(n * capacity_rate) development rows may be selected. All score
    ties receive the same decision, so the available budget can be under-filled.
    The threshold does not enforce a capacity guarantee after distribution drift.
    Labels/amounts are validated but do not change this capacity threshold.
    """
    _, scores, _ = _validated_arrays(y, score, amount)
    if objective != "recall":
        raise ValueError("Only capacity-constrained recall is supported here.")
    if not np.isfinite(capacity_rate) or not 0 <= capacity_rate <= 1:
        raise ValueError("capacity_rate must be between zero and one.")
    if len(scores) == 0:
        raise ValueError("Cannot fit a threshold on an empty policy period.")
    capacity = int(np.floor(len(scores) * capacity_rate))
    if capacity == len(scores):
        return float(scores.min())
    descending = np.sort(scores)[::-1]
    # Strictly exclude the (capacity+1)th score; exclude all ties at that boundary.
    return float(np.nextafter(descending[capacity], np.inf))


def _scenario(scenario: dict) -> dict:
    fields = ("loss_rate", "review_effectiveness", "review_cost", "false_positive_cost")
    if not all(key in scenario for key in fields):
        raise ValueError(f"Scenario requires {fields}.")
    values = {key: float(scenario[key]) for key in fields}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("Scenario inputs must be finite.")
    if not 0 <= values["loss_rate"] <= 1 or not 0 <= values["review_effectiveness"] <= 1:
        raise ValueError("Loss rate and review effectiveness must be in [0, 1].")
    if values["review_cost"] < 0 or values["false_positive_cost"] < 0:
        raise ValueError("Unit costs must be nonnegative.")
    return values


def scenario_costs(y, amount, selected, scenario: dict) -> dict:
    """Cost = missed potential loss + reviewed residual loss + review + FP cost.

    Fraud amounts are observed labeled amounts. Multiplying by loss_rate and
    effectiveness is a scenario, not an empirical estimate of realized losses.
    All amounts and unit costs must use a consistent unspecified monetary unit.
    """
    labels, _, amounts = _validated_arrays(y, np.zeros(len(y)), amount)
    selected = _selection(selected, len(labels))
    assumptions = _scenario(scenario)
    loss_rate = assumptions["loss_rate"]
    effectiveness = assumptions["review_effectiveness"]
    fraud = labels == 1
    reviewed_fraud_amount = float(amounts[fraud & selected].sum())
    missed_fraud_amount = float(amounts[fraud & ~selected].sum())
    total_fraud_amount = reviewed_fraud_amount + missed_fraud_amount
    reviewed_n = int(selected.sum())
    fp = int((~fraud & selected).sum())
    missed_potential_loss = missed_fraud_amount * loss_rate
    reviewed_residual_loss = reviewed_fraud_amount * loss_rate * (1 - effectiveness)
    review_cost = reviewed_n * assumptions["review_cost"]
    fp_cost = fp * assumptions["false_positive_cost"]
    hypothetical_reduction = reviewed_fraud_amount * loss_rate * effectiveness
    total_cost = missed_potential_loss + reviewed_residual_loss + review_cost + fp_cost
    no_review_cost = total_fraud_amount * loss_rate
    return {
        "scenario": str(scenario.get("name", "unnamed")),
        **assumptions,
        "reviewed_n": reviewed_n,
        "false_positive_n": fp,
        "observed_fraud_amount": total_fraud_amount,
        "reviewed_fraud_amount": reviewed_fraud_amount,
        "missed_fraud_amount": missed_fraud_amount,
        "missed_potential_loss": missed_potential_loss,
        "reviewed_residual_potential_loss": reviewed_residual_loss,
        "total_review_cost": review_cost,
        "total_false_positive_cost": fp_cost,
        "hypothetical_loss_reduction": hypothetical_reduction,
        "scenario_total_cost": total_cost,
        "no_review_scenario_cost": no_review_cost,
        "scenario_net_benefit": no_review_cost - total_cost,
    }


def fit_cost_threshold(y, score, amount, scenario: dict, max_candidates: int = 101) -> float:
    """Choose a minimum-scenario-cost threshold using policy-period labels only.

    A bounded quantile grid plus no-review and review-all candidates is searched.
    Scenario-specific thresholds must be frozen before final evaluation.
    """
    labels, scores, amounts = _validated_arrays(y, score, amount)
    _scenario(scenario)
    if len(scores) == 0 or max_candidates < 2:
        raise ValueError("Need nonempty policy data and at least two candidate quantiles.")
    candidates = np.unique(np.concatenate([
        np.quantile(scores, np.linspace(0, 1, max_candidates)),
        [np.nextafter(scores.max(), np.inf)],
    ]))
    # On tied cost, prefer fewer reviews via a higher threshold.
    best = min(
        candidates,
        key=lambda threshold: (
            scenario_costs(labels, amounts, scores >= threshold, scenario)["scenario_total_cost"],
            -threshold,
        ),
    )
    return float(best)


def evaluate_policies(
    frame, scores, threshold: float, rates, scenarios, batch_seconds: int = 86400,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate frozen policies on supplied labeled rows; no fitting occurs here."""
    labels = frame["isFraud"].to_numpy()
    amounts = frame["TransactionAmt"].to_numpy()
    decisions = [("fixed_threshold", fixed_threshold(scores, threshold), None, threshold)]
    for rate in rates:
        decisions.append((
            f"top_rate_{float(rate):g}",
            select_top_rate(frame, scores, float(rate), batch_seconds),
            float(rate),
            None,
        ))
    metric_rows, cost_rows = [], []
    for name, selected, rate, used_threshold in decisions:
        metadata = {"policy": name, "target_review_rate": rate, "threshold": used_threshold}
        metric_rows.append({**metadata, **score_metrics(labels, scores, amounts, selected)})
        for scenario in scenarios:
            cost_rows.append({
                **metadata,
                **scenario_costs(labels, amounts, selected, scenario),
            })
    return pd.DataFrame(metric_rows), pd.DataFrame(cost_rows)
