"""Transaction-level ranking, calibration and review metrics.

Average Precision uses sklearn's step-weighted definition, not trapezoidal PR AUC.
Undefined rates return None; their counts remain available for interpretation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _validated_arrays(y, score, amount):
    labels = np.asarray(y)
    scores = np.asarray(score, dtype=float)
    amounts = np.asarray(amount, dtype=float)
    if any(a.ndim != 1 for a in (labels, scores, amounts)):
        raise ValueError("Inputs must be one-dimensional transaction arrays.")
    if not (len(labels) == len(scores) == len(amounts)):
        raise ValueError("Labels, scores and amounts must have equal lengths.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be complete binary 0/1 values.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite.")
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("Amounts must be finite and nonnegative.")
    return labels.astype(np.int8), scores, amounts


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _selection(selected, n: int) -> np.ndarray:
    selected = np.asarray(selected)
    if selected.ndim != 1 or len(selected) != n:
        raise ValueError("Selection must have one value per transaction.")
    if not np.isin(selected, [False, True]).all():
        raise ValueError("Selection must contain only bool or binary values.")
    return selected.astype(bool)


def score_metrics(y, score, amount, selected=None) -> dict:
    """Compute metrics at transaction grain on exactly the supplied period/group.

    ``normal_flag_rate`` has *all actual normal transactions* as its denominator;
    it is different from the share of reviews that are false positives.
    Brier is None for scores outside [0, 1] (e.g. a raw amount ranking).
    """
    labels, scores, amounts = _validated_arrays(y, score, amount)
    n = len(labels)
    n_fraud = int(labels.sum())
    n_normal = n - n_fraud
    fraud_amount_total = float(amounts[labels == 1].sum())
    result = {
        "n": n,
        "n_fraud": n_fraud,
        "n_normal": n_normal,
        "fraud_rate": _safe_ratio(n_fraud, n),
        "transaction_amount_total": float(amounts.sum()),
        "fraud_amount_total": fraud_amount_total,
        "average_precision": (
            float(average_precision_score(labels, scores)) if n_fraud else None
        ),
        "roc_auc": float(roc_auc_score(labels, scores)) if n_fraud and n_normal else None,
        "brier": (
            float(brier_score_loss(labels, scores))
            if n and ((scores >= 0) & (scores <= 1)).all()
            else None
        ),
    }
    selection_fields = (
        "selected_n", "tp", "fp", "fn", "tn", "selected_rate", "precision", "recall",
        "normal_flag_rate", "captured_fraud_amount", "captured_fraud_amount_share",
    )
    if selected is None:
        result.update(dict.fromkeys(selection_fields))
        return result
    selected = _selection(selected, n)
    tp = int(((labels == 1) & selected).sum())
    fp = int(((labels == 0) & selected).sum())
    fn = n_fraud - tp
    tn = n_normal - fp
    selected_n = int(selected.sum())
    captured_amount = float(amounts[(labels == 1) & selected].sum())
    result.update({
        "selected_n": selected_n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "selected_rate": _safe_ratio(selected_n, n),
        "precision": _safe_ratio(tp, selected_n),
        "recall": _safe_ratio(tp, n_fraud),
        "normal_flag_rate": _safe_ratio(fp, n_normal),
        "captured_fraud_amount": captured_amount,
        "captured_fraud_amount_share": _safe_ratio(captured_amount, fraud_amount_total),
    })
    return result


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width probability bins; empty bins retain count=0 and NaN rates."""
    labels, probabilities, _ = _validated_arrays(y, p, np.zeros(len(y)))
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Calibration requires probabilities in [0, 1].")
    bins = np.minimum((probabilities * n_bins).astype(int), n_bins - 1)
    rows = []
    for i in range(n_bins):
        mask = bins == i
        count = int(mask.sum())
        rows.append({
            "bin": i,
            "bin_lower": i / n_bins,
            "bin_upper": (i + 1) / n_bins,
            "count": count,
            "fraud_count": int(labels[mask].sum()),
            "mean_predicted": float(probabilities[mask].mean()) if count else np.nan,
            "observed_fraud_rate": float(labels[mask].mean()) if count else np.nan,
        })
    return pd.DataFrame(rows)


def expected_calibration_error(y, p, n_bins: int = 10) -> float | None:
    """Count-weighted absolute calibration error; depends on chosen binning."""
    table = calibration_table(y, p, n_bins)
    n = int(table["count"].sum())
    if n == 0:
        return None
    error = (table["mean_predicted"] - table["observed_fraud_rate"]).abs().fillna(0)
    return float((error * table["count"]).sum() / n)
