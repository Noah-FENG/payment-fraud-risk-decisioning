"""Proxy-relation feasibility audit, deliberately separate from graph modelling.

An equal tuple is a shared anonymous attribute signature, never a verified entity.
This audit uses no target and does not grant a semantic licence to train a GNN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud.features import RELATION_CANDIDATES, proxy_codes


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, i: int) -> int:
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = int(self.parent[i])
        return i

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def _connect(union: _UnionFind, codes: np.ndarray) -> None:
    first: dict[int, int] = {}
    for position, key in enumerate(codes):
        if key < 0:
            continue
        code = int(key)
        if code in first:
            union.union(position, first[code])
        else:
            first[code] = position


def _component_summary(union: _UnionFind) -> dict:
    if len(union.parent) == 0:
        return {"components": 0, "largest_component_transactions": 0, "largest_component_fraction": None}
    roots = np.fromiter((union.find(i) for i in range(len(union.parent))), dtype=np.int64)
    _, counts = np.unique(roots, return_counts=True)
    return {
        "components": len(counts),
        "largest_component_transactions": int(counts.max()),
        "largest_component_fraction": float(counts.max() / len(roots)),
        "isolated_transaction_fraction": float(counts[counts == 1].sum() / len(roots)),
    }


def _quantiles(values: pd.Series) -> dict:
    if values.empty:
        return {"minimum": None, "median": None, "p95": None, "maximum": None}
    return {
        "minimum": float(values.min()), "median": float(values.median()),
        "p95": float(values.quantile(0.95)), "maximum": float(values.max()),
    }


def audit_relations(frame: pd.DataFrame) -> dict:
    """Audit only a caller-supplied development prefix; final test must be withheld.

    Memory is O(transactions + observed tuples), with no all-pairs similarity matrix.
    Connected components describe incidence connectivity across available history;
    they are not evidence of fraud rings. Structure thresholds are preregistered
    feasibility screens, not learned outcome-based selection.
    """
    if frame.empty:
        raise ValueError("A nonempty development prefix is required")
    times = frame["TransactionDT"].to_numpy(dtype=float)
    global_union = _UnionFind(len(frame))
    transaction_degree = np.zeros(len(frame), dtype=np.int64)
    candidates = {}
    for name, fields in {
        **RELATION_CANDIDATES,
        "broad_email_domain_control": ("P_emaildomain",),
        "broad_device_model_control": ("DeviceInfo",),
    }.items():
        codes = proxy_codes(frame, fields).to_numpy()
        valid = codes >= 0
        records = pd.DataFrame({"code": codes[valid], "time": times[valid]})
        grouped = records.groupby("code", sort=False).agg(
            degree=("time", "size"), first=("time", "min"), last=("time", "max")
        )
        degrees = grouped["degree"]
        repeated_rows = int(degrees[degrees >= 2].sum())
        spans = grouped["last"] - grouped["first"]
        individual_union = _UnionFind(len(frame))
        _connect(individual_union, codes)
        component = _component_summary(individual_union)
        broad = name.startswith("broad_")
        if not broad:
            _connect(global_union, codes)
            transaction_degree += valid
        coverage = float(valid.mean())
        repeated_fraction = repeated_rows / max(int(valid.sum()), 1)
        max_share = float(degrees.max() / len(frame)) if not degrees.empty else 0.0
        screen = (
            not broad and coverage >= 0.05 and repeated_fraction >= 0.10
            and max_share <= 0.05 and component["largest_component_fraction"] <= 0.50
        )
        candidates[name] = {
            "fields": list(fields),
            "all_fields_available": set(fields).issubset(frame.columns),
            "complete_tuple_transactions": int(valid.sum()), "coverage_fraction": coverage,
            "unique_proxy_tuples": len(grouped),
            "repeat_tuple_transactions": repeated_rows,
            "repeated_fraction_of_covered_transactions": repeated_fraction,
            "proxy_node_degree_transactions": _quantiles(degrees),
            "largest_proxy_fraction_of_all_transactions": max_share,
            "proxy_persistence_seconds": _quantiles(spans),
            "proxy_fraction_persisting_over_one_relative_day": (
                float((spans > 86400).mean()) if len(spans) else None
            ),
            "transaction_components": component,
            "role": "broadness_negative_control_excluded_from_graph" if broad else "candidate_anonymous_proxy",
            "passes_structural_screen": bool(screen),
            "semantic_status": "unverified_attribute_equality_not_real_entity_identity",
        }
    combined = _component_summary(global_union)
    structural = any(c["passes_structural_screen"] for c in candidates.values())
    giant = combined["largest_component_fraction"] > 0.50
    if structural and not giant:
        decision = "pending_semantic_review"
        reason = (
            "Some structural screens pass, but anonymous tuple identity and prediction-time "
            "semantics are unverified. Do not train a GNN until those assumptions are defensible."
        )
    else:
        decision = "decline_structural_screen"
        reason = (
            "Insufficient structurally suitable proxy relations or a giant combined component. "
            "Retain the relation audit and compare historical-statistic XGBoost only."
        )
    return {
        "scope": "caller_supplied_development_prefix_only",
        "rows": len(frame), "minimum_offset_seconds": float(times.min()),
        "maximum_offset_seconds": float(times.max()),
        "uses_labels": False,
        "relation_meaning": "Equality of incomplete anonymous attribute signatures; no real customer/card/device identity claim.",
        "missing_policy": "Any missing tuple component excludes that row from the relation.",
        "candidates": candidates,
        "combined_candidate_transaction_components": combined,
        "known_candidate_relations_per_transaction": _quantiles(pd.Series(transaction_degree)),
        "gnn_gate": {"decision": decision, "approved": False, "reason": reason},
        "structural_screen_assumptions": {
            "minimum_coverage": 0.05, "minimum_repeat_fraction_of_covered": 0.10,
            "maximum_proxy_degree_share": 0.05, "maximum_component_share": 0.50,
            "note": "Research feasibility assumptions, not validated business or safety thresholds.",
        },
        "future_gnn_requirements": [
            "Only nodes/edges/features available strictly before each scored transaction.",
            "Equal timestamps cannot observe each other; labels cannot enter graph messages.",
            "Compare against the same-split historical-statistic XGBoost at equal capacity.",
            "Evaluate new proxies, isolated transactions, missing identity and tuning/compute budgets.",
            "Use one simple GraphSAGE family only if semantics and structure justify it.",
        ],
    }
