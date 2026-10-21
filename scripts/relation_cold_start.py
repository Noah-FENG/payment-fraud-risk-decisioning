"""Stage 4: cold-start breakdown of the historical-relation challenger (A vs B).

GNN_DECISION.md requires that relation features proven on old proxy keys must not
hide cold-start performance: report new-key and isolated-transaction groups separately.

For every policy/test transaction and each proxy relation, "seen before" means the
exact complete tuple appeared at a strictly smaller TransactionDT (same-timestamp
batches never count, matching the strict-history rule). Tuple construction uses no
labels. Numbers are recomputed at run time from run artefacts.

Usage:
    uv run python scripts/relation_cold_start.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraud.features import RELATION_CANDIDATES, proxy_codes

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts" / "main"
OUTPUT = ROOT / "reports" / "generated" / "relation_cold_start.md"
RATE = 0.01
BATCH = 86400

MODELS = {"xgboost": "Scheme A (raw attributes)", "xgboost_relations": "Scheme B (+ historical relation statistics)"}


def pct(part, whole) -> str:
    return f"{100.0 * part / whole:.2f}%" if whole else "n/a"


def md_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(lines)


def select_daily_top(frame: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    day = np.floor(frame.TransactionDT.to_numpy() / BATCH).astype(int)
    selected = np.zeros(len(frame), dtype=bool)
    for d in np.unique(day):
        mask = day == d
        k = max(1, int(np.floor(mask.sum() * RATE)))
        top = np.argsort(-score[mask])[:k]
        selected[np.flatnonzero(mask)[top]] = True
    return selected


def main() -> None:
    needed = {"TransactionID", "TransactionDT", "isFraud"}
    fields = sorted({f for cols in RELATION_CANDIDATES.values() for f in cols})
    frame = pd.read_parquet(ROOT / "data" / "processed" / "transactions.parquet",
                            columns=list(needed | set(fields)))
    index = pd.read_parquet(RUN / "split_index.parquet")
    frame = frame.merge(index[["TransactionID", "split"]], on="TransactionID", validate="one_to_one")

    # seen_before: complete tuple observed at strictly earlier TransactionDT (all history).
    for name, cols in RELATION_CANDIDATES.items():
        codes = proxy_codes(frame, cols).to_numpy()
        seen = np.zeros(len(frame), dtype=bool)
        valid = codes >= 0
        key_min_dt = (pd.DataFrame({"k": codes[valid], "dt": frame.TransactionDT.to_numpy()[valid]})
                      .groupby("k")["dt"].min())
        dt_map = key_min_dt.to_dict()
        seen[valid] = [dt > dt_map[c] for c, dt in
                       zip(codes[valid], frame.TransactionDT.to_numpy()[valid])]
        frame[f"{name}_group"] = np.where(~valid, "tuple missing",
                                          np.where(seen, "key seen before", "key first appearance"))

    out = ["# Relation-feature cold-start breakdown (scheme A vs scheme B)\n",
           "Recomputed from the run outputs by `scripts/relation_cold_start.py`. Proxy keys are anonymised attribute tuples "
           "and do not stand for real customers, cards or devices; \"first appearance\" means the complete tuple has not appeared at a strictly earlier TransactionDT "
           "(same-offset batches do not count as history). Daily top-1% review matches the pipeline; read-only evaluation, no change to model selection.\n"]
    for period, pred_file, prefix in (("policy", "policy_predictions.parquet", ""),
                                      ("test", "test_predictions.parquet", "score__")):
        pred = pd.read_parquet(RUN / pred_file)
        seg = frame.loc[frame["split"] == period].merge(
            pred, on="TransactionID", suffixes=("", "_p"), validate="one_to_one")
        rows = []
        for proxy in RELATION_CANDIDATES:
            for group in ("key seen before", "key first appearance", "tuple missing"):
                g = seg.loc[seg[f"{proxy}_group"] == group]
                if not len(g):
                    continue
                row = [proxy, group, f"{len(g):,}", pct(int(g.isFraud.sum()), len(g))]
                aps, recs = [], []
                for model in MODELS:
                    score = g[prefix + model].to_numpy()
                    aps.append(f"{average_precision_score(g.isFraud, score):.4f}"
                               if g.isFraud.nunique() == 2 else "n/a")
                    sel = select_daily_top(g, score)
                    recs.append(pct(int((sel & (g.isFraud.to_numpy() == 1)).sum()), int(g.isFraud.sum())))
                rows.append(row + aps + recs)
        out.append(f"## {period} block\n")
        out.append(md_table(["Proxy relation", "Group", "Transactions", "Fraud rate",
                             "A: AP", "B: AP", "A: recall", "B: recall"], rows))
        out.append("")
    out.append("> Group metrics are descriptive evidence; AP is noisy in small groups and does not support a causal claim of benefit.\n")

    OUTPUT.write_text("\n".join(x for x in out if x != ""), encoding="utf-8")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
