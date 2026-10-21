# Relational information and the GraphSAGE decision

The current decision is an **evidenced decline: no GNN**, based on the 2026-10-21 audit of the real data and the tabular comparison (see the final record below). It is not a claim that GNNs never help; it is the gate's outcome under this project's data and constraints.

## Final record (2026-10-21)

- Audit output: `artifacts/main/relation_audit.json` (531,486 development-period transactions, `TransactionDT` 86,400 to 13,990,904, labels not used). Tabular comparison: `artifacts/main/gnn_decision.md`. Cold-start breakdown: `reports/generated/relation_cold_start.md`, recomputed by `scripts/relation_cold_start.py`.
- Proxy-key definitions: `card_address = (card1, card2, card3, card5, addr1)`; `device_address = (DeviceInfo, addr1)`. If any field is missing, the relation is not created. E-mail domain and device model were only broad controls; they failed the structural screen (largest key holds 38.7% and 8.1% of transactions) and were left out of the graph.
- Structural evidence: `card_address` covers 87.1% of transactions, 96.8% of covered transactions sit on a repeated key, 54.5% of keys persist for more than one relative day, and the largest connected component holds 0.99%. `device_address` covers 13.5%, and 86.9% of transactions are isolated in the graph. The structural screen passes.
- Manual semantic judgement: equal anonymous attribute tuples cannot be read as the same customer, card, device or fraud ring. The real meaning of components such as `addr1` is not public, so their availability at decision time cannot be verified. The semantic condition fails.
- Tabular evidence (same chronological blocks and tuning budget): scheme B (A plus strictly historical relation statistics) versus scheme A gives policy-block AP +0.0004 and test AP -0.0008. In the cold-start breakdown the GNN decision rule requires, the "key seen before" group shows test AP +0.0013 and recall +0.43 pp, and the "first appearance" and "missing" groups show no stable gain. Relation information adds nothing usable on this data.
- Conclusion: a failed semantic screen and no tabular gain are two independent reasons, and either is enough to stop. PyTorch and PyG are not installed and no GraphSAGE experiment was run. Finishing the project does not require a GNN win. This is the final record; the gate can be reopened if new data or semantic information appears.

## Research question

Beyond a strong XGBoost baseline that already uses historical proxy-relation statistics, can a simple GraphSAGE model raise identification at the same review capacity, stably and at a reasonable extra compute cost? Whether relations are useful and whether a GNN is useful are different questions.

## Stage 1: relation feasibility audit

Candidate combinations are limited to anonymous attributes with a defensible business boundary, such as card attributes with address, or device attributes with address. E-mail domain and device model can serve as broad controls but cannot be read as real entities. A transaction missing any required key field creates no relation.

For each candidate relation the audit reports: non-missing coverage, number of distinct proxy keys, share of repeated transactions, distribution of transactions per key, time persistence, share of first appearances, degree distribution, giant-component share and processing complexity. Connectivity is computed on a transaction-to-proxy incidence graph; no all-pairs transaction similarity matrix is built.

Structural statistics alone cannot establish semantics. Even with a high repeat rate, the audit must explain what the relation could link and why transactions sharing attributes cannot be called the same device, customer or fraud ring. The audit uses the development period only; the final-test labels must not be seen before choosing how to build the graph.

## Stage 2: finish the tabular comparison first

Scheme A uses the raw transaction attributes; scheme B adds strictly historical proxy-relation statistics. Both use the same chronological blocks, review capacity, tuning budget and evaluation definitions. Only information available before the transaction is used, and label history is off by default.

If coverage is too low, relations are too broad, time persistence is implausible, or the source cannot support availability at decision time, keep the audit report and stop the graph model. If relations help only for old proxy keys, report new keys and isolated transactions separately; the overall metric must not hide a cold-start problem.

## Stage 3: a single GraphSAGE extension, only if the gate passes

Only when semantics, structure, time availability and the tabular baseline all give sufficient evidence should the graph construction, node and edge definitions, features, sampling and tuning budget be recorded, and PyTorch / PyTorch Geometric introduced. The project does not install them now.

When scoring a transaction, neighbours may come only from nodes and edges that appeared before that moment; transactions with the same `TransactionDT` arrive as one batch and cannot give each other history. Message passing during training must not touch future-block transactions, future labels, or node features precomputed on all the data. Even without labels, connecting future nodes early can leak future structure.

GraphSAGE must be compared with scheme B on AP, precision and recall at fixed capacity, groups with missing identity or new proxy relations, stability across time windows, and training and scoring time and memory. Extra information and tuning budget are logged separately. If the gain is unstable or the cost unreasonable, keep the experiment and use the better tabular model.

## What the final record must contain

The real audit output paths, the periods used, the proxy-key definitions, the structural evidence, the manual semantic judgement, which conditions passed, and the reason to stop or continue. The status moves from deferred to evidenced decline or to GraphSAGE experiment completed only after a real run. The project does not require a GNN to win.
