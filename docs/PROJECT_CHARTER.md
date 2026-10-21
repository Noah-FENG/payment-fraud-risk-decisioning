# Project charter: payment-fraud scoring and review policy

The question: with a limited manual-review capacity, how can a transaction risk score raise historical fraud identification while keeping legitimate payments from being flagged? The deliverable is an explainable, reproducible offline analysis for risk, fintech and general data-analysis audiences.

## Data

Only the official `train_transaction.csv` and `train_identity.csv` from the IEEE-CIS Fraud Detection competition are used. The two tables join on `TransactionID`, identity records are incomplete, and `TransactionDT` is a relative time offset, not a calendar timestamp. Access requires a Kaggle account that has accepted the competition rules; the two files go in `data/raw/`. Access restrictions are not bypassed and third-party mirrors are not used.

Sources: [data description](https://www.kaggle.com/competitions/ieee-fraud-detection/data), [rules](https://www.kaggle.com/competitions/ieee-fraud-detection/rules), checked 2026-10-21. The source-code license is separate from the data terms. Raw data, per-transaction derivatives, models and credentials are not published by default.

## Scope and freeze order

1. Data audit: keys, labels, left join that keeps every transaction, missingness, time, amounts, feasibility of proxy relations.
2. Transaction-level analysis and features: DuckDB SQL plus Parquet; strictly historical statistics; a register of feature sources and availability at decision time.
3. Chronological blocks: train, tune, calibration, policy, final test. Fit and select models on the first two; fit calibration on the calibration block only; choose thresholds and the main model on the policy block only.
4. Compare an amount rule, logistic regression and a budget-limited XGBoost. Freeze the tabular pipeline and policy first, then open the final test. The final test is logged on first use, an existing output directory refuses to overwrite the evaluation, and any later change is reported as a repeated-exposure audit test.
5. Compare a fixed threshold, a fixed review capacity per relative day, and cost scenarios. Same-day ranking needs the day's batch to be complete, so it is a historical replay.
6. After the relation audit, compare XGBoost on raw attributes with XGBoost plus strictly historical relation statistics. A GraphSAGE module is considered only if semantics, structure, time constraints and the tabular results all support it.

## Metrics and business interpretation

The primary metric is scikit-learn `average_precision_score` (precision weighted by recall increments, not trapezoidal PR area). Also reported: ROC-AUC, precision and recall at a fixed review capacity, the share of legitimate transactions flagged, the share of fraud-labelled amount identified, Brier score and calibration plots. Every output carries its model, period, group, numerator and denominator.

`TransactionAmt` keeps the dataset's original unit and is not assumed to be AED. Fraud-labelled amount is not realised loss, review is not rejection, and offline capture is not prevented loss. Costs use explicit assumptions (loss rate, review effectiveness, cost per review, cost of a false alarm) with a sensitivity analysis.

## Completion criteria

Engineering completion needs the full CLI flow and key tests to pass. Empirical completion also needs a run on the official training data that yields traceable reports, charts, Tableau aggregates and every number used in the README. Test fixtures only verify code and are not results. No performance, savings, rank or GNN-feasibility claim is made beyond what the saved run supports.
