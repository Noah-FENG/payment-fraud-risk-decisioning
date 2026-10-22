# Payment Fraud Analytics & Risk Decisioning

**Question:** with a limited manual-review queue, how well can a transaction risk score find fraud while keeping legitimate payments from being flagged?

The project uses the official labelled training data of [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data) (Vesta): 590,540 e-commerce transactions, 3.5% of them fraud, over about 182 days of relative time. Only 24.4% of transactions have an identity record, so the identity table is left-joined and "no identity record" is treated as a group of its own. The data itself is not in this repository (see [data access](docs/DATA_ACCESS.md)). Everything here is an offline, retrospective analysis.

## Key findings

1. **A naive "big amounts are risky" rule fails.** Fraud rate is not monotonic in amount (6.97% below 25 versus 2.47% at 1,000 and above), and reviewing the largest amounts catches 1.3% of test fraud.
2. **A calibrated XGBoost model works far better under a fixed review budget.** Reviewing the top 1% of each day's transactions, it reaches 82.7% precision and 21.6% recall on a single held-out test block, and flags 0.18% of legitimate transactions.
3. **The lift comes from anonymised engineered columns, and errors are concentrated.** Without those columns AP falls from 0.51 to 0.20; transactions with no identity record hold 41.5% of test fraud and the model catches only 1.8% of it. Graph-style relation features added nothing, so no GNN was built.

## Method in brief

- **Chronological blocks** (never random): train 324,797, tune 88,582, calibration 59,053, policy 59,054, test 59,054 transactions. Models fit on train (then train plus tune); a calibrator, which turns raw scores into probabilities, fits on its own block; the review threshold and main model are chosen on the policy block; the test block evaluates the frozen design once.
- **No leakage by construction:** `TransactionID` and labels are never features, and historical statistics use only strictly earlier timestamps ([feature availability](docs/FEATURE_AVAILABILITY.md)).
- **Models:** an amount-rule baseline, logistic regression, and XGBoost (two predefined configurations), plus ablations that remove features (an *ablation* removes one input group to see how much the model depends on it). No SMOTE or other resampling.
- **Metrics:** average precision (AP, scikit-learn `average_precision_score`) is primary, with ROC-AUC and Brier score as supplements, plus precision, recall and legitimate-flag rate at a fixed review capacity.
- **Review policy:** review the top 1% of transactions per completed relative day (579 of 59,054 in the test block). Because a day must be complete before it is ranked, this is an offline replay, not a real-time system.
- **Costs:** loss rate, review effectiveness and unit costs are explicit assumptions with a sensitivity analysis. They are scenario estimates, not realised or prevented loss.

## Results

The test block (59,054 transactions, 2,213 fraud, 3.75%) was evaluated **once**, after the plan was frozen. It is a single, registered held-out evaluation that was not used for any selection decision. It is *not* described as never-seen data: its exposure is logged (first recorded 2026-10-21) and no further selection on it is allowed.

| Model (test block) | AP | ROC-AUC | Brier | Review precision | Fraud recall | Normal flag rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Amount rule (baseline) | 0.041 | 0.473 | n/a | 5.0% | 1.3% | 0.97% |
| Logistic regression | 0.178 | 0.825 | 0.0474 | 53.5% | 14.0% | 0.47% |
| **XGBoost (frozen main model)** | **0.512** | **0.898** | **0.0248** | **82.7%** | **21.6%** | **0.18%** |
| XGBoost without anonymous columns | 0.200 | 0.796 | 0.0330 | 36.4% | 9.5% | 0.65% |
| XGBoost + historical proxy-relation statistics | 0.511 | 0.897 | 0.0248 | 82.9% | 21.7% | 0.17% |

Precision, recall and flag rate are at the top-1%-per-day review capacity. On the policy block (used for selection) the frozen XGBoost had AP 0.459, precision 0.808 and recall 0.253.

**How certain are these numbers?** Resampling the test block's 22 relative days with replacement (1,000 draws, seed 42), XGBoost has test AP 0.512 with a 95% day-block bootstrap interval of [0.467, 0.547], ROC-AUC 0.898 [0.889, 0.906], precision 82.7% [76.5%, 88.5%] and recall 21.6% [20.3%, 23.2%] at top 1% per day. The AP advantage over logistic regression is +0.334 [+0.188, +0.395]. The amount rule's AP interval is [0.038, 0.044]. These intervals describe sampling variation across days only (`scripts/test_bootstrap.py`); they are not a new selection step.

- **Review capacity trades recall for precision.** For the frozen model on the test block: 0.1% daily capacity gives precision 91.7% and recall 2.0%; 0.5% gives 90.1% and 11.6%; 1% gives 82.7% and 21.6%.
- **Errors concentrate in a few groups.** Product `W` accounts for 51% of all missed fraud; transactions with no identity record are caught at 1.8% versus 35.7% when an identity record exists.
- **Time matters.** Logistic regression drops from policy-block AP 0.359 to test AP 0.178.
- **Relation features add nothing.** Test AP is 0.5108 with them versus 0.5116 without, and equal anonymous attributes do not identify a real customer, card or device. A GraphSAGE model was therefore deliberately not built; see [the decision record](docs/GNN_DECISION.md). This is a conclusion for this data, not about graph models in general.
- **Dropping the time column changes little.** The main model originally used `TransactionDT` (the relative time offset) as a feature, and test-block values fall outside the training range. A variant without it, evaluated on the tune and policy blocks only, scores nearly the same: tune AP 0.557 → 0.562, policy AP 0.459 → 0.460, policy precision 0.808 → 0.807 and recall 0.253 → 0.252 at top 1% per day. So the model is not leaning on time to separate fraud on those blocks. The frozen main model was left unchanged.

TreeSHAP importances (leading features C1, card6, C14, C5, TransactionAmt) are model associations, not causes.

## Limitations

- Retrospective research on anonymised e-commerce data: no production deployment, no realised savings, no competition rank, no claim of validity for UAE payments.
- Fraud-labelled amount is not chargeback or write-off loss, and amounts keep the dataset's unspecified unit.
- Label delay, how the anonymised fields are built upstream, investigator effectiveness and fairness attributes are unavailable.
- `TransactionDT` was a feature of the frozen model (see the time-column check above); the check covers the tune and policy blocks only.
- Proxy keys are anonymised attribute tuples, not verified customers, cards, devices or fraud rings, and scores for small groups are noisy.
- The test block has been exposed once by design; any new modelling choice needs a fresh holdout.
- The bootstrap intervals use only the 22 days of the test block as resampling units, so they are wide for small-sample metrics such as logistic AP. They do not include the uncertainty from choosing the model or tuning on earlier blocks.

## Reproduce locally

Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). From the repository root in PowerShell:

```powershell
uv sync --extra dev
uv run pytest
```

With a Kaggle account that has accepted the competition rules, put `train_transaction.csv` and `train_identity.csv` in `data/raw/` (`uv run --extra data python scripts/download_data.py` can fetch them). Then run the stages in order; only `evaluate` opens the test block:

```powershell
uv run fraud --config config/default.json check-data
uv run fraud --config config/default.json prepare
uv run python scripts/run_stage_supervised.py develop
uv run fraud --config config/default.json freeze
uv run python scripts/run_stage_supervised.py evaluate
uv run python scripts/run_stage_supervised.py report
uv run python scripts/run_stage_supervised.py explain
```

`develop` takes roughly an hour on the author's machine. The supervised launcher runs a stage with single-threaded BLAS and retries after a native crash seen on that Windows machine (`numpy 2.2.6` and `scipy 1.15.3` are pinned for the same reason); plain `uv run fraud --config config/default.json <stage>` works elsewhere. `evaluate` writes an append-only exposure ledger and should be run once per frozen plan. `uv run python scripts/time_ablation.py` reproduces the time-column check on the tune and policy blocks, and `uv run python scripts/test_bootstrap.py` recomputes the day-block intervals from the saved test predictions (it writes `artifacts/main/test_bootstrap.json` and changes no evaluation output).

Outputs go to `artifacts/main/` (ignored by Git): `report.md`, figures, SHAP explanations and `tableau/`, which holds 13 aggregate CSV files plus `schema.json` (grain, columns, metric definitions and file fingerprints). Per-transaction predictions stay local. [The Tableau guide](docs/TABLEAU_GUIDE.md) defines a four-page dashboard (Risk overview, Model validation, Review policy, Errors and stability); a static workbook was built locally from these aggregates but is not distributed here (see [publishing notes](docs/PUBLISHING.md)). `score` and `simulate` (`uv run fraud <command> --help`) apply the frozen model to new joined data or re-price the review capacity; `simulate` is a post-hoc scenario, not new test evidence.

## SQL

The DuckDB queries live in `sql/` as plain `.sql` files that the code reads and runs; each file states its grain and metric definitions in a header comment. `join_training.sql` (left join that keeps transactions without identity), `invalid_values.sql` and `audit_summary.sql` run during `prepare`; `risk_segments.sql`, `time_profile.sql` and `amount_distribution.sql` build the matching dashboard tables during `report`; the `audit_*.sql` files feed `scripts/data_audit_report.py`. `historical_proxy_statistics.sql` is a readable reference form of the strictly-earlier-history statistics that `src/fraud/features.py` implements. Moving the queries out of Python left every output unchanged: the rebuilt prepared table, the audit record and the three dashboard tables were compared with the earlier outputs and matched exactly.

## Repository layout

```text
config/    Experiment and scenario configuration
src/fraud/ Data, features, models, metrics, policies, reporting, CLI
sql/       The SQL behind the data checks and analysis tables (see below)
scripts/   Data download, supervised stage runner, audit, stability and time-ablation scripts
tests/     Join, time-boundary, feature, scoring, metric and ablation checks
docs/      Charter, data rules, feature availability, GNN decision, Tableau guide, publishing notes
```

## Attribution and license

Source: Addison Howard et al., *IEEE-CIS Fraud Detection*, Kaggle, 2019 ([competition](https://www.kaggle.com/competitions/ieee-fraud-detection), [rules](https://www.kaggle.com/competitions/ieee-fraud-detection/rules)).

The source code is under the [MIT license](LICENSE), which does not grant rights to the competition data, derived records or third-party material. The competition has ended; this repository is an independent study project and contains source code only, with no competition data.
