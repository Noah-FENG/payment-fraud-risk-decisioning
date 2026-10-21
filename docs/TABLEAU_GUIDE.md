# Tableau aggregates and dashboard build guide

The goal is to let a risk or business analyst inspect transaction structure, model quality and the review policy. The real-data aggregates are generated in `artifacts/main/tableau/` (13 CSV files plus `schema.json`). The contract below was checked against the reporting code; each run also writes `schema.json`, listing the actual fields, types, grain, row counts and file SHA-256 values, and that file is the final field dictionary for the run.

## Data rules

All tables come from internal chronological blocks of the official training data. The time axis uses relative day or week windows and is never converted to real dates. Amounts keep the original amount unit and are not labelled AED. Check the competition's data-use terms before sharing even aggregated exports; per-transaction predictions and false-alarm or miss records are not public by default.

`risk_segments.csv`, `time_profile.csv` and `amount_distribution.csv` come straight from `sql/risk_segments.sql`, `sql/time_profile.sql` and `sql/amount_distribution.sql`, whose header comments state each grain and metric definition.

Keep each summary fact table as its own data source where possible. Do not join the per-product risk table to the per-model, per-policy table: their grains differ and a join would duplicate counts and amounts. If a relationship is needed, use only dimension tables verified as unique, and check first that the totals stay unchanged.

## Export contract

Exports go to `run_dir/tableau/` (default `artifacts/main/tableau/`), one set per run so experiments do not overwrite each other. The table lists what each file must keep.

| Table | Grain (one row is) | Required information | Purpose |
| --- | --- | --- | --- |
| `risk_segments.csv` | `period x dimension x segment` | `n, n_fraud, amount_total, fraud_amount_total, fraud_rate, fraud_count_share, fraud_amount_share` | Risk and contribution by product, amount bucket, device and identity coverage |
| `time_profile.csv` | `period x relative_week` | `n, n_fraud, amount_total, fraud_amount_total, fraud_rate, identity_missing_fraction` and offset range | Changes in risk and transaction structure |
| `amount_distribution.csv` | `period x isFraud` | `n, amount_total, amount_p50, amount_p90, amount_p99, amount_max` | Amount distribution of normal versus fraud-labelled transactions |
| `model_metrics.csv` | `model x calibration x period` | `average_precision, roc_auc, brier` and transaction and review counts | Final-test model comparison |
| `model_resources.csv` | one model | `training_seconds, tuning_seconds, scoring_seconds, scoring_rows, feature_count, model_bytes, process_peak_rss_mb` | Training and scoring cost measured during development |
| `primary_capacity_comparison.csv` | one model on the final test at the frozen primary review capacity | Counts and ratios consistent with `policy_metrics.csv` | Direct model comparison at the same capacity |
| `relation_comparison.csv` | `xgboost / xgboost_relations x policy / test`, both sigmoid | Counts, probabilities and capacity metrics consistent with the model metrics table | Development versus final-test view of relation-history statistics |
| `policy_metrics.csv` | model x period x frozen policy | `selected_n, tp, fp, fn, tn, precision, recall, normal_flag_rate, captured_fraud_amount_share` and capacity or threshold | Workload versus identification trade-off |
| `cost_sensitivity.csv` | model x period x frozen policy x cost scenario | Loss rate, review effectiveness, unit costs, each cost component and total cost | Assumption sensitivity analysis |
| `sensitivity_ofat.csv` | main model x daily review rate x one varied cost assumption, others at base | Same cost components as `cost_sensitivity.csv`, plus `varied_parameter, varied_value` | One-assumption-at-a-time sensitivity |
| `calibration.csv` | `model x calibration x period x bin` | `count, fraud_count, mean_predicted, observed_fraud_rate, bin_lower, bin_upper` | Equal-width probability-bin calibration curve and bin support |
| `stability.csv` | `model x final-test dimension x segment` | Group AP / AUC / Brier, transaction and review counts, `target_review_rate, selection_scope` | Performance by relative week and key group |
| `error_segments.csv` | frozen main model x final-test dimension x segment | TP / FP / FN / TN, `false_negative_rate, share_of_all_false_positives, share_of_all_false_negatives` and amount captured | Locating where errors concentrate |

`period` is one of train, tune, calibration, policy, test. `relative_week = floor(TransactionDT / 604800)` and is not a calendar week. `dimension` can be product, a fixed amount bucket, device type, the 20 most frequent development-period device attributes, or the identity-missing flag. `history_new_proxy` is the missing / new / seen state of the card-address proxy key, not a verified new customer.

In `stability.csv` the review selection is made first over each complete relative day of the final test and only then tallied by group; the top 1% is not re-selected inside each product or device group. `process_peak_rss_mb` is a sampled peak of whole-process RSS, not model-only memory or an exact maximum. Resource figures were measured during development and must not be labelled production latency.

Do not average AP or AUC across different splits in one chart. Aggregate group rates as sum of numerators over sum of denominators; AP, AUC and Brier aggregate differently and cannot be summed like plain ratios, and a simple average of AP or AUC cannot recover the overall metric. For overall model comparison use the matching row of `model_metrics.csv`; for fixed-capacity comparison use `primary_capacity_comparison.csv`.

## Metric definitions

```text
Fraud rate            = SUM(n_fraud) / SUM(n)
Review precision      = SUM(tp) / SUM(selected_n)
Fraud recall          = SUM(tp) / SUM(n_fraud)
Normal flag rate      = SUM(fp) / SUM(n_normal)
Review workload share = SUM(selected_n) / SUM(n)
Fraud amount capture  = SUM(captured_fraud_amount) / SUM(fraud_amount_total)
```

Use these only within a single `period`, a single `dimension` and a single model or policy view. Several dimensions each re-tally the same transactions, so summing across dimensions double counts. A Tableau calculated field with a zero denominator returns `NULL` and shows "no applicable sample"; zero must not mask it. Calibration points are bin means, with `count` as tooltip or point size; an empty bin is not an observed rate of zero.

The cost breakdown uses `missed_potential_loss`, `reviewed_residual_potential_loss`, `total_review_cost` and `total_false_positive_cost`, which sum to `scenario_total_cost`. `hypothetical_loss_reduction` and `scenario_net_benefit` are scenario quantities under the stated inputs, not loss actually avoided.

## The four pages

**Risk overview**: top row shows the split, transaction count, fraud count, fraud-labelled amount and identity coverage. A relative-time line shows volume and fraud rate; product, amount bucket, device and identity-missing charts sit side by side with fraud rate and fraud contribution. Tooltips include the sample count and denominator, and small groups are flagged.

**Model validation**: compares AP, ROC-AUC, Brier and training and scoring cost by model. The calibration chart puts mean predicted probability on the x axis and observed fraud share on the y axis, with a `y = x` reference. A time-window view shows whether a model is good in only one period.

**Review policy**: compares precision, recall, normal flag rate, number selected and amount captured by model, split, review capacity and scenario. Costs appear as a decomposed bar chart of residual fraud exposure, review expense and normal-payment friction. Every cost chart is labelled explicitly as a scenario estimate.

**Errors and stability**: shows false-alarm and miss counts and shares by group, with each group's sample size. The identity-missing split and the missing / new / seen proxy-key states are shown on their own. `stability.csv` currently provides window-by-group performance only and does not implement PSI, so the table must not be labelled PSI or described as a completed distribution-drift test. Risk-structure change can be read from `time_profile.csv` and the per-split `risk_segments.csv`.

## Pre-publication checks

Reconcile the dashboard's total transactions, fraud count and amount with the CSVs of the same run and split. Check that filters do not double count, that rates use the right denominator, that final-test and development results are clearly separate, and that cost assumptions are visible. The dashboard must not claim "actual loss avoided", a "production real-time system" or a "completed interactive dashboard" beyond the evidence.

Tableau Public makes uploaded data public. Only upload outputs that the applicable rules allow and that pass an aggregate-disclosure check. Without permission, publish only dashboard design images and the field dictionary and keep the data files local. Before publishing project code, follow rule 8B and provide the code in the competition's Kaggle forum or kernels as well; see [data access and publication conditions](DATA_ACCESS.md) and [PUBLISHING](PUBLISHING.md).
