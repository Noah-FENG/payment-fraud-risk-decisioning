-- Grain: one row for the whole prepared table `t` (one row per transaction).
-- Headline counts for the data audit report. Amounts are in the original, unstated unit.
SELECT count(*) AS n,
       sum(isFraud) AS n_fraud,
       min(TransactionDT) AS time_min,
       max(TransactionDT) AS time_max,
       sum(TransactionAmt) AS amount_total,
       sum(CASE WHEN isFraud = 1 THEN TransactionAmt ELSE 0 END) AS fraud_amount_total,
       sum(CASE WHEN identity_missing THEN 1 ELSE 0 END) AS n_identity_missing
FROM t
