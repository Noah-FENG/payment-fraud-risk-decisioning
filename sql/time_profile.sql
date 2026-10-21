-- Grain: one row per chronological split x relative week.
-- Input: view `analysis`, one row per transaction with period, relative_week
--        (= floor(TransactionDT / 604800); a relative week, not a calendar week),
--        TransactionDT (seconds offset), isFraud, TransactionAmt and identity_missing (0/1).
-- Definitions:
--   fraud_rate                = avg(isFraud) = n_fraud / n
--   identity_missing_fraction = share of transactions without an identity record
-- A week that straddles two splits appears once for each split.
SELECT period, relative_week,
       min(TransactionDT) AS minimum_offset_seconds,
       max(TransactionDT) AS maximum_offset_seconds,
       count(*) AS n, sum(isFraud) AS n_fraud,
       sum(TransactionAmt) AS amount_total,
       sum(CASE WHEN isFraud = 1 THEN TransactionAmt ELSE 0 END) AS fraud_amount_total,
       avg(isFraud) AS fraud_rate,
       avg(identity_missing) AS identity_missing_fraction
FROM analysis
GROUP BY period, relative_week
ORDER BY relative_week, period
