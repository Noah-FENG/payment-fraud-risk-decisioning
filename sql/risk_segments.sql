-- Grain: one row per period x dimension x segment (period = chronological split).
-- Input: view `analysis`, one row per transaction with period, the five grouping columns,
--        isFraud (0/1) and TransactionAmt (original, unstated unit).
-- Definitions (shares are always taken inside one period and one dimension):
--   fraud_rate         = n_fraud / n
--   fraud_count_share  = segment fraud count / fraud count of the period for that dimension
--   fraud_amount_share = segment fraud-labelled amount / fraud-labelled amount of the period for that dimension
-- Each dimension re-tallies the same transactions, so never sum rows across dimensions.
-- A missing segment value is reported as 'Missing', never dropped.
WITH groups AS (
    SELECT period, 'ProductCD' AS dimension,
           coalesce(CAST("ProductCD" AS VARCHAR), 'Missing') AS segment,
           count(*) AS n, sum(isFraud) AS n_fraud, sum(TransactionAmt) AS amount_total,
           sum(CASE WHEN isFraud = 1 THEN TransactionAmt ELSE 0 END) AS fraud_amount_total
    FROM analysis GROUP BY period, "ProductCD"
    UNION ALL
    SELECT period, 'amount_bucket' AS dimension,
           coalesce(CAST("amount_bucket" AS VARCHAR), 'Missing') AS segment,
           count(*) AS n, sum(isFraud) AS n_fraud, sum(TransactionAmt) AS amount_total,
           sum(CASE WHEN isFraud = 1 THEN TransactionAmt ELSE 0 END) AS fraud_amount_total
    FROM analysis GROUP BY period, "amount_bucket"
    UNION ALL
    SELECT period, 'DeviceType' AS dimension,
           coalesce(CAST("DeviceType" AS VARCHAR), 'Missing') AS segment,
           count(*) AS n, sum(isFraud) AS n_fraud, sum(TransactionAmt) AS amount_total,
           sum(CASE WHEN isFraud = 1 THEN TransactionAmt ELSE 0 END) AS fraud_amount_total
    FROM analysis GROUP BY period, "DeviceType"
    UNION ALL
    SELECT period, 'identity_missing' AS dimension,
           coalesce(CAST("identity_missing" AS VARCHAR), 'Missing') AS segment,
           count(*) AS n, sum(isFraud) AS n_fraud, sum(TransactionAmt) AS amount_total,
           sum(CASE WHEN isFraud = 1 THEN TransactionAmt ELSE 0 END) AS fraud_amount_total
    FROM analysis GROUP BY period, "identity_missing"
    UNION ALL
    SELECT period, 'DeviceInfo_group' AS dimension,
           coalesce(CAST("DeviceInfo_group" AS VARCHAR), 'Missing') AS segment,
           count(*) AS n, sum(isFraud) AS n_fraud, sum(TransactionAmt) AS amount_total,
           sum(CASE WHEN isFraud = 1 THEN TransactionAmt ELSE 0 END) AS fraud_amount_total
    FROM analysis GROUP BY period, "DeviceInfo_group"
)
SELECT *,
       n_fraud::DOUBLE / nullif(n, 0) AS fraud_rate,
       n_fraud::DOUBLE / nullif(sum(n_fraud) OVER (PARTITION BY period, dimension), 0) AS fraud_count_share,
       fraud_amount_total / nullif(sum(fraud_amount_total) OVER (PARTITION BY period, dimension), 0) AS fraud_amount_share
FROM groups
ORDER BY period, dimension, segment
