-- Grain: one row per chronological split x observed fraud label.
-- Input: view `analysis`, one row per transaction with period, isFraud and TransactionAmt
--        (original, unstated unit).
-- Quantiles use linear interpolation (quantile_cont). The fraud-labelled rows (isFraud = 1)
-- describe amounts of labelled fraud, which is not realised loss.
SELECT period, isFraud,
       count(*) AS n,
       sum(TransactionAmt) AS amount_total,
       quantile_cont(TransactionAmt, 0.5) AS amount_p50,
       quantile_cont(TransactionAmt, 0.9) AS amount_p90,
       quantile_cont(TransactionAmt, 0.99) AS amount_p99,
       max(TransactionAmt) AS amount_max
FROM analysis
GROUP BY period, isFraud
ORDER BY period, isFraud
