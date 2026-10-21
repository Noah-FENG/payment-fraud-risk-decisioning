-- Grain: one row for the whole `joined` view (all transactions after the left join).
-- Time range of the relative offset, and amount profile in the original, unstated unit.
SELECT min(TransactionDT) AS time_min,
       max(TransactionDT) AS time_max,
       min(TransactionAmt) AS amount_min,
       median(TransactionAmt) AS amount_median,
       quantile_cont(TransactionAmt, 0.99) AS amount_p99,
       max(TransactionAmt) AS amount_max,
       count(*) FILTER (WHERE TransactionAmt = 0) AS zero_amounts
FROM joined
