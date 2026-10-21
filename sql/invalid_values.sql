-- Grain: one row for the whole `transaction` view.
-- Counts rows that violate the ingestion contract; any non-zero count stops ingestion
-- (values are rejected, never clipped or silently repaired):
--   bad_labels  isFraud missing or not 0/1
--   bad_time    TransactionDT missing, non-numeric, non-finite or negative
--   bad_amount  TransactionAmt missing, non-numeric, non-finite or negative
SELECT
    count(*) FILTER (WHERE isFraud IS NULL OR try_cast(isFraud AS DOUBLE) IS NULL
                     OR try_cast(isFraud AS DOUBLE) NOT IN (0, 1)) AS bad_labels,
    count(*) FILTER (WHERE TransactionDT IS NULL
                     OR try_cast(TransactionDT AS DOUBLE) IS NULL
                     OR NOT isfinite(try_cast(TransactionDT AS DOUBLE))
                     OR try_cast(TransactionDT AS DOUBLE) < 0) AS bad_time,
    count(*) FILTER (WHERE TransactionAmt IS NULL
                     OR try_cast(TransactionAmt AS DOUBLE) IS NULL
                     OR NOT isfinite(try_cast(TransactionAmt AS DOUBLE))
                     OR try_cast(TransactionAmt AS DOUBLE) < 0) AS bad_amount
FROM transaction
