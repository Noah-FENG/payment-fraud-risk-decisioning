-- Grain: one row per proxy tuple x TransactionDT batch.
-- Reference form only: src/fraud/features.py (historical_features) implements the same logic for the models.
-- Input: view `proxy_transactions` with proxy_tuple, TransactionDT and TransactionAmt.
-- Exact proxy_tuple is constructed only when ALL components are present.
-- First collapse simultaneous transactions, then exclude the entire current group.
-- No isFraud is referenced: label confirmation delay is unavailable.
WITH simultaneous AS (
    SELECT proxy_tuple, TransactionDT,
           count(*) AS batch_count, sum(TransactionAmt) AS batch_amount
    FROM proxy_transactions
    WHERE proxy_tuple IS NOT NULL
    GROUP BY proxy_tuple, TransactionDT
)
SELECT *,
    sum(batch_count) OVER history AS prior_count,
    sum(batch_amount) OVER history / nullif(sum(batch_count) OVER history, 0) AS prior_mean_amount,
    TransactionDT - lag(TransactionDT) OVER (
        PARTITION BY proxy_tuple ORDER BY TransactionDT
    ) AS seconds_since_previous
FROM simultaneous
WINDOW history AS (
    PARTITION BY proxy_tuple ORDER BY TransactionDT
    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
);
