-- Grain: one row per value of one categorical column of the prepared table `t`.
-- Template: {column} is the grouping column, {limit_clause} is empty or "LIMIT k" (top k by count).
-- A missing value is its own segment ('Missing'). Fraud rate = nf / n; famt = fraud-labelled amount.
SELECT coalesce(CAST({column} AS VARCHAR), 'Missing') AS seg,
       count(*) AS n,
       sum(isFraud) AS nf,
       sum(CASE WHEN isFraud = 1 THEN TransactionAmt ELSE 0 END) AS famt
FROM t
GROUP BY seg
ORDER BY n DESC
{limit_clause}
