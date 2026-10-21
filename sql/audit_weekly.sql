-- Grain: one row per relative week (floor(TransactionDT / 604800); not a calendar week).
-- Input: prepared table `t`. id_miss is the share of transactions without an identity record.
SELECT CAST(floor(TransactionDT / (86400 * 7)) AS BIGINT) AS wk,
       count(*) AS n,
       sum(isFraud) AS nf,
       avg(TransactionAmt) AS avg_amt,
       avg(CASE WHEN identity_missing THEN 1.0 ELSE 0.0 END) AS id_miss
FROM t
GROUP BY wk
ORDER BY wk
