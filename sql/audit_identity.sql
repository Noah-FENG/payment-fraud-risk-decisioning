-- Grain: one row per identity-coverage group (has / no identity record) of the prepared table `t`.
-- The left join keeps transactions without an identity record; missing identity is a data
-- collection structure, not evidence that a transaction is suspicious.
SELECT CASE WHEN identity_missing THEN 'no identity record' ELSE 'has identity record' END AS seg,
       count(*),
       sum(isFraud),
       sum(CASE WHEN isFraud = 1 THEN TransactionAmt ELSE 0 END)
FROM t
GROUP BY seg
ORDER BY seg
