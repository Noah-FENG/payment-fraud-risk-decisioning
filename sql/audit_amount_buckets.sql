-- Grain: one row per fixed amount bucket (original, unstated unit), buckets 0 to 6:
--   0: [0,25)  1: [25,50)  2: [50,100)  3: [100,250)  4: [250,500)  5: [500,1000)  6: [1000, infinity)
-- Input: prepared table `t`. famt is the fraud-labelled amount, which is not realised loss.
SELECT CASE WHEN TransactionAmt >= 0 AND TransactionAmt < 25 THEN 0
            WHEN TransactionAmt >= 25 AND TransactionAmt < 50 THEN 1
            WHEN TransactionAmt >= 50 AND TransactionAmt < 100 THEN 2
            WHEN TransactionAmt >= 100 AND TransactionAmt < 250 THEN 3
            WHEN TransactionAmt >= 250 AND TransactionAmt < 500 THEN 4
            WHEN TransactionAmt >= 500 AND TransactionAmt < 1000 THEN 5
            WHEN TransactionAmt >= 1000 AND TransactionAmt < 'Infinity'::DOUBLE THEN 6 END AS bucket,
       count(*) AS n,
       sum(isFraud) AS nf,
       sum(TransactionAmt) AS amt,
       sum(CASE WHEN isFraud = 1 THEN TransactionAmt ELSE 0 END) AS famt
FROM t
GROUP BY bucket
ORDER BY bucket
