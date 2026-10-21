-- Grain: one labelled official training transaction.
-- Inputs: views `transaction` and `identity` over the two official CSV files.
-- Ingestion checks unique, non-null keys in BOTH inputs before this join is created as view `joined`.
-- No WHERE clause on identity: transactions without an identity record must stay in the data as an
-- explicit risk-analysis group; identity_missing = 1 marks them.
SELECT
    t.*,
    i.* EXCLUDE (TransactionID),
    CAST(i.TransactionID IS NULL AS UTINYINT) AS identity_missing
FROM transaction AS t
LEFT JOIN identity AS i USING (TransactionID)
