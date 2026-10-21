# Data access and source register

This project uses only the official labelled training data of the IEEE-CIS Fraud Detection competition. The data is anonymised, so it suits research on payment-risk scoring but cannot be used to recover real customers, cards, devices or real losses.

## What the official description says

| Item | Official meaning and how this project treats it |
| --- | --- |
| `isFraud` | Binary fraud label; the supervised target |
| `TransactionID` | Join key between the transaction and identity tables; never used as a feature |
| `transaction` and `identity` | Not every transaction has an identity record, so the join is a left join that keeps all transactions |
| `TransactionDT` | Time offset from an undisclosed reference moment; not a real timestamp |
| `train_*.csv` | Official labelled data; split chronologically inside this project |
| `test_*.csv` | Official unlabelled data; cannot yield AP, AUC, recall or any other truth-based metric |
| Data license | Governed by the Competition Rules; the MIT license of the source code does not apply to the data |

Sources: [data description](https://www.kaggle.com/competitions/ieee-fraud-detection/data), [overview](https://www.kaggle.com/competitions/ieee-fraud-detection/overview), [Competition Rules](https://www.kaggle.com/competitions/ieee-fraud-detection/rules). The summary below reflects the rules as read on 2026-10-21; the full official text is authoritative.

## Data use and code publication

- **Rule 7A** limits data use to non-commercial purposes, including participating in the competition, the Kaggle forums, and academic research and education. Research use here does not extend to a production payment system or any commercial use.
- **Rule 7B** restricts giving the data to people who are not competition participants and requires reasonable security measures. The project therefore keeps raw data, per-transaction derivatives and models local; a file does not become publishable because it is called "processed".
- **Rule 8B** allows public competition code but requires that the code also be available in the competition's Kaggle forum or kernels. Meet this before publishing the repository on GitHub; see [PUBLISHING](PUBLISHING.md) for the checklist. The MIT license covers the source code only.

## Getting the data

You need a Kaggle account that has accepted the competition rules.

1. Open the competition's data page and read the rules.
2. Download `train_transaction.csv` and `train_identity.csv`.
3. Put them in `data/raw/`, or set `raw_dir` in `config/default.json` to the folder that holds them.
4. Run `uv run fraud --config config/default.json check-data`, then `prepare`, which verifies keys, labels and the left join.

With the Kaggle CLI configured (credentials stay outside the repository and are never written into code, config or notebooks), the official command downloads one file at a time:

```powershell
kaggle competitions download -c ieee-fraud-detection -f train_transaction.csv -p data/raw
kaggle competitions download -c ieee-fraud-detection -f train_identity.csv -p data/raw
```

If you receive `.zip` files, unzip them to exactly those two file names. The project does not depend on the Kaggle CLI, and the rules must be accepted by you on Kaggle itself. Do not use unofficial mirrors.

## Local data handling

`prepare` writes a transaction-level Parquet file and a data-audit record. The raw CSVs stay read-only, and missing-value handling and feature transforms happen in derived tables. The audit record keeps input paths, file fingerprints, record counts, field counts, time range and missingness so that a run can be traced.

Not committed by default: raw CSVs, archives, per-transaction Parquet files, per-transaction predictions or cases, fitted models, credentials and environment secrets. `.gitignore` is a safeguard, so check the staging area before publishing. Check the rules before sharing aggregate outputs too, and avoid exposing small groups or individual cases.

Amounts keep the dataset's original, unstated unit and are not labelled USD or AED. The amount of transactions with `isFraud = 1` is called "fraud-labelled transaction amount"; it is not a write-off, chargeback, net loss or recovered amount. The small synthetic data used in tests only checks algorithm edge cases and is never a result.
