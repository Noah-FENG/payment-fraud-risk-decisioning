# Publishing notes: Kaggle rule 8B, dashboards, GitHub

The competition rules were last read on 2026-10-21 (see [DATA_ACCESS](DATA_ACCESS.md)). Re-read the official rules before publishing anything.

## Kaggle rule 8B: provide the public code on Kaggle too

Before or together with making the GitHub repository public:

1. Create a topic in the competition's Discussion tab that links the repository and summarises the method (for example "Payment-fraud review policy: chronological validation, calibration, top-1% daily review"). Do not paste data.
2. Publish a public Kaggle notebook attached to the competition that contains the same code, or the `src/fraud` package with a runner notebook. Use the competition's own data source inside Kaggle and do not upload copies of the data.
3. State in the notebook which stages need the full data and how long they take.
4. Link the topic and notebook from the README once they exist.
5. Keep these out of both places: raw CSVs, per-transaction predictions or case files, credentials.

## Dashboards

The dashboard files contain only aggregate tables, but rule 7B restricts giving competition data to non-participants, and how it treats derived aggregates is not spelled out. The data-bearing workbook is therefore not published. Dashboard design screenshots and the field dictionary (`tableau/schema.json` and [the Tableau guide](TABLEAU_GUIDE.md)) can be shared, and the workbook and extract files stay local. If the rules or the organisers confirm that aggregates may be shared, the workbook can be published from your own Tableau Public account.

## GitHub

Publish the source code (MIT) only: no data, derived per-transaction files, model binaries, credentials or private notes. See `.gitignore`.
