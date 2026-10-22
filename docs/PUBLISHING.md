# Publishing notes: data, dashboards, GitHub

The competition rules were last read on 2026-10-21 (see [DATA_ACCESS](DATA_ACCESS.md)). Re-read the official rules before sharing any data or derived data.

## Scope of what is shared

This repository is an independent study project built on a finished competition's public training data. It shares source code, documentation and aggregate descriptions of results only. It does not contain raw data, per-transaction predictions or case files, model binaries or credentials.

## Dashboards

The dashboard files contain only aggregate tables, but rule 7B restricts giving competition data to non-participants, and how it treats derived aggregates is not spelled out. The data-bearing workbook is therefore not published. Dashboard design screenshots and the field dictionary (`tableau/schema.json` and [the Tableau guide](TABLEAU_GUIDE.md)) can be shared, and the workbook and extract files stay local. If the rules or the organisers confirm that aggregates may be shared, the workbook can be published from your own Tableau Public account.

## GitHub

Publish the source code (MIT) only: no data, derived per-transaction files, model binaries, credentials or private notes. See `.gitignore`.
