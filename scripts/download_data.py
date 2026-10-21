"""Download only the two official training files after external Kaggle authentication.

Usage: uv run --extra data python scripts/download_data.py
Authenticate with `uv run --extra data kaggle auth login` outside source control.
Never paste a token into this script, a config file, a notebook, or a commit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import zipfile

from fraud.io import digest, dump_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data/raw")
    args = parser.parse_args()
    raw = args.raw_dir.resolve()
    raw.mkdir(parents=True, exist_ok=True)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    files = ["train_transaction.csv", "train_identity.csv"]
    for name in files:
        if (raw / name).exists():
            print(f"Preserving existing {name}; ingest will validate it")
            continue
        api.competition_download_file("ieee-fraud-detection", name, path=str(raw), quiet=False)
        # Official downloads can be zipped individually. Extract only the named file.
        if not (raw / name).exists():
            archive = raw / (name + ".zip")
            if not archive.exists():
                raise FileNotFoundError(f"Official API did not produce {name} or {archive.name}")
            with zipfile.ZipFile(archive) as handle:
                matching = [member for member in handle.infolist() if member.filename == name]
                if len(matching) != 1:
                    raise ValueError(f"Archive must contain exactly the expected {name}")
                with handle.open(matching[0]) as source, (raw / name).open("xb") as target:
                    import shutil
                    shutil.copyfileobj(source, target)
    manifest = {"downloaded_utc": datetime.now(timezone.utc).isoformat(),
                "source": "Kaggle official competition API", "competition": "ieee-fraud-detection",
                "files": {name: {"bytes": (raw / name).stat().st_size, "sha256": digest(raw / name)} for name in files}}
    dump_json(raw / "acquisition_manifest.json", manifest)
    print("Both official labelled training files are available; run fraud prepare")


if __name__ == "__main__":
    main()
