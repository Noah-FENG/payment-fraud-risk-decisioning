"""Command-line entry points. No synthetic-data command is exposed."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from fraud.data import ingest
from fraud.io import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Payment fraud analytics and review decision research")
    parser.add_argument("--config", type=Path, default=Path("config/default.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check-data", "prepare", "develop", "freeze", "evaluate", "report", "explain"):
        sub.add_parser(name)
    score = sub.add_parser("score")
    score.add_argument("--input", type=Path, required=True, help="Joined transaction-grain CSV/Parquet")
    score.add_argument("--history", type=Path, help="Earlier joined history, required for relation model")
    score.add_argument("--output", type=Path, required=True)
    simulation = sub.add_parser("simulate")
    simulation.add_argument("--review-rate", type=float, default=0.01)
    simulation.add_argument("--loss-rate", type=float, default=0.5)
    simulation.add_argument("--effectiveness", type=float, default=0.75)
    simulation.add_argument("--review-cost", type=float, default=2.0)
    simulation.add_argument("--false-positive-cost", type=float, default=5.0)
    simulation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config = load_config(args.config)
        if args.command == "check-data":
            files = [Path(config["raw_dir"]) / f"train_{table}.csv" for table in ("transaction", "identity")]
            result = {"files": {str(p): p.is_file() for p in files},
                      "source": "https://www.kaggle.com/competitions/ieee-fraud-detection/data",
                      "credentials_in_repository": "never required"}
            print(json.dumps(result, indent=2))
            return 0 if all(p.is_file() for p in files) else 2
        if args.command == "prepare":
            processed = Path(config["processed_dir"])
            if (processed / "transactions.parquet").exists():
                raise FileExistsError("Prepared dataset exists; preserve it and configure a new processed_dir")
            result = ingest(Path(config["raw_dir"]), processed)
        elif args.command in {"develop", "freeze", "evaluate"}:
            from fraud import pipeline
            run = Path(config["run_dir"])
            # Keep development logs outside the new run's overwrite guard.
            log_dir = run.parent / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_dir / f"{run.name}_{args.command}.log", encoding="utf-8")
            logging.getLogger().addHandler(handler)
            result = getattr(pipeline, args.command)(config)
            result = {"status": result.get("status", "frozen"), "run_dir": str(run),
                      "chosen_model": result.get("chosen_model")}
        elif args.command == "report":
            from fraud.reporting import generate_report
            result = generate_report(config)
        elif args.command == "explain":
            from fraud.explain import explain
            result = explain(config)
        elif args.command == "score":
            from fraud.scoring import score_file
            result = score_file(config, args.input, args.output, args.history)
        else:
            from fraud.scoring import simulate
            result = simulate(config, args.review_rate, args.loss_rate, args.effectiveness,
                              args.review_cost, args.false_positive_cost, args.output)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        logging.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
