"""Command line: train, tune, compare, predict, serve."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DATA = Path("data/raw/train.csv")
CONFIG = Path("configs/train.json")
MODEL = Path("artifacts/model")
REPORTS = Path("reports")


def _train(args: argparse.Namespace) -> int:
    from delivery_eta.training import TrainConfig, train

    report, failures = train(args.data, TrainConfig.load(args.config), args.model)
    test = report["test"]
    overall = test["overall"]
    print(f"Tested on {test['orders']:,} orders from {report['data']['test_starts']} onwards:")
    guess = test["guess_the_average_mae"]
    print(f"  average error      {overall['mae']:.2f} min (guessing: {guess:.2f})")
    print(f"  within 5 minutes   {overall['within_5_min']:.0%}")
    print(f"  range is right     {overall['coverage']:.0%}, {overall['mean_width']:.1f} min wide")
    if failures:
        print("Refused by the quality gate; nothing was saved:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Saved model {report['model_version']} to {args.model}")
    return 0


def _tune(args: argparse.Namespace) -> int:
    from delivery_eta.experiments import tune

    before, after = tune(args.data, args.config, args.trials)
    print(f"Average error {before:.4f} -> {after:.4f} min on the validation folds")
    print(f"Saved to {args.config}" if after < before else "Nothing better found; config unchanged")
    return 0


def _compare(args: argparse.Namespace) -> int:
    from delivery_eta.experiments import compare
    from delivery_eta.training import TrainConfig

    result = compare(args.data, TrainConfig.load(args.config), args.out)
    print(result.round(3).to_string(index=False))
    print(f"Saved to {args.out}")
    return 0


def _predict(args: argparse.Namespace) -> int:
    from delivery_eta.batch import predict_file

    done, skipped = predict_file(args.input, args.output, args.model)
    print(f"Predicted {done:,} orders, skipped {skipped:,} (see the problem column): {args.output}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import os

    import uvicorn

    os.environ.setdefault("ETA_MODEL_DIR", str(args.model))
    uvicorn.run(
        "delivery_eta.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        access_log=False,
        log_config=None,
        server_header=False,
        timeout_graceful_shutdown=10,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="delivery-eta", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name: str, run: Any, about: str) -> argparse.ArgumentParser:
        command = commands.add_parser(name, help=about, description=about)
        command.set_defaults(run=run)
        return command

    c = add("train", _train, "Train, check against the quality gate, and save the model.")
    c.add_argument("--data", type=Path, default=DATA)
    c.add_argument("--config", type=Path, default=CONFIG)
    c.add_argument("--model", type=Path, default=MODEL, help="folder to save the model in")
    c.add_argument("--report", type=Path, help="also write the evaluation here as JSON")

    c = add("tune", _tune, "Search for better settings and save them to the config.")
    c.add_argument("--data", type=Path, default=DATA)
    c.add_argument("--config", type=Path, default=CONFIG)
    c.add_argument("--trials", type=int, default=40)

    c = add("compare", _compare, "Compare model types on the same folds.")
    c.add_argument("--data", type=Path, default=DATA)
    c.add_argument("--config", type=Path, default=CONFIG)
    c.add_argument("--out", type=Path, default=REPORTS / "model_comparison.csv")

    c = add("predict", _predict, "Predict every order in a raw CSV.")
    c.add_argument("input", type=Path)
    c.add_argument("output", type=Path)
    c.add_argument("--model", type=Path, default=MODEL)

    c = add("serve", _serve, "Run the API and web page.")
    c.add_argument("--host", default="127.0.0.1")
    c.add_argument("--port", type=int, default=8000)
    c.add_argument("--model", type=Path, default=MODEL)

    args = parser.parse_args(argv)
    try:
        return int(args.run(args))
    except (ValueError, RuntimeError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
