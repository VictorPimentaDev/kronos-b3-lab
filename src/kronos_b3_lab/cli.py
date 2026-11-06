import argparse
from pathlib import Path

from .data import demo_candles, fetch_yahoo, load_candles
from .evaluation import Config, evaluate
from .models import make_models
from .report import save_result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="B3 forecasting lab — research only, no investment advice"
    )
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Evaluate fixtures or an explicit asset/IBOV CSV pair")
    run.add_argument("--asset", type=Path)
    run.add_argument("--ibov", type=Path)
    run.add_argument("--models", default="seasonal-naive,random-walk")
    run.add_argument("--context", type=int, default=60)
    run.add_argument("--horizon", type=int, default=5)
    run.add_argument("--cost-bps", type=float, default=10)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--out", type=Path, default=Path("runs/demo"))
    fetch = commands.add_parser(
        "fetch", help="Download adjusted Yahoo daily candles; network required"
    )
    fetch.add_argument("ticker")
    fetch.add_argument("--start", required=True)
    fetch.add_argument("--end", required=True)
    fetch.add_argument("--out", type=Path, required=True)
    return root


def main() -> None:
    root = parser()
    args = root.parse_args()
    try:
        if args.command == "fetch":
            fetch_yahoo(args.ticker, args.start, args.end, args.out)
            print(f"Saved candles and provenance to {args.out}")
            return
        if bool(args.asset) != bool(args.ibov):
            raise ValueError("Pass both --asset and --ibov, or neither for synthetic demo")
        if args.asset:
            asset, ibov = load_candles(args.asset), load_candles(args.ibov)
            label = "USER CSV · VERIFY SOURCE AND ADJUSTMENTS"
        else:
            asset, ibov = demo_candles()
            label = "SYNTHETIC FIXTURE · NOT MARKET PERFORMANCE"
        config = Config(args.context, args.horizon, args.cost_bps, args.seed)
        names = [name.strip() for name in args.models.split(",")]
        models = make_models(names, config.context, config.horizon, config.seed)
        result = evaluate(asset, ibov, models, config)
        save_result(result, args.out, asset, ibov, label)
        print(label)
        print(result.metrics.round(4).to_string(index=False))
        print(f"Saved report, CSVs and manifest to {args.out}")
        print("Not investment advice. Synthetic results do not establish predictive skill.")
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        root.exit(2, f"b3-lab: {exc}\n")
