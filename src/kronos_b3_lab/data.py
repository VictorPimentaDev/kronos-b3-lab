"""Strict daily OHLCV input: never fill missing prices or silently substitute fixtures."""

import json
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def validate_candles(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")
    frame = frame[COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame.empty or frame["date"].isna().any():
        raise ValueError("Candles must contain dated rows")
    dates = frame["date"]
    if (
        not pd.api.types.is_datetime64_any_dtype(dates)
        or dates.dt.tz is not None
        or not dates.equals(dates.dt.normalize())
    ):
        raise ValueError("Use timezone-free daily session dates")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Dates must be unique and strictly increasing")
    values = frame[COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Candle values must be finite")
    frame[COLUMNS[1:]] = values
    if (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("Prices must be positive; volume cannot be negative")
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("High must contain open, close and low")
    if (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Low must contain open, close and high")
    return frame.reset_index(drop=True)


def load_candles(path: str | Path) -> pd.DataFrame:
    return validate_candles(pd.read_csv(path))


def demo_candles() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = files("kronos_b3_lab").joinpath("fixtures")
    return tuple(load_candles(root.joinpath(name)) for name in ("asset.csv", "ibov.csv"))


def fetch_yahoo(ticker: str, start: str, end: str, output: Path) -> None:
    """Download explicit adjusted daily candles; store provenance alongside CSV."""
    if output.suffix.lower() != ".csv":
        raise ValueError("Use a .csv output path; the JSON sidecar stores provenance separately")
    if not ticker.endswith(".SA") and ticker != "^BVSP":
        raise ValueError("Use a B3 Yahoo symbol such as WEGE3.SA, or ^BVSP")
    import yfinance as yf

    frame = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
        threads=False,
    )
    frame = frame.reset_index().rename(columns=str.lower)
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    frame = validate_candles(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "source": "Yahoo Finance via yfinance",
                "ticker": ticker,
                "downloaded_at": datetime.now(UTC).isoformat(),
                "start": start,
                "end_exclusive": end,
                "auto_adjust": True,
                "rows": len(frame),
                "synthetic": False,
                "warning": "Revised adjusted history is not a point-in-time market database.",
            },
            indent=2,
        )
        + "\n"
    )
