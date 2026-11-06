"""Regenerate fictional candles; no market data or real performance claims."""

from pathlib import Path

import numpy as np
import pandas as pd

from kronos_b3_lab.data import validate_candles

root = Path(__file__).resolve().parents[1] / "src/kronos_b3_lab/fixtures"
rng = np.random.default_rng(20261106)
dates = pd.bdate_range("2025-01-02", periods=180)
market = rng.normal(0.0003, 0.009, len(dates))
for name, initial, returns in (
    ("asset", 36, market * 1.1 + rng.normal(0.0001, 0.008, len(dates))),
    ("ibov", 125000, market),
):
    close = initial * np.exp(np.cumsum(returns))
    opening = np.r_[initial, close[:-1]] * np.exp(rng.normal(0, 0.003, len(dates)))
    spread = rng.uniform(0.002, 0.018, len(dates))
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": opening,
            "high": np.maximum(opening, close) * (1 + spread),
            "low": np.minimum(opening, close) * (1 - spread),
            "close": close,
            "volume": rng.integers(100000, 5000000, len(dates)),
        }
    )
    validate_candles(frame).to_csv(root / f"{name}.csv", index=False, float_format="%.8f")
