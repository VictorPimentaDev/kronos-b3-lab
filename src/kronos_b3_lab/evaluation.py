"""Disjoint rolling-origin forecasts and next-open, long/cash paper experiments."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .data import validate_candles
from .models import Forecaster


@dataclass(frozen=True)
class Config:
    context: int = 60
    horizon: int = 5
    cost_bps: float = 10.0
    seed: int = 42

    def __post_init__(self):
        if self.context < 6 or not 1 <= self.horizon <= 64:
            raise ValueError("Context must be >= 6 and horizon between 1 and 64")
        if not np.isfinite(self.cost_bps) or not 0 <= self.cost_bps < 10000:
            raise ValueError("Cost must be finite, >= 0 and < 10000 basis points per side")


@dataclass
class Result:
    predictions: pd.DataFrame
    equity: pd.DataFrame
    metrics: pd.DataFrame
    config: dict


def evaluate(
    asset: pd.DataFrame, ibov: pd.DataFrame, models: list[Forecaster], config: Config | None = None
) -> Result:
    config = config or Config()
    asset, ibov = validate_candles(asset), validate_candles(ibov)
    if not asset["date"].equals(ibov["date"]):
        raise ValueError("Asset and IBOV must have identical session dates; align explicitly first")
    if len(asset) < config.context + config.horizon:
        raise ValueError("Not enough candles for one complete evaluation window")
    names = [model.name for model in models]
    if (
        not names
        or len(names) != len(set(names))
        or set(names) & {"asset-buy-hold", "ibov-buy-hold"}
    ):
        raise ValueError("Models require unique, non-reserved names")
    predictions, equity = [], []
    capital = dict.fromkeys(names, 1.0)
    first = config.context
    cost = config.cost_bps / 10000
    start = {"date": asset.iloc[first]["date"], "asset-buy-hold": 1.0, "ibov-buy-hold": 1.0}
    equity.append({**start, **capital})
    for origin in range(first, len(asset) - config.horizon + 1, config.horizon):
        history = asset.iloc[origin - config.context : origin].copy()
        future = asset.iloc[origin : origin + config.horizon]
        actual = future["close"].to_numpy()
        scale = np.mean(np.abs(np.diff(history["close"].to_numpy(), n=1)))
        row = {"date": future.iloc[-1]["date"]}
        for model in models:
            # Only calendar dates cross the boundary; no future prices or volumes do.
            forecast = np.asarray(model.predict(history.copy(), future["date"].copy()), dtype=float)
            if (
                forecast.shape != actual.shape
                or not np.isfinite(forecast).all()
                or (forecast <= 0).any()
            ):
                raise ValueError(f"{model.name} must return {len(actual)} finite positive closes")
            for step, (expected, observed) in enumerate(zip(forecast, actual, strict=True)):
                predictions.append(
                    {
                        "model": model.name,
                        "origin": history.iloc[-1]["date"],
                        "date": future.iloc[step]["date"],
                        "step": step + 1,
                        "forecast": expected,
                        "actual": observed,
                        "scale": scale,
                    }
                )
            # Decision is fixed after previous close; executable price is the next open.
            if forecast[-1] > history.iloc[-1]["close"]:
                capital[model.name] *= (
                    actual[-1] * (1 - cost) / (future.iloc[0]["open"] * (1 + cost))
                )
            row[model.name] = capital[model.name]
        for label, candles in (("asset-buy-hold", asset), ("ibov-buy-hold", ibov)):
            end_close = candles.iloc[origin + config.horizon - 1]["close"]
            row[label] = end_close * (1 - cost) / (candles.iloc[first]["open"] * (1 + cost))
        equity.append(row)
    predictions = pd.DataFrame(predictions)
    equity = pd.DataFrame(equity)
    return Result(predictions, equity, summarize(predictions, equity), asdict(config))


def summarize(predictions: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, group in predictions.groupby("model", sort=False):
        error = group["forecast"] - group["actual"]
        scaled = error.abs() / group["scale"].replace(0, np.nan)
        curve = equity[name]
        rows.append(
            {
                "model": name,
                "MAE": error.abs().mean(),
                "RMSE": np.sqrt((error**2).mean()),
                "sMAPE_pct": (
                    200 * error.abs() / (group["forecast"].abs() + group["actual"].abs())
                ).mean(),
                "MASE": scaled.mean(),
                "return_pct": 100 * (curve.iloc[-1] - 1),
                "window_drawdown_pct": 100 * (curve / curve.cummax() - 1).min(),
                "forecasts": len(group),
            }
        )
    return pd.DataFrame(rows)
