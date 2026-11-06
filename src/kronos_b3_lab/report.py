"""Portable CSV/JSON evidence and a readable, headless matplotlib report."""

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .evaluation import Result

COLORS = ["#56dfbe", "#f5bc64", "#8daaff", "#e49cbf", "#c4cf6a"]


def figure(result: Result, label: str = "SYNTHETIC FIXTURE · NOT MARKET PERFORMANCE"):
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig = plt.figure(figsize=(14, 9), facecolor="#101b25", layout="constrained")
        grid = fig.add_gridspec(3, 2, height_ratios=[0.25, 1, 1])
        header = fig.add_subplot(grid[0, :])
        header.axis("off")
        header.text(0, 0.7, "KRONOS / B3 LAB", color="#f0f4f5", fontsize=23, weight="bold")
        header.text(0, 0.13, label, color="#56dfbe", fontsize=10)
        price = fig.add_subplot(grid[1, :])
        performance = fig.add_subplot(grid[2, 0])
        errors = fig.add_subplot(grid[2, 1])
        for ax in (price, performance, errors):
            ax.set_facecolor("#101b25")
            ax.tick_params(colors="#b9c6d1", labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#344551")
            ax.grid(axis="y", color="#263844", alpha=0.6)

            ax.yaxis.label.set_color("#b9c6d1")
        observed = result.predictions.drop_duplicates("date")
        price.plot(
            observed["date"], observed["actual"], color="#eff4f6", lw=2, label="Observed close"
        )
        for color, (name, group) in zip(
            COLORS, result.predictions.groupby("model", sort=False), strict=False
        ):
            price.plot(
                group["date"], group["forecast"], color=color, lw=1.2, alpha=0.85, label=name
            )
        price.set_title(
            "01  /  OUT-OF-SAMPLE CLOSE FORECASTS", loc="left", fontsize=11, pad=16, color="#f0f4f5"
        )
        price.set_ylabel("Close / BRL")
        price.legend(
            facecolor="#101b25", edgecolor="none", labelcolor="#dce6ed", ncols=3, fontsize=8
        )
        palette = dict(zip(result.metrics["model"], COLORS, strict=False))
        palette.update({"asset-buy-hold": "#eef3f6", "ibov-buy-hold": "#8b9ba6"})
        for column in result.equity.columns[1:]:
            is_benchmark = column.endswith("buy-hold")
            performance.plot(
                result.equity["date"],
                result.equity[column],
                label=column,
                color=palette[column],
                ls="--" if is_benchmark else "-",
                lw=1.5,
            )
        performance.axhline(1, color="#607580", lw=0.6)
        performance.set_title(
            "02  /  LONG–CASH EXPERIMENT", loc="left", fontsize=11, pad=16, color="#f0f4f5"
        )
        performance.set_ylabel("Wealth / starts at 1.00")
        performance.legend(facecolor="#101b25", edgecolor="none", labelcolor="#dce6ed", fontsize=7)
        errors.barh(
            result.metrics["model"],
            result.metrics["MAE"],
            color=COLORS[: len(result.metrics)],
            height=0.5,
        )
        errors.set_title("03  /  ABSOLUTE ERROR", loc="left", fontsize=11, pad=16, color="#f0f4f5")
        errors.set_xlabel("MAE / BRL · lower is better", color="#b9c6d1", fontsize=9)
        errors.invert_yaxis()
        fig.supxlabel(
            "Research only · forecasts are not advice · window-end equity hides intrawindow risk",
            color="#8298a6",
            fontsize=9,
        )
        return fig


def save_result(
    result: Result, output: Path, asset: pd.DataFrame, ibov: pd.DataFrame, label: str
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(output / "predictions.csv", index=False)
    result.equity.to_csv(output / "equity.csv", index=False)
    result.metrics.to_csv(output / "metrics.csv", index=False)
    packages = {}
    for name in ("kronos-b3-lab", "numpy", "pandas", "torch", "timesfm", "chronos-forecasting"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    manifest = {
        "label": label,
        "config": result.config,
        "versions": packages,
        "models": result.metrics["model"].tolist(),
        "asset_sha256": hashlib.sha256(asset.to_csv(index=False).encode()).hexdigest(),
        "ibov_sha256": hashlib.sha256(ibov.to_csv(index=False).encode()).hexdigest(),
        "first_origin": str(result.predictions["origin"].min()),
        "last_target": str(result.predictions["date"].max()),
        "protocol": "disjoint windows; forecast at close; enter next open; liquidate last close",
        "disclaimer": "Research only. Not investment advice. See README for limitations.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    fig = figure(result, label)
    fig.savefig(output / "report.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
