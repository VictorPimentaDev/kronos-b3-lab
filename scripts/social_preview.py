"""Create a 1200 x 630 repository preview from the checked-in synthetic fixture."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kronos_b3_lab.data import demo_candles

asset, _ = demo_candles()
fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor="#101b25")
fig.text(0.065, 0.88, "VICTOR PIMENTA  /  OPEN RESEARCH", color="#56dfbe", fontsize=12)
fig.text(0.065, 0.67, "KRONOS", color="#eff4f6", fontsize=54, weight="bold")
fig.text(0.065, 0.53, "/ B3 LAB", color="#eff4f6", fontsize=54, weight="bold")
fig.text(
    0.07, 0.36, "Forecasts deserve\na fair test.", color="#b9c6d1", fontsize=24, linespacing=1.4
)
fig.text(0.07, 0.105, "KRONOS  ·  TIMESFM  ·  CHRONOS", color="#56dfbe", fontsize=12)
ax = fig.add_axes((0.57, 0.21, 0.37, 0.55), facecolor="#101b25")
x = np.arange(len(asset))
ax.plot(x, asset.close, color="#56dfbe", lw=2)
ax.fill_between(x, asset.close, asset.close.min() - 2, color="#56dfbe", alpha=0.08)
ax.axis("off")
fig.text(0.59, 0.14, "SYNTHETIC DATA  /  NOT INVESTMENT ADVICE", color="#8298a6", fontsize=8)
fig.savefig(Path(__file__).resolve().parents[1] / "docs/social-preview.png", dpi=100)
plt.close(fig)
