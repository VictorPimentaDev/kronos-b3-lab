import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from kronos_b3_lab.data import demo_candles, fetch_yahoo
from kronos_b3_lab.evaluation import evaluate
from kronos_b3_lab.models import SeasonalNaive
from kronos_b3_lab.report import save_result


def test_cli_installed_entrypoint_creates_evidence(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "kronos_b3_lab", "run", "--out", str(tmp_path)],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "SYNTHETIC" in completed.stdout
    assert (tmp_path / "report.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest["asset_sha256"]) == 64
    assert manifest["models"] == ["seasonal-naive", "random-walk"]
    assert len(pd.read_csv(tmp_path / "predictions.csv")) == 240


def test_cli_requires_asset_and_ibov_together():
    completed = subprocess.run(
        [sys.executable, "-m", "kronos_b3_lab", "run", "--asset", "missing.csv"],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert "both --asset and --ibov" in completed.stderr


def test_result_reproducible(tmp_path):
    asset, ibov = demo_candles()
    first = evaluate(asset, ibov, [SeasonalNaive()])
    second = evaluate(asset, ibov, [SeasonalNaive()])
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    save_result(first, tmp_path, asset, ibov, "SYNTHETIC")
    assert (tmp_path / "equity.csv").is_file()


def test_streamlit_default_and_rerun():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
    assert not app.exception
    assert app.metric[1].value == "24"
    app.slider[1].set_value(10)
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert app.metric[1].value == "12"
    assert "SYNTHETIC" in app.session_state["label"]


def test_fetch_rejects_sidecar_collision_before_network_or_overwrite(tmp_path):
    output = tmp_path / "candles.json"
    output.write_text("existing evidence")
    with pytest.raises(ValueError, match=".csv output path"):
        fetch_yahoo("WEGE3.SA", "2024-01-01", "2025-01-01", output)
    assert output.read_text() == "existing evidence"
