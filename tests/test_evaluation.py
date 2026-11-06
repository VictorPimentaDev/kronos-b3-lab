import numpy as np
import pandas as pd
import pytest

from kronos_b3_lab.data import demo_candles, validate_candles
from kronos_b3_lab.evaluation import Config, evaluate
from kronos_b3_lab.models import SeasonalNaive


class Recorder:
    name = "recording-test-double"

    def __init__(self):
        self.history = []

    def predict(self, history, dates):
        assert history.date.max() < dates.min()
        assert "actual" not in history
        self.history.append(history.copy())
        return np.repeat(history.close.iloc[-1] * 1.01, len(dates))


def test_future_perturbation_cannot_change_first_forecast_or_signal():
    asset, ibov = demo_candles()
    before = Recorder()
    original = evaluate(asset, ibov, [before])
    altered = asset.copy()
    altered.loc[60:, ["open", "high", "low", "close"]] *= 10
    after = Recorder()
    changed = evaluate(altered, ibov, [after])
    pd.testing.assert_frame_equal(before.history[0], after.history[0])
    np.testing.assert_array_equal(
        original.predictions.forecast[:5], changed.predictions.forecast[:5]
    )
    assert original.predictions.actual.iloc[0] != changed.predictions.actual.iloc[0]
    assert len(before.history) == 24
    assert original.predictions.date.is_unique


def test_next_open_execution_and_costs_match_hand_calculation():
    asset, ibov = demo_candles()
    asset, ibov = asset.iloc[:65].copy(), ibov.iloc[:65].copy()
    # A large overnight gap makes accidental previous-close execution obvious.
    asset.loc[60, "open"] = asset.loc[59, "close"] * 2
    asset.loc[60, "high"] = asset.loc[60, "open"] * 1.01
    result = evaluate(asset, ibov, [Recorder()], Config(cost_bps=100))
    expected = asset.close.iloc[-1] * 0.99 / (asset.open.iloc[60] * 1.01)
    assert result.equity["recording-test-double"].iloc[-1] == pytest.approx(expected)
    ibov_expected = ibov.close.iloc[-1] * 0.99 / (ibov.open.iloc[60] * 1.01)
    assert result.equity["ibov-buy-hold"].iloc[-1] == pytest.approx(ibov_expected)


def test_random_walk_stays_cash_and_flat_scale_is_undefined():
    asset, ibov = demo_candles()
    asset[["open", "high", "low", "close"]] = 10
    result = evaluate(asset, ibov, [SeasonalNaive(1)])
    assert (result.equity["random-walk"] == 1).all()
    assert result.metrics.MAE.iloc[0] == 0
    assert np.isnan(result.metrics.MASE.iloc[0])


def test_seasonal_repeats_last_five_observed_values():
    asset, _ = demo_candles()
    expected = np.resize(asset.close.iloc[:10].to_numpy()[-5:], 12)
    result = SeasonalNaive().predict(asset.iloc[:10], asset.date.iloc[10:22])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("context", 5),
        ("horizon", 0),
        ("horizon", 65),
        ("cost_bps", -1),
        ("cost_bps", float("nan")),
        ("cost_bps", 10000),
    ],
)
def test_invalid_config(field, value):
    with pytest.raises(ValueError):
        Config(**{field: value})


@pytest.mark.parametrize("invalid", ["nan", "negative", "ohlc", "duplicate", "order", "date"])
def test_input_validation(invalid):
    asset, _ = demo_candles()
    if invalid == "nan":
        asset.loc[0, "close"] = np.nan
    elif invalid == "negative":
        asset.loc[0, "volume"] = -1
    elif invalid == "ohlc":
        asset.loc[0, "high"] = 1
    elif invalid == "duplicate":
        asset.loc[1, "date"] = asset.loc[0, "date"]
    elif invalid == "order":
        asset = asset.iloc[::-1]
    else:
        asset.loc[0, "date"] = pd.NaT
    with pytest.raises(ValueError):
        validate_candles(asset)


def test_alignment_and_short_history_fail_explicitly():
    asset, ibov = demo_candles()
    with pytest.raises(ValueError, match="identical"):
        evaluate(asset, ibov.iloc[1:], [Recorder()])
    with pytest.raises(ValueError, match="Not enough"):
        evaluate(asset.iloc[:60], ibov.iloc[:60], [Recorder()])


@pytest.mark.parametrize("output", [[1], [np.nan] * 5, [-1] * 5])
def test_invalid_model_output_is_not_silently_replaced(output):
    class Broken:
        name = "broken"

        def predict(self, history, dates):
            return output

    with pytest.raises(ValueError, match="finite positive"):
        evaluate(*demo_candles(), [Broken()])


def test_duplicate_model_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        evaluate(*demo_candles(), [Recorder(), Recorder()])


@pytest.mark.parametrize(
    "dates",
    [
        ["2025-01-02T00:00:00-03:00", "2025-01-03T00:00:00-02:00"],
        ["2025-01-02T00:00:00Z", "2025-01-03T00:00:00Z"],
        ["2025-01-02T10:00:00", "2025-01-03T10:00:00"],
    ],
)
def test_non_daily_or_timezone_dates_are_rejected_cleanly(dates):
    asset, _ = demo_candles()
    asset = asset.iloc[:2].copy()
    asset["date"] = dates
    with pytest.raises(ValueError, match="timezone-free daily"):
        validate_candles(asset)
