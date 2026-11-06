"""Contract smoke tests, not evidence that model weights perform well."""

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kronos_b3_lab.data import demo_candles
from kronos_b3_lab.models import Chronos, Kronos, TimesFM, make_models


class ArrayTensor:
    def __init__(self, data):
        self.data = np.array(data)

    def __getitem__(self, key):
        return ArrayTensor(self.data[key])

    def cpu(self):
        return self

    def numpy(self):
        return self.data


def test_chronos_contract(monkeypatch):
    calls = {}

    class Pipeline:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls["model"] = name
            return cls()

        def predict(self, context, prediction_length, num_samples):
            calls["context"] = context
            assert num_samples == 20
            return ArrayTensor(np.full((1, 20, prediction_length), 35.0))

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            float32="float32",
            tensor=lambda values, dtype: values,
            manual_seed=lambda seed: None,
        ),
    )
    monkeypatch.setitem(sys.modules, "chronos", SimpleNamespace(ChronosPipeline=Pipeline))
    asset, _ = demo_candles()
    result = Chronos().predict(asset.iloc[:60], asset.date.iloc[60:65])
    assert result.tolist() == [35] * 5
    assert calls["model"] == "amazon/chronos-t5-tiny"
    assert len(calls["context"]) == 60


def test_timesfm_contract(monkeypatch):
    calls = {}

    class Network:
        @classmethod
        def from_pretrained(cls, name):
            assert name == "google/timesfm-2.5-200m-pytorch"
            return cls()

        def compile(self, config):
            calls["config"] = config

        def forecast(self, *, horizon, inputs):
            assert len(inputs) == 1 and len(inputs[0]) == 60
            return np.full((1, horizon), 42), None

    monkeypatch.setitem(
        sys.modules,
        "timesfm",
        SimpleNamespace(
            TimesFM_2p5_200M_torch=Network,
            ForecastConfig=lambda **kwargs: kwargs,
        ),
    )
    asset, _ = demo_candles()
    result = TimesFM(60, 5).predict(asset.iloc[:60], asset.date.iloc[60:65])
    assert result.tolist() == [42] * 5
    assert calls["config"]["max_horizon"] == 5


def test_kronos_contract_includes_calendar_but_no_future_ohlcv(monkeypatch):
    calls = {}

    class Network:
        @classmethod
        def from_pretrained(cls, name):
            return cls()

    class Predictor:
        def __init__(self, network, tokenizer, device, max_context):
            assert device == "cpu" and max_context == 512

        def predict(self, **kwargs):
            calls.update(kwargs)
            return pd.DataFrame({"close": [50] * kwargs["pred_len"]})

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda seed: None))
    monkeypatch.setitem(
        sys.modules,
        "model",
        SimpleNamespace(
            Kronos=Network,
            KronosTokenizer=Network,
            KronosPredictor=Predictor,
        ),
    )
    asset, _ = demo_candles()
    result = Kronos().predict(asset.iloc[:60], asset.date.iloc[60:65])
    assert result.tolist() == [50] * 5
    assert calls["x_timestamp"].max() < calls["y_timestamp"].min()
    assert len(calls["df"]) == 60
    assert list(calls["df"].columns) == ["open", "high", "low", "close", "volume"]


@pytest.mark.parametrize("names", [[], ["unknown"], ["chronos", "chronos"]])
def test_unknown_or_duplicate_models_rejected_before_download(names):
    with pytest.raises(ValueError):
        make_models(names, 60, 5, 42)


def test_kronos_context_limit():
    with pytest.raises(ValueError, match="512"):
        make_models(["kronos"], 513, 5, 42)
