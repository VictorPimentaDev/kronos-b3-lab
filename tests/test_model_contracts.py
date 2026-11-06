"""Adapter contract regressions.

Doubles mirror the upstream signatures audited on 2026-11-06 and have no **kwargs, so a
wrong call fails. They prove how this lab calls upstream, never that weights load or
forecast well. test_upstream_* inspect the real packages when installed and skip otherwise.
Real inference is a manual step: see docs/model-smoke.md.
"""

import inspect
import re
import shlex
import sys
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kronos_b3_lab.cli import parser
from kronos_b3_lab.data import demo_candles
from kronos_b3_lab.evaluation import Config, evaluate
from kronos_b3_lab.models import Chronos, Kronos, TimesFM, make_models

ROOT = Path(__file__).parents[1]


class Samples:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)

    def __getitem__(self, key):
        return Samples(self.data[key])

    def cpu(self):
        return self

    def numpy(self):
        return self.data


@pytest.fixture
def candles():
    return demo_candles()


@pytest.fixture
def seeds(monkeypatch):
    seen = []
    torch = SimpleNamespace(
        float32="float32",
        tensor=lambda values, dtype: np.asarray(values, dtype=np.float32),
        manual_seed=seen.append,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return seen


@pytest.fixture
def chronos_upstream(monkeypatch, seeds):
    calls = SimpleNamespace(load=None, contexts=[], samples=None)

    class ChronosPipeline:
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, device_map, torch_dtype):
            calls.load = (pretrained_model_name_or_path, device_map, torch_dtype)
            return cls()

        def predict(self, inputs, prediction_length=None, num_samples=None):
            calls.contexts.append(inputs)
            if calls.samples is not None:
                return Samples(calls.samples)
            return Samples(np.full((1, num_samples, prediction_length), 35.0))

    monkeypatch.setitem(sys.modules, "chronos", SimpleNamespace(ChronosPipeline=ChronosPipeline))
    return calls


@pytest.fixture
def timesfm_upstream(monkeypatch):
    calls = SimpleNamespace(load=None, config=None, contexts=[])

    @dataclass(frozen=True)
    class ForecastConfig:
        max_context: int = 0
        max_horizon: int = 0
        normalize_inputs: bool = False

    class Network:
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path):
            calls.load = pretrained_model_name_or_path
            return cls()

        def compile(self, forecast_config):
            calls.config = forecast_config

        def forecast(self, horizon, inputs):
            calls.contexts.extend(inputs)
            # Upstream decodes whole 128-step patches before truncating; keep the adapter honest.
            return np.arange(1.0, 129.0).reshape(1, -1), None

    monkeypatch.setitem(
        sys.modules,
        "timesfm",
        SimpleNamespace(TimesFM_2p5_200M_torch=Network, ForecastConfig=ForecastConfig),
    )
    return calls


@pytest.fixture
def kronos_upstream(monkeypatch, seeds):
    calls = SimpleNamespace(load=[], predictor=None, requests=[])

    class Loaded:
        training = True

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path):
            calls.load.append(pretrained_model_name_or_path)
            return cls()

        def eval(self):
            self.training = False
            return self

    class KronosPredictor:
        def __init__(self, model, tokenizer, device=None, max_context=512, clip=5):
            calls.predictor = SimpleNamespace(
                model=model, tokenizer=tokenizer, device=device, max_context=max_context
            )

        def predict(
            self,
            df,
            x_timestamp,
            y_timestamp,
            pred_len,
            T=1.0,
            top_k=0,
            top_p=0.9,
            sample_count=1,
            verbose=True,
        ):
            calls.requests.append(
                SimpleNamespace(
                    df=df,
                    x=x_timestamp,
                    y=y_timestamp,
                    pred_len=pred_len,
                    T=T,
                    top_p=top_p,
                    sample_count=sample_count,
                    verbose=verbose,
                )
            )
            columns = ["open", "high", "low", "close", "volume", "amount"]
            values = [[10.0 * (i + 1) for i in range(len(columns))]] * pred_len
            return pd.DataFrame(values, columns=columns, index=y_timestamp)

    monkeypatch.setitem(
        sys.modules,
        "model",
        SimpleNamespace(Kronos=Loaded, KronosTokenizer=Loaded, KronosPredictor=KronosPredictor),
    )
    return calls


def test_chronos_loads_tiny_checkpoint_on_cpu_in_float32(chronos_upstream):
    Chronos()
    assert chronos_upstream.load == ("amazon/chronos-t5-tiny", "cpu", "float32")


def test_chronos_point_forecast_is_median_of_samples_per_step(chronos_upstream, candles):
    asset, _ = candles
    # A mean would be pulled to 34 by the outlying sample.
    chronos_upstream.samples = [[[1.0, 1.0], [2.0, 3.0], [100.0, 100.0]]]
    forecast = Chronos().predict(asset.iloc[:60], asset.date.iloc[60:62])
    assert forecast.tolist() == [2.0, 3.0]


def test_chronos_reseeds_at_every_origin(chronos_upstream, seeds, candles):
    asset, _ = candles
    model = Chronos(seed=7)
    for _ in range(2):
        model.predict(asset.iloc[:60], asset.date.iloc[60:63])
    assert seeds == [7, 7]


def test_timesfm_compiles_requested_window_with_normalization(timesfm_upstream):
    TimesFM(160, 3)
    assert timesfm_upstream.load == "google/timesfm-2.5-200m-pytorch"
    assert timesfm_upstream.config.max_context == 160
    assert timesfm_upstream.config.max_horizon == 3
    assert timesfm_upstream.config.normalize_inputs is True


def test_timesfm_keeps_only_requested_horizon(timesfm_upstream, candles):
    asset, _ = candles
    forecast = TimesFM(60, 3).predict(asset.iloc[:60], asset.date.iloc[60:63])
    assert forecast.tolist() == [1.0, 2.0, 3.0]


def test_timesfm_missing_torch_backend_is_a_clear_error(monkeypatch):
    # Real timesfm/__init__.py hides ImportError, leaving only ForecastConfig behind.
    monkeypatch.setitem(sys.modules, "timesfm", SimpleNamespace(ForecastConfig=object))
    with pytest.raises(RuntimeError, match=r"no torch backend.*docs/models\.md"):
        make_models(["timesfm"], 60, 5, 42)


def test_kronos_networks_run_in_eval_mode(kronos_upstream):
    # Real regression: huggingface_hub 0.33.1, pinned by upstream, loads in train mode,
    # which left dropout active during forecasting.
    Kronos()
    assert kronos_upstream.predictor.model.training is False
    assert kronos_upstream.predictor.tokenizer.training is False


def test_kronos_unreachable_checkpoint_is_a_clear_error(kronos_upstream, monkeypatch):
    # Real hub mixin behavior offline: config is dropped, then Kronos.__init__ raises TypeError.
    def unreachable(name):
        raise TypeError("__init__() missing 11 required positional arguments")

    monkeypatch.setattr(sys.modules["model"].Kronos, "from_pretrained", unreachable)
    with pytest.raises(OSError, match="check network/cache"):
        make_models(["kronos"], 60, 5, 42)


def test_kronos_loads_small_checkpoint_on_cpu_with_upstream_limit(kronos_upstream):
    Kronos()
    assert kronos_upstream.load == ["NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base"]
    assert kronos_upstream.predictor.device == "cpu"
    assert kronos_upstream.predictor.max_context == 512


def test_kronos_request_matches_documented_sampling_and_shapes(kronos_upstream, candles):
    asset, _ = candles
    history = asset.iloc[40:100]
    Kronos().predict(history, asset.date.iloc[100:103])
    request = kronos_upstream.requests[0]
    assert (request.T, request.top_p, request.sample_count, request.verbose) == (
        1.0,
        0.9,
        5,
        False,
    )
    assert request.pred_len == 3
    assert list(request.df.columns) == ["open", "high", "low", "close", "volume"]
    # Upstream reads .dt, and the frame and timestamps must share a 0-based index.
    assert request.df.index.tolist() == list(range(60))
    assert request.x.index.tolist() == list(range(60))
    assert request.y.index.tolist() == [0, 1, 2]
    assert request.x.dt.weekday.notna().all()
    assert request.y.dt.weekday.notna().all()


def test_kronos_returns_the_close_column(kronos_upstream, candles):
    asset, _ = candles
    forecast = Kronos().predict(asset.iloc[:60], asset.date.iloc[60:62])
    assert forecast.tolist() == [40.0, 40.0]


@pytest.mark.parametrize(
    ("adapter", "upstream"),
    [
        (lambda: Chronos(), "chronos_upstream"),
        (lambda: TimesFM(60, 5), "timesfm_upstream"),
        (lambda: Kronos(), "kronos_upstream"),
    ],
    ids=["chronos", "timesfm", "kronos"],
)
def test_upstream_sees_only_past_prices_at_every_origin(adapter, upstream, request, candles):
    calls = request.getfixturevalue(upstream)
    asset, ibov = candles
    evaluate(asset, ibov, [adapter()], Config(context=60, horizon=5))
    seen = (
        [item.df["close"].to_numpy() for item in calls.requests]
        if upstream == "kronos_upstream"
        else calls.contexts
    )
    origins = list(range(60, len(asset) - 5 + 1, 5))
    assert len(seen) == len(origins)
    closes = asset["close"].to_numpy()
    for origin, context in zip(origins, seen, strict=True):
        assert np.allclose(context, closes[origin - 60 : origin])


@pytest.mark.parametrize(
    ("name", "module"),
    [("chronos", "chronos"), ("timesfm", "timesfm"), ("kronos", "model")],
)
def test_missing_dependency_names_its_cause_and_the_guide(monkeypatch, name, module):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="float32"))
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(RuntimeError, match=rf"{module}.*docs/models\.md") as caught:
        make_models([name], 60, 5, 42)
    assert isinstance(caught.value.__cause__, ImportError)


def test_kronos_accepts_exactly_its_context_limit(kronos_upstream):
    assert make_models(["kronos"], 512, 5, 42)[0].name == "kronos-small"


def test_timesfm_extra_stays_below_the_restrictively_licensed_major():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pyproject["project"]["optional-dependencies"]["timesfm"] == ["timesfm[torch]>=2,<3"]


def documented_runs() -> list:
    runs = []
    for name in ("docs/model-smoke.md", "docs/models.md"):
        text = (ROOT / name).read_text()
        for line in re.findall(r"^(?:\S+=\S+ )*b3-lab run .*$", text, flags=re.MULTILINE):
            runs.append(pytest.param(line, id=f"{name}:{line.split('--models ')[1].split()[0]}"))
    return runs


@pytest.mark.parametrize("line", documented_runs())
def test_documented_commands_fit_the_cli_and_the_fixture(line):
    words = [word for word in shlex.split(line) if "=" not in word or word.startswith("--")]
    args = parser().parse_args(words[1:])
    asset, _ = demo_candles()
    config = Config(args.context, args.horizon)
    assert config.context + config.horizon <= len(asset)
    known = {"seasonal-naive", "random-walk", "chronos", "timesfm", "kronos"}
    assert set(args.models.split(",")) <= known


def parameters(function):
    return set(inspect.signature(function).parameters)


def test_upstream_chronos_signature_when_installed():
    chronos = pytest.importorskip("chronos")
    assert {"inputs", "prediction_length", "num_samples"} <= parameters(
        chronos.ChronosPipeline.predict
    )


def test_upstream_timesfm_signature_when_installed():
    timesfm = pytest.importorskip("timesfm")
    network = timesfm.TimesFM_2p5_200M_torch
    assert {"horizon", "inputs"} <= parameters(network.forecast)
    assert "forecast_config" in parameters(network.compile)
    assert {"max_context", "max_horizon", "normalize_inputs"} <= {
        field.name for field in fields(timesfm.ForecastConfig)
    }


def test_upstream_kronos_signature_when_installed():
    upstream = pytest.importorskip("model")
    if not hasattr(upstream, "KronosPredictor"):
        pytest.skip("an unrelated top-level package is named model")
    assert {"model", "tokenizer", "device", "max_context"} <= parameters(
        upstream.KronosPredictor.__init__
    )
    assert {
        "df",
        "x_timestamp",
        "y_timestamp",
        "pred_len",
        "T",
        "top_p",
        "sample_count",
        "verbose",
    } <= parameters(upstream.KronosPredictor.predict)
