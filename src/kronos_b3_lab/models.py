"""Small adapters around upstream APIs. No model weights are bundled."""

from typing import Protocol

import numpy as np
import pandas as pd


class Forecaster(Protocol):
    name: str

    def predict(self, history: pd.DataFrame, dates: pd.Series) -> np.ndarray: ...


class SeasonalNaive:
    def __init__(self, season: int = 5):
        if season < 1:
            raise ValueError("Season must be positive")
        self.season = season
        self.name = "seasonal-naive" if season > 1 else "random-walk"

    def predict(self, history: pd.DataFrame, dates: pd.Series) -> np.ndarray:
        if len(history) < self.season:
            raise ValueError("History is shorter than the seasonal period")
        return np.resize(history["close"].to_numpy()[-self.season :], len(dates))


class Chronos:
    name = "chronos-t5-tiny"

    def __init__(self, seed: int = 42):
        import torch
        from chronos import ChronosPipeline

        self.torch = torch
        self.seed = seed
        self.pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny",
            device_map="cpu",
            torch_dtype=torch.float32,
        )

    def predict(self, history: pd.DataFrame, dates: pd.Series) -> np.ndarray:
        self.torch.manual_seed(self.seed)
        context = self.torch.tensor(history["close"].to_numpy(), dtype=self.torch.float32)
        samples = self.pipeline.predict(context, prediction_length=len(dates), num_samples=20)
        return np.median(samples[0].cpu().numpy(), axis=0)


class TimesFM:
    name = "timesfm-2.5-200m"

    def __init__(self, context: int, horizon: int):
        import timesfm

        # timesfm/__init__.py swallows ImportError, so a missing backend is a missing attribute.
        if not hasattr(timesfm, "TimesFM_2p5_200M_torch"):
            raise ImportError("timesfm has no torch backend; install the [timesfm] extra")
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch",
        )
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=context,
                max_horizon=horizon,
                normalize_inputs=True,
            )
        )

    def predict(self, history: pd.DataFrame, dates: pd.Series) -> np.ndarray:
        point, _ = self.model.forecast(
            horizon=len(dates),
            inputs=[history["close"].to_numpy(dtype=np.float32)],
        )
        return np.asarray(point[0, : len(dates)])


class Kronos:
    name = "kronos-small"

    def __init__(self, seed: int = 42):
        import torch
        from model import Kronos as Network
        from model import KronosPredictor, KronosTokenizer

        self.torch = torch
        self.seed = seed
        try:
            network = Network.from_pretrained("NeoQuasar/Kronos-small")
            tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        except TypeError as exc:
            # The hub mixin swallows an unreachable config.json, then __init__ lacks arguments.
            raise OSError(f"Kronos checkpoint unavailable, check network/cache: {exc}") from exc
        # huggingface_hub 0.33.1, pinned by upstream, loads safetensors in train mode;
        # without eval() dropout stays active and can flip the long/cash signal.
        self.predictor = KronosPredictor(
            network.eval(), tokenizer.eval(), device="cpu", max_context=512
        )

    def predict(self, history: pd.DataFrame, dates: pd.Series) -> np.ndarray:
        self.torch.manual_seed(self.seed)
        result = self.predictor.predict(
            df=history[["open", "high", "low", "close", "volume"]].reset_index(drop=True),
            x_timestamp=history["date"].reset_index(drop=True),
            y_timestamp=dates.reset_index(drop=True),
            pred_len=len(dates),
            T=1.0,
            top_p=0.9,
            sample_count=5,
            verbose=False,
        )
        return result["close"].to_numpy()


def make_models(names: list[str], context: int, horizon: int, seed: int) -> list[Forecaster]:
    factories = {
        "seasonal-naive": lambda: SeasonalNaive(5),
        "random-walk": lambda: SeasonalNaive(1),
        "chronos": lambda: Chronos(seed),
        "timesfm": lambda: TimesFM(context, horizon),
        "kronos": lambda: Kronos(seed),
    }
    if len(names) != len(set(names)) or not names or set(names) - factories.keys():
        raise ValueError(f"Choose unique models from {', '.join(factories)}")
    if "kronos" in names and context > 512:
        raise ValueError("Kronos-small supports at most 512 context rows")
    try:
        return [factories[name]() for name in names]
    except ImportError as exc:
        raise RuntimeError(
            f"Optional model dependency missing ({exc}); see docs/models.md"
        ) from exc
