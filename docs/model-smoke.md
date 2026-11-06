# Real model smoke guide

This is the shortest way to check that an adapter runs with real weights. It checks
compatibility, not forecasting skill: the data are the synthetic fixtures in this repo,
the windows are few, and the numbers below say nothing about B3 or about which model
is better.

All three checkpoints are public and ungated, so no login or token is needed. The
"unauthenticated requests" warning from Hugging Face is harmless. Use one virtual
environment per model: Kronos pins `huggingface_hub==0.33.1` and `pandas==2.2.2`,
and a shared environment was not tested.

| Model | Weights download | Peak RAM | Cached run, CPU | Windows |
|---|---|---|---|---|
| `chronos` | 34 MB | 0.5 GB | about 4 s | 10 |
| `kronos` | 115 MB (model + tokenizer) | 0.5 GB | about 5 s | 10 |
| `timesfm` | 925 MB | 2.1 GB | about 4 s | 6 |

Measured on an Apple Silicon Mac with `/usr/bin/time -l`, with weights cached and the
page cache warm; a cold start took up to 30 s. PyTorch and its dependencies are extra.
Start with Chronos if you only want one path.

## Chronos

```sh
python3.12 -m venv .venv-chronos && source .venv-chronos/bin/activate
pip install -e '.[chronos]'
b3-lab run --models seasonal-naive,random-walk,chronos --context 150 --horizon 3 --out runs/chronos
```

## TimesFM 2.5

```sh
python3.12 -m venv .venv-timesfm && source .venv-timesfm/bin/activate
pip install -e '.[timesfm]'
b3-lab run --models seasonal-naive,random-walk,timesfm --context 160 --horizon 3 --out runs/timesfm
```

Context is 160 because upstream left-pads to a multiple of 32, and those zeros also
enter its input normalization. The adapter does not change that. At one origin of
the fixture, contexts 150 and 160 gave terminal forecasts of 38.27 and 38.30 around a
last close of 38.29, so the long/cash signal flipped. That compares different inputs
and does not isolate the padding, but context is worth choosing deliberately.

## Kronos

Kronos has no PyPI package, so its code comes from a checkout. Read it before running.
The commit below is the one tested.

```sh
git clone https://github.com/shiyu-coder/Kronos.git ../Kronos
git -C ../Kronos checkout 67b630e67f6a18c9e9be918d9b4337c960db1e9a
python3.12 -m venv .venv-kronos && source .venv-kronos/bin/activate
pip install -e .
pip install -r ../Kronos/requirements.txt
PYTHONPATH="../Kronos" b3-lab run --models seasonal-naive,random-walk,kronos --context 150 --horizon 3 --out runs/kronos
```

## What a pass looks like

Exit code 0, a metrics row with the model's name and `forecasts` equal to 3 × windows,
plus `report.png`, `predictions.csv`, `metrics.csv`, `equity.csv` and `manifest.json`
in the output folder. Running the same command twice on one machine should give a
byte-identical `predictions.csv`; compare with `cmp`.

The manifest does not record checkpoint or Kronos revisions. For an experiment you
intend to publish, save them next to it:

```sh
cat ~/.cache/huggingface/hub/models--NeoQuasar--Kronos-small/refs/main
git -C ../Kronos rev-parse HEAD
```

## When it fails

Failures exit with code 2 and a one-line message. There is no silent fallback to a
baseline.

| Message contains | Cause | Fix |
|---|---|---|
| `No module named 'model'` | Kronos checkout not on the path | Run from the lab root with `PYTHONPATH="../Kronos"` |
| `No module named 'chronos'` or `'torch'` | Extra not installed | Install the extra in the active environment |
| `timesfm has no torch backend` | `timesfm` installed without torch | `pip install -e '.[timesfm]'` |
| `Kronos checkpoint unavailable` | Weights not cached and no network | Connect once, or check the cache |
| `locate the file on the Hub`, `Check your internet connection` | Same, for TimesFM or Chronos | Same |

`torch_dtype is deprecated` from Chronos under transformers 5 is only a warning: the
weights were confirmed to load as float32 on CPU.

## Evidence from 2026-11-06

Unedited output from the commands above is in
[`verification/smoke-2026-11-06`](verification/smoke-2026-11-06). The Kronos checkout
was in a temporary folder, so `PYTHONPATH` was an absolute path to the same commit.
Environments came from `uv venv --seed --python 3.12`, followed by the same `pip`
commands, because `python3.12` is not on this machine's `PATH`.
Environment: macOS on Apple Silicon, CPU only, Python 3.12.13, numpy 2.5.4, torch 2.15.0.

| Model | Checkpoint revision | Other packages | Ran twice |
|---|---|---|---|
| `chronos-t5-tiny` | `29d808298f1a` | chronos-forecasting 2.3.2, transformers 5.22.2, pandas 2.3.3 | identical |
| `timesfm-2.5-200m` | `1d952420fba8` | timesfm 2.0.2, huggingface_hub 2.4.0, pandas 2.3.3 | identical |
| `kronos-small` | `901c26c13326`, tokenizer `0e0117387f39` | Kronos `67b630e67f6a`, huggingface_hub 0.33.1, pandas 2.2.2 | identical |

Also run against the real packages:

- Upstream's `tests/test_kronos_regression.py` at that commit, using the same
  revisions: 4 passed. That test belongs to Kronos and does not exercise this
  lab's adapter.
- `tests/test_model_contracts.py` in each environment, so its upstream signature
  checks ran instead of skipping.
- Dropout state after the adapter loads each model. Active layers: Chronos 0 of 30,
  TimesFM has none, and Kronos none once `eval()` is called.

### The bug this found

Doubles had passed while the Kronos adapter forecast with dropout on. The version of
`huggingface_hub` that upstream pins leaves safetensors checkpoints in train mode, and
Kronos-small uses dropout up to 0.25. At the first fixture origin with seed 42, the
terminal forecast moved from 36.37 to 37.13 around a last close of 36.39, turning a
cash signal into a long one. The adapter now calls `eval()`. Over the 10 windows, MAE
went from 0.3716 to 0.4011. The fix was not chosen for its result: 30 forecasts on
synthetic data cannot support a claim either way.

## Still not verified

- Any real market data with these weights, or a fresh chronological holdout.
- GPU, Linux, Windows and Python 3.11. CI runs Python 3.11 and 3.12 with doubles only.
- Newer upstream releases. The extras allow them, and the code does not pin
  checkpoint revisions.
- Licensing for your use. Hugging Face tags list MIT for both Kronos repos and
  Apache-2.0 for the other two; read the model cards before relying on that.
