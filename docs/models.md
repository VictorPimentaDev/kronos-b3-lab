# Optional model integrations

The adapters are original integration code. They call upstream libraries; no model
implementation or weights are vendored. Sources checked on 2026-11-06:

| CLI name | Checkpoint | Input | Integration |
|---|---|---|---|
| `kronos` | `NeoQuasar/Kronos-small` + `NeoQuasar/Kronos-Tokenizer-base` | past OHLCV + known future session dates | `KronosPredictor.predict` |
| `timesfm` | `google/timesfm-2.5-200m-pytorch` | past close only | `TimesFM_2p5_200M_torch.forecast` |
| `chronos` | `amazon/chronos-t5-tiny` | past close only | `ChronosPipeline.predict`, median of 20 samples |

These are particular model versions, not a claim to support every release. Kronos
gets richer inputs than the two univariate adapters; treat this as a comparison
of complete forecasting systems, not a controlled architecture benchmark.
The CPU paths avoid a mandatory GPU. Context defaults to 60 sessions; Kronos-small
is limited to 512 and this lab caps every horizon at 64.

## Chronos: small real inference path

Python 3.11 or 3.12 is recommended for optional machine-learning dependencies.
From the lab checkout:

```sh
pip install -e '.[chronos]'
b3-lab run --models seasonal-naive,random-walk,chronos --context 150 --horizon 3 --out runs/chronos
```

This downloads weights from Hugging Face. The fixture contains 180 observations,
so this example runs ten test windows. It remains a **synthetic-data experiment**
even with real weights. The seed is reset to 42 at each origin; GPU/backend or
library changes can still change results. Use `--seed` to repeat a sensitivity study,
not to cherry-pick the best result.

[Official Chronos source/API](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos.py)
· [checkpoint and license](https://huggingface.co/amazon/chronos-t5-tiny).

## TimesFM 2.5

```sh
pip install -e '.[timesfm]'
b3-lab run --models seasonal-naive,random-walk,timesfm --context 150 --horizon 3 --out runs/timesfm
```

The package range deliberately stays on `timesfm>=2,<3`; the adapter targets the
2.5 checkpoint. It compiles the upstream forecast configuration once and reuses
loaded weights. The upstream code and 2.5 weights use Apache-2.0; TimesFM 3 weights
have a different, restrictive license and are not used here.
[Official API](https://github.com/google-research/timesfm/blob/master/timesfm-forecasting/references/api_reference.md)
· [license notice](https://github.com/google-research/timesfm#license-notice-for-pretrained-weights).

## Kronos

Upstream uses a top-level `model` package, not an ordinary PyPI package named
`kronos`. Clone it outside this repository and expose its root on `PYTHONPATH`:

```sh
git clone https://github.com/shiyu-coder/Kronos.git ../Kronos
pip install -r ../Kronos/requirements.txt
PYTHONPATH="../Kronos" b3-lab run --models seasonal-naive,random-walk,kronos --context 150 --horizon 3 --out runs/kronos
```

Run from the lab root, record `git -C ../Kronos rev-parse HEAD`, and inspect upstream
requirements before installing. The adapter uses five sampled paths, temperature
1.0, top-p 0.9 and CPU. It passes actual evaluation session dates, never fabricated
weekday calendars or future candles. It does not invent the optional turnover
`amount` field. Source MIT does not automatically cover every checkpoint; review
[upstream](https://github.com/shiyu-coder/Kronos) and the
[model card](https://huggingface.co/NeoQuasar/Kronos-small) for your intended use.

## Verification boundary

CI exercises contracts with explicit test doubles and never downloads model
weights. `manifest.json` records package versions, parameters and input hashes.
Upstream revisions and Hugging Face revisions are not pinned automatically: record
resolved revisions and hardware separately for a publishable experiment. A mock
contract pass is not proof of compatibility with every upstream version, and no
reported demo metric establishes real-market forecasting skill.
