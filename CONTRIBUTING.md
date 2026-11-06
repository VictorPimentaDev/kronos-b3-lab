# Contributing

Start with one reproducible failure or one well-scoped improvement.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,app]'
ruff check .
ruff format --check .
pytest -q
```

Keep CI offline. Model integration tests must say whether they use actual weights
or test doubles. New predictors must not see future OHLCV; preserve the future
perturbation test. Keep reports explicit about data source, revisions, fees and
execution timing. Never use synthetic results to claim market skill.

Useful first contributions: a B3 calendar alignment helper with explicit dropped
session reporting; a timestamped, redistributable market fixture; a GPU integration
workflow that is manually dispatched and records model revisions. Open an issue
with the data license and expected behavior before adding a provider.

This repository's code is MIT. Respect the independent licenses of data and weights.
