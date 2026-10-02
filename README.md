# Milan 2026 — ML for Algorithmic Trading

## Final submission

The submission exposes exactly the API used by the starter:

```python
from model import Model

m = Model()
m.train(X, y)
signals = m.predict(X_history)
```

`predict()` returns one continuous value per asset at the current timestamp, in sorted `asset_id` order, bounded to `[-1, 1]`. Inactive assets receive zero.

## Final architecture

The final model is a **single CatBoost regressor trained on a medium-horizon, execution-aligned target**.

- The official evaluator-provided `y` is accepted to preserve the required API, but it is not used as a feature or manually blended into the final signal.
- The trained target is a **causal five-step forward open-to-open return constructed inside `train()` from historical training OHLCV only**.

The auxiliary target for row `t` is:

```text
after-signal execution open = Open(t+1)
5-step holding target       = Open(t+6) / Open(t+1) - 1
```

This target is never passed as an input feature and is only constructed from historical training data. At prediction time, no future observation is accessed.

The 5-step target was consistently stronger and more stable than direct one-step proxy prediction in the research sweep, including on a separate public MSFT finance check. Removing the hand-set ensemble weight also makes the final submission simpler and easier to audit against the competition's ensemble-combination guidance.

## Causal features

The model uses a shared feature representation across all active assets. Feature groups include:

- 1/2/3/5/10/20/30/40/60/90-period returns and log returns
- OHLC candle geometry and gaps
- lagged returns and volume changes
- rolling volatility, mean, median, absolute returns, skew and kurtosis
- EMA distance and price-to-rolling-mean distance
- rolling volume statistics and volume z-scores
- EWMA returns and volatility
- cross-sectional mean/std/z-score/rank/deviation features using only the current timestamp
- market/common-factor returns and idiosyncratic returns
- rolling beta to the common factor
- calendar sin/cos descriptors

Maximum lookback is 90 observations. `predict()` reconstructs features only from the most recent 128 timestamps, so repeated evaluation does not rebuild the entire historical dataset at every timestep.

## Model configuration

CatBoost configuration:

```text
iterations      = 120
depth            = 5
learning_rate   = 0.03
l2_leaf_reg     = 8
loss_function    = RMSE
CPU-only
```

Signals are bounded with a training-derived 99.5th-percentile scale followed by clipping to `[-1, 1]`. This is a mathematical output bound, not a manually designed trading threshold.

## Rule-compliance

- No random train/test splitting.
- No future/test observations used as features.
- No manually coded buy/sell rules.
- No hard-coded asset allocations.
- No manually assigned asset ranking.
- No manually selected trading threshold.
- Cross-sectional ranks are **data-derived features**, computed independently at each timestamp; they are not used as hard-coded portfolio rankings.
- Signals come from trained ML models.
- Predictions are continuous and bounded to `[-1, 1]`.
- Inactive assets are forced to zero because they are non-tradable.
- No external dataset or internet access is required by the submission.

## Local smoke test

```bash
pip install -r requirements.txt
python train.py --data path/to/training.csv
```

The competition runner should import `model.py` from the submission root. No change to the official class name or method signatures is required. `train.py` is only a local smoke-test helper; when no target column is supplied it uses a dummy `y` because the competition evaluator passes `y` directly to `Model.train(X, y)`.


API robustness note: `Model.train(X, y)` accepts the competition target argument for interface compatibility; the submitted model does not depend on the supplied `y` and constructs its execution-aligned training label from historical `X`.
